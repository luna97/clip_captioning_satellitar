import torch
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
import clip
from torch.utils.data import DataLoader
from transformers import GPT2Config, GPT2LMHeadModel, get_linear_schedule_with_warmup, AutoTokenizer, T5ForConditionalGeneration, T5Config, LlamaConfig, AutoModelForCausalLM, AutoTokenizer
from torchrs.datasets import RSICD
import torch.nn as nn
from dataset import get_datasets
from model import ClipGPT
from torchmetrics.text.bleu import BLEUScore


device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f'Using device: {device}')

net = ClipGPT(device=device, generator='gpt2').to(device)

dataset_train, dataset_val, dataset_test = get_datasets(net.preprocess_clip)

print(f'Length of train dataset: {len(dataset_train)}')
print(f'Length of val dataset: {len(dataset_val)}')
print(f'Length of test dataset: {len(dataset_test)}')

def collate_fn(batch):
    """
    select a random caption from each image
    """
    images = [item['x'] for item in batch]
    # get a random caption from each image
    random_index = [np.random.randint(0, len(item['captions'])) for item in batch]
    captions = [ item['captions'][random_index[i]].replace('.', '').strip()
                for i, item in enumerate(batch)]
    return torch.stack(images), captions

batch_size = 16
dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
dataloader_val_rand = DataLoader(dataset_val, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
dataloader_val= DataLoader(dataset_val, batch_size=batch_size, shuffle=True)
dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)


# train clip in a contrastive way with the captions
import torch.nn as nn
from tqdm import tqdm

### ------------ Train Clip ------------ ###


# load clip model if exists
if os.path.exists('data/models/clip.pth'):
    # load on device
    net.clip.load_state_dict(torch.load('data/models/clip.pth', map_location=device))
else:
    def clip_loss(image_features, text_features):
        loss_img = nn.CrossEntropyLoss()
        loss_txt = nn.CrossEntropyLoss()

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logits = torch.matmul(image_features, text_features.T)
        target = torch.arange(image_features.shape[0]).to(device)

        loss = (loss_img(logits, target) + loss_txt(logits.T, target)) / 2
        return loss
    
    clip_epochs = 10
    optimizer_clip = torch.optim.AdamW(net.clip.parameters(), lr=1e-6, weight_decay=0.0000001)
    sched_clip = get_linear_schedule_with_warmup(
        optimizer_clip,
        num_warmup_steps=len(dataloader_train) * clip_epochs * 0.005,
        num_training_steps=len(dataloader_train) * clip_epochs
    )

    #train clip justo for one epoch
    train_pbar = tqdm(range(clip_epochs))
    best_val_loss = np.inf
    for epoch in range(clip_epochs):
        net.train()
        epoch_bar = tqdm(dataloader_train, total=len(dataloader_train), leave=False, desc=f'Epoch {epoch}/{clip_epochs}')
        train_losses = []
        for images, captions in epoch_bar:
            images = images.to(device)
            image_features, text_features = net.train_clip(images, captions)

            loss = clip_loss(image_features, text_features)
            loss.backward()
            optimizer_clip.step()
            optimizer_clip.zero_grad()
            sched_clip.step()

            epoch_bar.set_postfix(loss_clip=loss.item())
            train_losses.append(loss.item())

        net.eval()
        eval_losses = []
        with torch.no_grad():
            val_pbar = tqdm(dataloader_val_rand, total=len(dataloader_val_rand), leave=False, desc=f'Validation')
            for images, captions in val_pbar:
                images = images.to(device)
                image_features, text_features = net.train_clip(images, captions)
                loss = clip_loss(image_features, text_features)
                eval_losses.append(loss.item())
                if loss < best_val_loss:
                    best_val_loss = loss
                    torch.save(net.clip.state_dict(), 'data/models/clip.pth')

        train_pbar.set_postfix(train_loss_clip=np.mean(train_losses), val_loss_clip=np.mean(eval_losses))


### ------------ Train Decoder ------------ ###

epochs = 10
lr=1e-4
optimizer_gen = torch.optim.AdamW([
    { 'params': net.generator.parameters(), 'lr': 1e-5},
    # { 'params': net.head.parameters(), 'lr': lr},
    { 'params': net.adapted_layer.parameters(), 'lr': 1e-3}
], lr=lr, weight_decay=0.0000001)

sched_gpt = get_linear_schedule_with_warmup(
    optimizer_gen,
    num_warmup_steps=len(dataloader_train) * epochs * 0.005, 
    num_training_steps=len(dataloader_train) * epochs
)
best_score = 0
bleu_scorer = BLEUScore()
train_pbar = tqdm(range(epochs), desc='Decoder training', leave=True)
for epoch in train_pbar:
    net.train()
    train_losses = []
    epoch_bar = tqdm(dataloader_train, total=len(dataloader_train), desc=f'Epoch {epoch}/{epochs}', leave=False)
    for images, captions in epoch_bar:
        images = images.to(device)
        loss = net.train_generator(captions, images)
        loss.backward()
        train_losses.append(loss.item())
        optimizer_gen.step()
        optimizer_gen.zero_grad()
        sched_gpt.step()

        epoch_bar.set_postfix(loss_gpt=loss.item())

    # evaluate the model
    net.eval()
    scores = []
    with torch.no_grad():
        val_pbar = tqdm(dataloader_val, total=len(dataloader_val), leave=False, desc=f'Validation {epoch}/{epochs}')
        for batch in val_pbar:
            images = batch['x'].to(device)
            captions = batch['captions']

            images = images.to(device)
            res = net.get_caption(images)

            # reshape captions from [num_captions, batch size] to [batch size, num_captions]
            captions = [ [captions[j][i] for j in range(len(captions))] for i in range(images.shape[0])]

            bleu_score = bleu_scorer(res, captions)
            scores.append(bleu_score)

            # save the model if the score is better
            if bleu_score > best_score:
                best_score = bleu_score
                torch.save(net.generator.state_dict(), 'data/models/full.pth')

    train_pbar.set_postfix(train_loss_gpt=np.mean(train_losses), val_bleu=np.mean(scores))
            