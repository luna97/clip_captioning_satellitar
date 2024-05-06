import torch
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from torch.utils.data import DataLoader
from transformers import GPT2Config, GPT2LMHeadModel, get_cosine_schedule_with_warmup
from torchrs.datasets import RSICD
import torch.nn as nn
from dataset import get_datasets
from model import ClipGPT
from torchmetrics.text.bleu import BLEUScore
import torch.nn as nn
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f'Using device: {device}')

net = ClipGPT(device=device, generator='gpt2').to(device)

dataset_train, dataset_val = get_datasets(net.preprocess_clip)

print(f'Length of train dataset: {len(dataset_train)}')
print(f'Length of val dataset: {len(dataset_val)}')

augmentation = T.Compose([
    T.RandomResizedCrop(224, scale=(0.9, 1.0)),
    T.RandomHorizontalFlip(),
    T.RandomRotation(10),
    T.RandomAdjustSharpness(sharpness_factor=2),
    T.RandomAutocontrast()
])

def collate_fn(batch):
    """
    select a random caption from each image
    """
    images = [augmentation(item['x']) for item in batch]
    # get a random caption from each image
    random_index = [np.random.randint(0, len(item['captions'])) for item in batch]
    captions = [ item['captions'][random_index[i]].replace('.', '').strip()
                for i, item in enumerate(batch)]
    return torch.stack(images), captions

batch_size = 16
dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

# train clip in a contrastive way with the captions
def clip_loss(image_features, text_features):
    """
    contrastive loss between image and text features. 
    """
    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss(label_smoothing=0.1)

    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    logits = torch.matmul(image_features, text_features.T)
    target = torch.arange(image_features.shape[0]).to(device)

    loss = (loss_img(logits, target) + loss_txt(logits.T, target)) / 2
    return loss

clip_epochs = 20
optimizer_clip = torch.optim.AdamW(net.clip.parameters(), lr=1e-5, weight_decay=0.01
                                   )
sched_clip = get_cosine_schedule_with_warmup(
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
        val_pbar = tqdm(dataloader_val, total=len(dataloader_val), leave=False, desc=f'Validation')
        for images, captions in val_pbar:
            images = images.to(device)
            image_features, text_features = net.train_clip(images, captions)
            loss = clip_loss(image_features, text_features)
            eval_losses.append(loss.item())
        if np.mean(eval_losses) < best_val_loss:
            best_val_loss = np.mean(eval_losses)
            torch.save(net.clip.state_dict(), 'data/models/clip.pth')

    train_pbar.set_postfix(train_loss_clip=np.mean(train_losses), val_loss_clip=np.mean(eval_losses))

print('Finished training clip')