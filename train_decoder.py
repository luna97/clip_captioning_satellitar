import torch
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
from torchrs.datasets import RSICD
import torch.nn as nn
from dataset import get_datasets
from model import ClipGPT
from torchmetrics.text.bleu import BLEUScore
from tqdm import tqdm
import clip
import argparse
from huggingface_hub import hf_hub_download
from nltk.translate.meteor_score import meteor_score as Meteor
from pycocoevalcap.bleu.bleu import Bleu

# Add argument parser for hyperparameters
parser = argparse.ArgumentParser(description='Train Decoder')
parser.add_argument('--device', type=str, default=None, help='Device to use for training')
parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
parser.add_argument('--lr_gen', type=float, default=1e-6, help='Learning rate for generator')
parser.add_argument('--lr_adapter', type=float, default=2e-5, help='Learning rate for adapted layer')
parser.add_argument('--weight_decay', type=float, default=1e-6, help='Weight decay for optimizer')
parser.add_argument('--warmup_steps', type=float, default=100, help='Number of warmup ratio for scheduler')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
parser.add_argument('--use_remote_clip', action='store_true', help='Use remote clip model')
parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate for GPT2')
parser.add_argument('--datasets', type=str, default='nwpu,', help='Dataset to use for training')
args = parser.parse_args()

augmentation = T.Compose([
    T.RandomAdjustSharpness(sharpness_factor=2),
    T.RandomAutocontrast(),
    T.RandomAffine(degrees=10, translate=(0.1, 0.1)),
])

device = args.device if args.device else 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f'Using device: {device}')

net = ClipGPT(device=device, generator='gpt2', dropout=args.dropout).to(device)

if args.use_remote_clip:
    checkpoint_path = hf_hub_download("chendelong/RemoteCLIP", "RemoteCLIP-ViT-B-32.pt", cache_dir='checkpoints')
    net.clip.load_state_dict(torch.load(checkpoint_path, map_location=device))
else:
   # assert path exists
    assert os.path.exists('data/models/clip.pth')
    net.clip.load_state_dict(torch.load('data/models/clip.pth', map_location=device))

datasets = args.datasets.split(',') if args.datasets else ['rsicd', 'ucm', 'nwpu', 'sidney']

dataset_train, dataset_val = get_datasets(net.preprocess_clip, datasets)

# Add argument for batch size

print(f'Length of train dataset: {len(dataset_train)}')
print(f'Length of val dataset: {len(dataset_val)}')

def collate_fn_train(batch):
    """
    select a random caption from each image
    """
    images = [augmentation(item['x']) for item in batch]
    # get a random caption from each image
    random_index = [np.random.randint(0, len(item['captions'])) for item in batch]
    captions = [ item['captions'][random_index[i]]
                for i, item in enumerate(batch)]
    return torch.stack(images), captions

def collate_fn_val(batch):
    """
    select a random caption from each image
    """
    images = [item['x'] for item in batch]
    # get a random caption from each image
    captions = [ item['captions'] for item in batch]
    return torch.stack(images), captions

batch_size = args.batch_size
dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_train)
dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_val)
 
epochs = args.epochs
optimizer_gen = torch.optim.AdamW([
    { 'params': net.generator.parameters(), 'lr': args.lr_gen},
    # { 'params': [ p for name, p in net.generator.named_parameters() if 'bias' not in name], 'lr': args.lr_gen},
    { 'params': net.adapted_layer.parameters(), 'lr': args.lr_adapter}
], weight_decay=args.weight_decay)

sched_gpt = get_cosine_schedule_with_warmup(
    optimizer_gen,
    num_warmup_steps=args.warmup_steps, 
    num_training_steps=len(dataloader_train) * epochs
)
best_meteor = 0
best_bleu = 0
bleu_scorer = BLEUScore(n_gram=4)
train_pbar = tqdm(range(epochs), desc='Decoder training', leave=True)
for epoch in train_pbar:
    net.train()
    train_losses = []
    epoch_bar = tqdm(dataloader_train, total=len(dataloader_train), desc=f'Epoch {epoch}/{epochs}', leave=False)
    for images, captions in epoch_bar:
        images = images.to(device)
        loss = net.train_generator(captions, images=images)
        loss.backward()
        train_losses.append(loss.item())
        optimizer_gen.step()
        optimizer_gen.zero_grad()
        sched_gpt.step()
        epoch_bar.set_postfix(loss_gpt=loss.item())

    # evaluate the model
    net.eval()

    refs = {}
    res = {}
    count = 0
    bleu_scorer = Bleu(n=4)

    with torch.no_grad():
        val_pbar = tqdm(dataloader_val, total=len(dataloader_val), leave=False, desc=f'Validation {epoch}/{epochs}')
        for b, (images, captions) in enumerate(val_pbar):
            # generate captions
            images = images.to(device)
            clip_embeddings = net.clip.encode_image(images)
            results = net.get_caption(clip_embeddings)

            for c in range(len(results)):
                refs[count + c] = captions[c]
                res[count + c] = [results[c]]

            count += len(results)

        # meteor_score = Meteor(refs, res)
        bleu_score = bleu_scorer.compute_score(refs, res, verbose=False)[0][3]

        # save the model if the score is better
        #if meteor_score > best_meteor:
        #   best_meteor = meteor_score
        #    torch.save(net.state_dict(), 'data/models/full.pth')
        #elif meteor_score == best_meteor and bleu_score > best_bleu:
        #    torch.save(net.state_dict(), 'data/models/full.pth')

        if bleu_score > best_bleu:
            best_bleu = bleu_score
            print(f'Saving model with BLEU score: {bleu_score}')
            torch.save(net.state_dict(), 'data/models/full.pth')

    train_pbar.set_postfix(train_loss_gpt=np.mean(train_losses), val_meteor=bleu_score)