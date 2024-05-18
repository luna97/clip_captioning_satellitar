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
from dataset import get_datasets, get_test_datasets
from model import ClipGPT
from torchmetrics.text.bleu import BLEUScore
import torch.nn as nn
from tqdm import tqdm
import argparse
from torch.cuda.amp import GradScaler, autocast
import wandb

# Add argument parser for hyperparameters
parser = argparse.ArgumentParser(description='Train ClipGPT model')
parser.add_argument('--device', type=str, default=None, help='Device to use for training (cuda, mps, cpu)')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for training')
parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate for optimizer')
parser.add_argument('--weight_decay', type=float, default=0.2, help='Weight decay for optimizer')
parser.add_argument('--eps', type=float, default=1e-6, help='Epsilon for optimizer')
parser.add_argument('--warmup_steps', type=float, default=100, help='Number of warmup ratio for scheduler')
parser.add_argument('--log', action='store_true', help='Log to wandb')
args = parser.parse_args()

device = args.device if args.device else 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f'Using device: {device}')

net = ClipGPT(device=device, generator='gpt2').to(device)

dataset_train, dataset_val = get_datasets(net.preprocess_clip)
dataset_test = get_test_datasets(net.preprocess_clip)

print(f'Length of train dataset: {len(dataset_train)}')
print(f'Length of val dataset: {len(dataset_val)}')

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

batch_size = args.batch_size
dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

# train clip in a contrastive way with the captions
def clip_loss(image_features, text_features):
    """
    contrastive loss between image and text features. 
    """
    
    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()

    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    logits = torch.matmul(image_features, text_features.T)
    target = torch.arange(image_features.shape[0]).to(device)

    loss = (loss_img(logits, target) + loss_txt(logits.T, target)) / 2
    return loss

epochs = args.epochs
optimizer = torch.optim.AdamW(net.clip.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.98), eps=args.eps)
scaler = GradScaler()

sched_clip = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=args.warmup_steps,
    num_training_steps=len(dataloader_train) * epochs
)

if device == 'cuda':
    scaler = GradScaler()

if args.log:
    wandb.init(project='clip_captioning_satellitar')

train_pbar = tqdm(range(epochs))
best_val_loss = np.inf
for epoch in train_pbar:
    net.train()
    epoch_bar = tqdm(dataloader_train, total=len(dataloader_train), desc=f'Epoch {epoch}/{epochs}')
    train_losses = []
    for images, captions in epoch_bar:
        images = images.to(device)

        with autocast():
            image_features, text_features = net.train_clip(images, captions)

            loss = clip_loss(image_features, text_features)

            # Backpropagation with gradient scaling
            if device == 'cuda':
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            sched_clip.step()

        epoch_bar.set_postfix(loss_clip=loss.item())
        train_losses.append(loss.item())

    net.eval()
    eval_losses = []
    with torch.no_grad():
        val_pbar = tqdm(dataloader_val, total=len(dataloader_val), desc=f'Validation')
        for images, captions in val_pbar:
            images = images.to(device)
            image_features, text_features = net.train_clip(images, captions)
            loss = clip_loss(image_features, text_features)
            eval_losses.append(loss.item())
        if np.mean(eval_losses) < best_val_loss:
            best_val_loss = np.mean(eval_losses)
            torch.save(net.clip.state_dict(), 'data/models/clip.pth')
            
    train_pbar.set_postfix(train_loss_clip=np.mean(train_losses), val_loss_clip=np.mean(eval_losses))
    
    if args.log:
        wandb.log({'train_loss_clip': np.mean(train_losses), 'val_loss_clip': np.mean(eval_losses)})

print('Finished training clip')
if args.log:
    wandb.finish()