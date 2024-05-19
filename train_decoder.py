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
from pycocoevalcap.spice.spice import Spice
import wandb


# Add argument parser for hyperparameters
parser = argparse.ArgumentParser(description='Train Decoder')
parser.add_argument('--device', type=str, default=None, help='Device to use for training')
parser.add_argument('--epochs', type=int, default=15, help='Number of training epochs')
parser.add_argument('--lr_gen', type=float, default=1e-4, help='Learning rate for generator')
parser.add_argument('--lr_adapter', type=float, default=3e-4, help='Learning rate for adapted layer')
parser.add_argument('--weight_decay', type=float, default=0., help='Weight decay for optimizer')
parser.add_argument('--warmup_steps', type=float, default=200, help='Number of warmup ratio for scheduler')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
parser.add_argument('--use_remote_clip', action='store_true', help='Use remote clip model')
parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate for GPT2')
parser.add_argument('--datasets', type=str, default=None, help='Dataset to use for training')
parser.add_argument('--log', action='store_true', help='Log to wandb')
args = parser.parse_args()

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
    images = [item['x'] for item in batch]
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
best_spice = 0
spice_scorer = Spice()

if args.log:
    wandb.init(project='clip_captioning_satellitar')
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
    # bleu_scorer = Bleu(n=4)

    with torch.no_grad():
        val_pbar = tqdm(dataloader_val, total=len(dataloader_val), leave=False, desc=f'Validation {epoch}/{epochs}')
        for b, (images, captions) in enumerate(val_pbar):
            # generate captions
            images = images.to(device)
            results = net.get_caption(images)

            for c in range(len(results)):
                refs[count + c] = captions[c]
                res[count + c] = [results[c]]

            count += len(results)

        # bleu_score = bleu_scorer.compute_score(refs, res, verbose=False)[0][3]
        spice_score, _ = spice_scorer.compute_score(refs, res)

        if spice_score > best_spice:
            best_spice = spice_score
            print(f'Saving model with SPICE score: {spice_score}')
            torch.save(net.state_dict(), 'data/models/full.pth')

    train_pbar.set_postfix(train_loss_gpt=np.mean(train_losses), val_spice=spice_score)

    # Log metrics to wandb
    if args.log:
        wandb.log({"train_loss_gpt": np.mean(train_losses), "val_spice": spice_score})

print('Finished training decoder')
if args.log:
    wandb.finish()

