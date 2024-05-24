import torch
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
import torch.nn as nn
from dataset import get_datasets, RSICD_, UCM, NWPU, SIDNEY
from model import ClipGPT, CLIP, REMOTE_CLIP, VGG
import argparse
from huggingface_hub import hf_hub_download
from pycocoevalcap.spice.spice import Spice
from pycocoevalcap.cider.cider import Cider
import wandb
from utils import collate_fn_train, collate_fn_val
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm


# Add argument parser for hyperparameters
parser = argparse.ArgumentParser(description='Train Decoder')
parser.add_argument('--device', type=str, default=None, help='Device to use for training')
parser.add_argument('--epochs', type=int, default=15, help='Number of training epochs')
parser.add_argument('--lr_gen', type=float, default=1e-5, help='Learning rate for generator')
parser.add_argument('--lr_adapter', type=float, default=3e-5, help='Learning rate for adapted layer')
parser.add_argument('--weight_decay', type=float, default=0., help='Weight decay for optimizer')
parser.add_argument('--warmup_steps', type=float, default=200, help='Number of warmup ratio for scheduler')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
parser.add_argument('--use_remote_clip', action='store_true', help='Use remote clip model')
parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate for GPT2')
parser.add_argument('--dataset', type=str, default=None, help='Dataset to use for training')
parser.add_argument('--log', action='store_true', help='Log to wandb')
parser.add_argument('--encoder', type=str, default=CLIP, help='Decoder model to use')
parser.add_argument('--lr_encoder', type=float, default=1e-5, help='Learning rate for decoder')
# path of datasets
parser.add_argument('--rsicd_path', type=str, default='data/rsicd/', help='Path to RSICD dataset')
parser.add_argument('--ucm_path', type=str, default='data/ucm/', help='Path to UCM dataset')
parser.add_argument('--nwpu_path', type=str, default='data/nwpu/', help='Path to NWPU dataset')
parser.add_argument('--sydney_path', type=str, default='data/sidney', help='Path to SIDNEY dataset')

args = parser.parse_args()

assert args.encoder in [CLIP, REMOTE_CLIP, VGG]
assert args.dataset is None or args.dataset in [RSICD_, UCM, NWPU, SIDNEY]

# load device
device = args.device if args.device else 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f'Using device: {device}')

# load model
net = ClipGPT(device=device, encoder=args.encoder, dropout=args.dropout).to(device)

# load datasets
datasets = args.dataset.split(',') if args.dataset else ['rsicd', 'ucm', 'nwpu', 'sidney']
kwargs = {'rsicd_path': args.rsicd_path, 'ucm_path': args.ucm_path, 'nwpu_path': args.nwpu_path, 'sydney_path': args.sydney_path}
dataset_train, dataset_val = get_datasets(net.preprocess, datasets, **kwargs)
dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn_train)
dataloader_val = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn_val)
 
if args.encoder == VGG:
    optimizer = torch.optim.AdamW([
        { 'params': net.generator.parameters(), 'lr': args.lr_gen},
        { 'params': net.adapted_layer.parameters(), 'lr': args.lr_adapter},
        { 'params': net.vgg.features.parameters(), 'lr': args.lr_encoder}
    ], weight_decay=args.weight_decay)
else:
    optimizer = torch.optim.AdamW([
        { 'params': net.generator.parameters(), 'lr': args.lr_gen},
        { 'params': net.adapted_layer.parameters(), 'lr': args.lr_adapter}
    ], weight_decay=args.weight_decay)

sched_gpt = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=args.warmup_steps, 
    num_training_steps=len(dataloader_train) * args.epochs
)

# store best results
best_score = 0
spice_scorer = Spice()
cider_scorer = Cider()

if device == 'cuda':
    scaler = GradScaler()

if args.log: wandb.init(project='clip_captioning_satellitar', config=args)

train_pbar = tqdm(range(args.epochs), desc='Decoder training', leave=True)
for epoch in train_pbar:
    net.train()
    train_losses = []
    epoch_bar = tqdm(dataloader_train, total=len(dataloader_train), desc=f'Epoch {epoch}/{args.epochs}', leave=False)
    for images, captions in epoch_bar:
        images = images.to(device)
        with autocast():
            loss = net.train_generator(captions, images=images)

            # Backpropagation with gradient scaling
            if device == 'cuda':
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            train_losses.append(loss.item())
            sched_gpt.step()
            epoch_bar.set_postfix(loss_gpt=loss.item())    

    # evaluate the model
    net.eval()
    refs, res = {}, {}
    count = 0
    with torch.no_grad():
        val_pbar = tqdm(dataloader_val, total=len(dataloader_val), leave=False, desc=f'Validation {epoch}/{args.epochs}')
        for b, (images, captions) in enumerate(val_pbar):
            # generate captions
            images = images.to(device)
            results = net.get_caption(images)

            for c in range(len(results)):
                refs[count + c] = captions[c]
                res[count + c] = [results[c]]
            count += len(results)

        # non mi va spice sul mac
        if device == 'mps':
            score, _ = cider_scorer.compute_score(refs, res)
        else:
            score, _ = spice_scorer.compute_score(refs, res)

        if score > best_score:
            best_score = score
            print(f'Saving model with best score: {score}')
            torch.save(net.state_dict(), 'data/models/full.pth')

    train_pbar.set_postfix(train_loss_gpt=np.mean(train_losses), val_spice=score)

    if args.log: wandb.log({"train_loss_gpt": np.mean(train_losses), "val_spice": score})

print('Finished training decoder')
if args.log: wandb.finish()

