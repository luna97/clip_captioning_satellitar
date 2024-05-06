import torch
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from torch.utils.data import DataLoader
from transformers import GPT2Config, GPT2LMHeadModel, get_linear_schedule_with_warmup, AutoTokenizer, T5ForConditionalGeneration, T5Config, LlamaConfig, AutoModelForCausalLM, AutoTokenizer
from torchrs.datasets import RSICD
import torch.nn as nn
from dataset import get_datasets
from model import ClipGPT
from torchmetrics.text.bleu import BLEUScore
from tqdm import tqdm
import clip
import argparse

# Add argument parser for hyperparameters
parser = argparse.ArgumentParser(description='Train Decoder')
parser.add_argument('--device', type=str, default=None, help='Device to use for training')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
parser.add_argument('--lr_gen', type=float, default=1e-6, help='Learning rate for generator')
parser.add_argument('--lr_adapter', type=float, default=1e-4, help='Learning rate for adapted layer')
parser.add_argument('--weight_decay', type=float, default=0.0000001, help='Weight decay for optimizer')
parser.add_argument('--warmup_ratio', type=float, default=0.005, help='Number of warmup ratio for scheduler')
args = parser.parse_args()

device = args.device if args.device else 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f'Using device: {device}')

net = ClipGPT(device=device, generator='gpt2').to(device)

# assert path exists
assert os.path.exists('data/models/clip.pth')

net.clip.load_state_dict(torch.load('data/models/clip.pth', map_location=device))


dataset_train, dataset_val = get_datasets(net.preprocess_clip)

print(f'Length of train dataset: {len(dataset_train)}')
print(f'Length of val dataset: {len(dataset_val)}')

def collate_fn(batch):
    """
    I need only the text, not the images
    """
    # get a random caption from each image
    random_index = [np.random.randint(0, len(item['captions'])) for item in batch]
    captions = [ item['captions'][random_index[i]].replace('.', '').strip() for i, item in enumerate(batch)]
    return captions

batch_size = args.batch_size
dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
 
epochs = args.epochs
optimizer_gen = torch.optim.AdamW([
    { 'params': net.generator.parameters(), 'lr': args.lr_gen},
    { 'params': net.adapted_layer.parameters(), 'lr': args.lr_adapter}
], weight_decay=args.weight_decay)

sched_gpt = get_linear_schedule_with_warmup(
    optimizer_gen,
    num_warmup_steps=len(dataloader_train) * epochs * args.warmup_ratio, 
    num_training_steps=len(dataloader_train) * epochs
)
best_score = 0
bleu_scorer = BLEUScore()
train_pbar = tqdm(range(epochs), desc='Decoder training', leave=True)
for epoch in train_pbar:
    net.train()
    train_losses = []
    epoch_bar = tqdm(dataloader_train, total=len(dataloader_train), desc=f'Epoch {epoch}/{epochs}', leave=False)
    for captions in epoch_bar:
        loss = net.train_generator(captions)
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
        for captions in val_pbar:
            clip_embedding = net.clip.encode_text(clip.tokenize(captions).to(device))
            res = net.get_caption(clip_embedding)

            bleu_score = bleu_scorer(res, captions)
            scores.append(bleu_score)

        # save the model if the score is better
        if np.mean(scores) > best_score:
            best_score = np.mean(scores)
            torch.save(net.state_dict(), 'data/models/full.pth')


    train_pbar.set_postfix(train_loss_gpt=np.mean(train_losses), val_bleu=np.mean(scores))