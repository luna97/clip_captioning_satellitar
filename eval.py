# load model
import torch
from model import ClipGPT
from dataset import NWPUCaptions, SidneyCaptions
from torch.utils.data import DataLoader
from torchrs.datasets import RSICD, UCMCaptions
import clip
import os
from torchvision import transforms as T
from tqdm import tqdm
from pycocoevalcap.cider.cider import Cider
# from pycocoevalcap.meteor.meteor import Meteor
from nltk.translate.meteor_score import meteor_score as Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.spice.spice import Spice
import numpy as np

path = 'data/models/full.pth'
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f'Using device: {device}')

net = ClipGPT(device=device, generator='gpt2').to(device)


# load datasets
rsicd_dataset = RSICD(
    root="data/rsicd/",
    split="test", 
    transform=net.preprocess_clip
)
    
# UCM Captions datasets
ucm_dataset = UCMCaptions(
    root="data/ucm/",
    split="test", 
    transform=net.preprocess_clip
)

# NWPUCaptions datasets
nwpucaptions_dataset = NWPUCaptions(
    root="data/nwpu/",
    split="test", 
    transform=net.preprocess_clip
)

# Sidney Captions datasets
sydney_dataset = SidneyCaptions(
    root="data/sidney/",
    split="test", 
    transform=net.preprocess_clip
)

bartch_size = 16
rsicd_dataloader = DataLoader(rsicd_dataset, batch_size=bartch_size, shuffle=False)
ucm_dataloader = DataLoader(ucm_dataset, batch_size=bartch_size, shuffle=False)
nwpucaptions_dataloader = DataLoader(nwpucaptions_dataset, batch_size=bartch_size, shuffle=False)
sydney_dataloader = DataLoader(sydney_dataset, batch_size=bartch_size, shuffle=False)

bleu1_scorer = Bleu(n=1)
bleu2_scorer = Bleu(n=2)
bleu3_scorer = Bleu(n=3)
bleu4_scorer = Bleu(n=4)

cider_scorer = Cider()
rouge_scorer = Rouge()
spice_scorer = Spice()  # Add Spice scorer

def test(dataloader):
    net.eval()
    blue1_scores = []
    blue2_scores = []  # Add blue2_scores list
    blue3_scores = []  # Add blue3_scores list
    blue4_scores = []  # Add blue4_scores list
    rouge_scores = []
    cider_scores = []
    meteor_scores = []
    spice_scores = []  # Add spice_scores list
    count = 0
    with torch.no_grad():
        for batch in tqdm(dataloader):
            images = batch['x'].to(device)
            captions = batch['captions']
            captions = [[captions[j][i] for j in range(len(captions))] for i in range(len(captions[0]))]

            clip_embeddings = net.clip.encode_image(images)
            results = net.get_caption(clip_embeddings)

            refs = {count + i: captions[i] for i in range(len(captions))}
            res = {count + i: [results[i]] for i in range(len(results))}
            
            bleu1_score = bleu1_scorer.compute_score(refs, res, verbose=False)[0][0]
            blue1_scores.append(bleu1_score)

            bleu2_score = bleu2_scorer.compute_score(refs, res, verbose=False)[0][0]  # Compute blue2 score
            blue2_scores.append(bleu2_score)

            bleu3_score = bleu3_scorer.compute_score(refs, res, verbose=False)[0][0]  # Compute blue3 score
            blue3_scores.append(bleu3_score)

            bleu4_score = bleu4_scorer.compute_score(refs, res, verbose=False)[0][0]  # Compute blue4 score
            blue4_scores.append(bleu4_score)

            rouge_score = rouge_scorer.compute_score(refs, res)[0]
            rouge_scores.append(rouge_score)

            cider_score = cider_scorer.compute_score(refs, res)[0]
            cider_scores.append(cider_score)

            meteor_score = Meteor(captions, results)  # Compute meteor score
            meteor_scores.append(meteor_score)

            count += len(captions)

    return {
        'bleu1': np.mean(blue1_scores),
        'bleu2': np.mean(blue2_scores),  # Add bleu2 score
        'bleu3': np.mean(blue3_scores),  # Add bleu3 score
        'bleu4': np.mean(blue4_scores),  # Add bleu4 score
        'rouge': np.mean(rouge_scores),
        'cider': np.mean(cider_scores),
        'meteor': np.mean(meteor_scores),
    }

res = test(rsicd_dataloader)
print("-------------- RSICD results --------------")
print(f'RSICD BLEU1: {res["bleu1"]}')
print(f'RSICD BLEU2: {res["bleu2"]}')
print(f'RSICD BLEU3: {res["bleu3"]}')
print(f'RSICD BLEU4: {res["bleu4"]}')
print(f'RSICD ROUGE: {res["rouge"]}')
print(f'RSICD CIDER: {res["cider"]}')
print(f'RSICD METEOR: {res["meteor"]}')
print('\n\n')

print("-------------- UCM results --------------")
res = test(ucm_dataloader)
print(f'UCM BLEU1: {res["bleu1"]}')
print(f'UCM BLEU2: {res["bleu2"]}')
print(f'UCM BLEU3: {res["bleu3"]}')
print(f'UCM BLEU4: {res["bleu4"]}')
print(f'UCM ROUGE: {res["rouge"]}')
print(f'UCM CIDER: {res["cider"]}')
print(f'UCM METEOR: {res["meteor"]}')
print('\n\n')

print("-------------- NWPUCaptions results --------------")
res = test(nwpucaptions_dataloader)
print(f'NWPUCaptions BLEU1: {res["bleu1"]}')
print(f'NWPUCaptions BLEU2: {res["bleu2"]}')
print(f'NWPUCaptions BLEU3: {res["bleu3"]}')
print(f'NWPUCaptions BLEU4: {res["bleu4"]}')
print(f'NWPUCaptions ROUGE: {res["rouge"]}')
print(f'NWPUCaptions CIDER: {res["cider"]}')
print(f'NWPUCaptions METEOR: {res["meteor"]}')
print('\n\n')

print("-------------- Sidney results --------------")
res = test(sydney_dataloader)
print(f'Sydney BLEU1: {res["bleu1"]}')
print(f'Sydney BLEU2: {res["bleu2"]}')
print(f'Sydney BLEU3: {res["bleu3"]}')
print(f'Sydney BLEU4: {res["bleu4"]}')
print(f'Sydney ROUGE: {res["rouge"]}')
print(f'Sydney CIDER: {res["cider"]}')
print(f'Sydney METEOR: {res["meteor"]}')
print('\n\n')


