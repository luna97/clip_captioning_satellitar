# load model
import torch
from model import ClipGPT
from dataset import NWPUCaptions, SidneyCaptions
from torch.utils.data import DataLoader
from torchrs.datasets import RSICD, UCMCaptions
from torchvision import transforms as T
from tqdm import tqdm
from pycocoevalcap.cider.cider import Cider
# from pycocoevalcap.meteor.meteor import Meteor
from nltk.translate.meteor_score import meteor_score as Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.spice.spice import Spice
import numpy as np
from dataset import get_test_datasets

import nltk
nltk.download('wordnet')

path = 'data/models/full.pth'
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f'Using device: {device}')

net = ClipGPT(device=device, generator='gpt2').to(device)

test_datasets = get_test_datasets(net.preprocess_clip)

# load datasets
bartch_size = 16

bleu_scorer = Bleu(n=4)

cider_scorer = Cider()
rouge_scorer = Rouge()
spice_scorer = Spice()

def test(dataloader):
    net.eval()

    refs = {}
    res = {}

    count = 0
    with torch.no_grad():
        for batch in tqdm(dataloader):
            images = batch['x'].to(device)
            captions = batch['captions']
            captions = [[captions[j][i] for j in range(len(captions))] for i in range(len(captions[0]))]

            clip_embeddings = net.clip.encode_image(images)
            results = net.get_caption(clip_embeddings)

            for c in range(len(captions)):
                refs[count + c] = captions[c]
                res[count + c] = [results[c]]

            count += len(captions)

        bleu_score = bleu_scorer.compute_score(refs, res, verbose=False)[0]

        rouge_score, _ = rouge_scorer.compute_score(refs, res)

        cider_score, _ = cider_scorer.compute_score(refs, res)

        # spice_score, _ = spice_scorer.compute_score(refs, res)

        meteor_score = Meteor(captions, results)  # Compute meteor score


    return {
        'bleu1': bleu_score[0],
        'bleu2': bleu_score[1],
        'bleu3': bleu_score[2],
        'bleu4': bleu_score[3],
        'rouge': rouge_score,
        'cider': cider_score,
        'meteor': meteor_score,
        #'spice': spice_score
    }

if 'rsicd' in test_datasets.keys():
    rsicd_dataset = test_datasets['rsicd']
    rsicd_dataloader = DataLoader(rsicd_dataset, batch_size=bartch_size, shuffle=False)
    res = test(rsicd_dataloader)

    print("-------------- RSICD results --------------")
    print(f'RSICD BLEU1: {res["bleu1"]}')
    print(f'RSICD BLEU2: {res["bleu2"]}')
    print(f'RSICD BLEU3: {res["bleu3"]}')
    print(f'RSICD BLEU4: {res["bleu4"]}')
    print(f'RSICD ROUGE: {res["rouge"]}')
    print(f'RSICD CIDER: {res["cider"]}')
    print(f'RSICD METEOR: {res["meteor"]}')
    # print(f'RSICD SPICE: {res["spice"]}')
    print('\n\n')

if 'ucm' in test_datasets.keys():
    ucm_dataset = test_datasets['ucm']
    ucm_dataloader = DataLoader(ucm_dataset, batch_size=bartch_size, shuffle=False)
    res = test(ucm_dataloader)

    print("-------------- UCM results --------------")
    print(f'UCM BLEU1: {res["bleu1"]}')
    print(f'UCM BLEU2: {res["bleu2"]}')
    print(f'UCM BLEU3: {res["bleu3"]}')
    print(f'UCM BLEU4: {res["bleu4"]}')
    print(f'UCM ROUGE: {res["rouge"]}')
    print(f'UCM CIDER: {res["cider"]}')
    print(f'UCM METEOR: {res["meteor"]}')
    # print(f'UCM SPICE: {res["spice"]}')
    print('\n\n')

if 'nwpu' in test_datasets.keys():
    nwpucaptions_dataset = test_datasets['nwpu']
    nwpucaptions_dataloader = DataLoader(nwpucaptions_dataset, batch_size=bartch_size, shuffle=False)
    res = test(nwpucaptions_dataloader)

    print("-------------- NWPUCaptions results --------------")
    print(f'NWPUCaptions BLEU1: {res["bleu1"]}')
    print(f'NWPUCaptions BLEU2: {res["bleu2"]}')
    print(f'NWPUCaptions BLEU3: {res["bleu3"]}')
    print(f'NWPUCaptions BLEU4: {res["bleu4"]}')
    print(f'NWPUCaptions ROUGE: {res["rouge"]}')
    print(f'NWPUCaptions CIDER: {res["cider"]}')
    print(f'NWPUCaptions METEOR: {res["meteor"]}')
    # print(f'NWPUCaptions SPICE: {res["spice"]}')
    print('\n\n')

if 'sidney' in test_datasets.keys():
    sidneycaptions_dataset = test_datasets['sidney']
    sidneycaptions_dataloader = DataLoader(sidneycaptions_dataset, batch_size=bartch_size, shuffle=False)
    res = test(sidneycaptions_dataloader)

    print("-------------- SidneyCaptions results --------------")
    print(f'SidneyCaptions BLEU1: {res["bleu1"]}')
    print(f'SidneyCaptions BLEU2: {res["bleu2"]}')
    print(f'SidneyCaptions BLEU3: {res["bleu3"]}')
    print(f'SidneyCaptions BLEU4: {res["bleu4"]}')
    print(f'SidneyCaptions ROUGE: {res["rouge"]}')
    print(f'SidneyCaptions CIDER: {res["cider"]}')
    print(f'SidneyCaptions METEOR: {res["meteor"]}')
   #  print(f'SidneyCaptions SPICE: {res["spice"]}')
    print('\n\n')