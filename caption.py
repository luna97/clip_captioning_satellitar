# args image path

import os
import numpy as np
import argparse
import torch
from model import ClipGPT

parser = argparse.ArgumentParser(description='ClipGPT model for inference')
parser.add_argument('--image_path', type=str, default=None, help='Path to image')

args = parser.parse_args()

image_path = args.image_path
assert os.path.exists(image_path), f'Image path {image_path} does not exist'
assert os.path.exists('data/models/full.pth')

# load model
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
net = ClipGPT(device=device, generator='gpt2').to(device)

# load model
net.load_state_dict(torch.load('data/models/full.pth', map_location=device))

# load image
from PIL import Image

image = Image.open(image_path).convert("RGB")

# preprocess image
image = net.preprocess_clip(image).unsqueeze(0).to(device)

# get caption
net.eval()
with torch.no_grad():
    clip_embeddings = net.clip.encode_image(image)
    caption = net.get_caption(clip_embeddings)
print('caption for image: ', image_path)
print(caption)
