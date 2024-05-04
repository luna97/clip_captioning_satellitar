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


class ClipGPT(nn.Module):
    def __init__(self, device, generator='gpt2', prefix_size=512, prefix_length=10):
        super(ClipGPT, self).__init__()
        if generator == 'gpt2':
            gpt2_config = GPT2Config.from_pretrained('gpt2')
            self.generator  = GPT2LMHeadModel.from_pretrained('gpt2', config=gpt2_config)
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.gen_embedding_size = self.generator.transformer.wte.weight.shape[1]

            self.adapted_layer = nn.Sequential(
                nn.Linear(
                    prefix_size, 
                    self.gen_embedding_size * prefix_length // 2
                ),
                nn.Tanh(),
                nn.Linear(
                    self.gen_embedding_size * prefix_length // 2, 
                    self.gen_embedding_size * prefix_length
                )
            )


        elif generator == 't5':
            t5_config = T5Config.from_pretrained('t5-small')
            self.generator = T5ForConditionalGeneration.from_pretrained('t5-small', config=t5_config)
            self.tokenizer = AutoTokenizer.from_pretrained("t5-small")
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.adapted_layer = nn.Linear(prefix_size, 512)

        elif generator == 'LSTM':
            self.generator = nn.LSTM(512, 512, batch_first=True)
            self.adapted_layer = nn.Linear(prefix_size, 512)
            self.head = nn.Linear(512, 50264)
            # i can use the gpt2 tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        else:
            raise ValueError('Generator not supported')

        clip_model, preprocess = clip.load("ViT-B/32", device=device)
        # use full precision for the model
        clip_model.type(torch.float32)
        self.clip = clip_model
        self.preprocess_clip = preprocess

        self.generator_type = generator
        self.prefix_length = prefix_length
        self.prefix_size = prefix_size

        self.device = device

        self.gen_loss = torch.nn.CrossEntropyLoss(ignore_index=0)

    def vision_transformer(self, x):
        x = x.type(self.clip.dtype)
        x = self.clip.visual.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.clip.visual.class_embedding.to(x.dtype) 
                       + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.clip.visual.positional_embedding.to(x.dtype)
        x = self.clip.visual.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.clip.visual.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        return x

    def train_clip(self, images, captions):
        image_features = self.clip.encode_image(images)
        captions_clip = clip.tokenize(captions).to(images.device)
        text_features = self.clip.encode_text(captions_clip)
        return image_features, text_features
    
    def train_generator(self, captions, images):
        # remove the . from the captions
        captions = [caption.replace('.', '') for caption in captions]
        # trim th caption on blank spaces
        captions = [caption.strip() for caption in captions]
        with torch.no_grad():
            captions_clip = clip.tokenize(captions).to(self.device)
            clip_embedding = self.clip.encode_text(captions_clip)
            # clip_embedding = self.vision_transformer(images)
            # clip_embedding = clip_embedding / clip_embedding.norm(dim=-1, pow=2, keepdim=True)
            clip_embedding = self.adapted_layer(clip_embedding.detach())
            clip_embedding = clip_embedding.view(-1, self.prefix_length, self.gen_embedding_size)
            
        tokens = self.tokenizer(captions, return_tensors='pt', truncation=True, padding=True)
        input_ids = tokens.input_ids.to(self.device)
        att_mask = tokens.attention_mask.to(self.device)

        if self.generator_type == 'LSTM':
            h0 = torch.zeros(1, 512).to(self.device)
            c0 = torch.zeros(1, 512).to(self.device)
            output, (h0, c0) = self.generator(clip_embedding, (h0, c0))
            outputs = [ output ]
            for i in range(input_ids.shape[1] - 1):
                output, (h0, c0) = self.generator(output, (h0, c0))
                outputs.append(output)
            outputs = torch.stack(outputs, dim=1)
            logits = self.head(outputs)
            loss = self.gen_loss(logits.flatten(0, 1), input_ids.flatten(0, 1))
            return loss

        # clip_embedding = clip_embedding.unsqueeze(1)
        if self.generator_type == 'gpt2':
            gen_embeddings = self.generator.transformer.wte(input_ids)
        elif self.generator_type == 't5':
            gen_embeddings = self.generator.shared(input_ids)

        emb_cat = torch.cat([clip_embedding, gen_embeddings], dim=1)
        # att_mask = torch.cat([torch.ones(clip_embedding.shape[0], clip_embedding.shape[1]).to(self.device), att_mask], dim=1)
        zeros = torch.zeros(input_ids.shape[0], self.prefix_length).to(self.device)
        labels = torch.cat([zeros, input_ids], dim=1)

        att_mask = torch.cat([zeros + 1, att_mask], dim=1)

        logits = self.generator(
            inputs_embeds=emb_cat, 
            attention_mask=att_mask,
            labels=labels
        ).logits[:, self.prefix_length -1 : -1]
        # logits = torch.argmax(logits, dim=-1)
        loss = self.gen_loss(logits.flatten(0, 1), input_ids.flatten(0, 1))
        return loss
    
    def get_caption(self, images):
        clip_embedding = self.clip.encode_image(images)
        # clip_embedding = clip_embedding / clip_embedding.norm(pow=2, dim=-1, keepdim=True)
        clip_embedding = self.adapted_layer(clip_embedding.detach())
        clip_embedding = clip_embedding.view(-1, self.prefix_length, self.gen_embedding_size)

        if self.generator_type == 'LSTM':
            h0 = torch.zeros(1, 512).to(self.device)
            c0 = torch.zeros(1, 512).to(self.device)
            output, (h0, c0) = self.generator(embeddings, (h0, c0))
            outputs = [ output ]
            for i in range(32):
                output, (h0, c0) = self.generator(output, (h0, c0))
                outputs.append(output)
            outputs = torch.stack(outputs, dim=1)
            logits = self.head(outputs)
            return torch.argmax(logits, dim=-1)

        text = self.generator.generate(
            inputs_embeds=clip_embedding,
            max_new_tokens=32,
            pad_token_id=self.generator.config.eos_token_id
        )
        return text
    

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f'Using device: {device}')


net = ClipGPT(device=device, generator='gpt2').to(device)

transform = T.Compose([
    net.preprocess_clip,
    T.RandomCrop(180),
    T.ColorJitter(),
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.RandomResizedCrop(180, scale=(0.8, 1.2), ratio=(1.0, 1.0)),
    T.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
])


dataset_train = RSICD(
    root="data/rsicd/",
    split="train", 
    transform=net.preprocess_clip
)

dataset_val = RSICD(
    root="data/rsicd/",
    split="val", 
    transform=net.preprocess_clip
)

dataset_test = RSICD(
    root="data/rsicd/",
    split="test", 
    transform=net.preprocess_clip
)

print(f'Length of train dataset: {len(dataset_train)}')
print(f'Length of val dataset: {len(dataset_val)}')
print(f'Length of test dataset: {len(dataset_test)}')


def collate_fn(batch):
    images = [item['x'] for item in batch]
    # get a random caption from each image
    random_index = [np.random.randint(0, len(item['captions'])) for item in batch]
    captions = [item['captions'][random_index[i]] for i, item in enumerate(batch)]
    return torch.stack(images), captions

batch_size = 32
dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)


# train clip in a contrastive way with the captions
import torch.nn as nn
from tqdm import tqdm

epochs = 10
warmup_ratio = 0.005
warmup_steps = len(dataloader_train) * epochs * warmup_ratio 
# load clip model if exists
if os.path.exists('data/clip.pth'):
    # load on device
    net.clip.load_state_dict(torch.load('data/clip.pth', map_location=device))
else:
    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()

    optimizer_clip = torch.optim.AdamW(net.clip.parameters(), lr=1e-5, weight_decay=0.00001)
    sched_clip = get_linear_schedule_with_warmup(optimizer_clip, num_warmup_steps=warmup_steps, num_training_steps=len(dataloader_train) * epochs)

    #train clip justo for one epoch
    for epoch in range(epochs):
        net.train()
        epoch_bar = tqdm(dataloader_train, total=len(dataloader_train))
        for images, captions in epoch_bar:
            images = images.to(device)
            image_features, text_features = net.train_clip(images, captions)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            logits = torch.matmul(image_features, text_features.T)
            target = torch.arange(image_features.shape[0]).to(device)

            loss = (loss_img(logits, target) + loss_txt(logits.T, target)) / 2

            # cosine_loss = loss(image_features, text_features, torch.ones(image_features.shape[0]).to(device))
            # cosine_loss.backward()
            loss.backward()
            optimizer_clip.step()
            optimizer_clip.zero_grad()
            sched_clip.step()

            epoch_bar.set_postfix(loss_clip=loss.item())

# save clip model
torch.save(net.clip.state_dict(), 'data/clip.pth')

epochs = 50
lr=1e-4
optimizer_gen = torch.optim.AdamW([
    { 'params': net.generator.parameters(), 'lr': 1e-6},
    # { 'params': net.head.parameters(), 'lr': lr},
    { 'params': net.adapted_layer.parameters(), 'lr': 1e-3}
], lr=lr, weight_decay=0.0000001)

sched_gpt = get_linear_schedule_with_warmup(
    optimizer_gen,
    num_warmup_steps=warmup_steps, 
    num_training_steps=len(dataloader_train) * epochs
)

for epoch in range(epochs):
    net.train()
    epoch_bar = tqdm(dataloader_train, total=len(dataloader_train), desc=f'Epoch {epoch+1}/{epochs}')
    for images, captions in epoch_bar:
        images = images.to(device)
        loss = net.train_generator(captions, images)

        loss.backward()
        optimizer_gen.step()
        optimizer_gen.zero_grad()
        sched_gpt.step()

        epoch_bar.set_postfix(loss_gpt=loss.item())

    # print(f'Epoch: {epoch+1}, Loss: {output.item()}')

    # evaluate the model
    net.eval()
    for images, target in dataloader_val:
        images = images.to(device)
        captions = net.get_caption(images)
        # print(captions.shape)
        # convert captions to text
        # text = clip.tokenize(captions)
        for i in range(captions.size(0)):
            text = net.tokenizer.decode(captions[i], skip_special_tokens=True)
            print(f'prediction: {text}')
            print(f'target: {target[i]}')
        break
            