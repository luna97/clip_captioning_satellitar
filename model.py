import torch
import torch.nn as nn
import clip
from transformers import GPT2LMHeadModel, GPT2Config, GPT2Tokenizer
from transformers import StoppingCriteria, StoppingCriteriaList


class ClipGPT(nn.Module):
    def __init__(self, device, generator='gpt2', dropout=0.1, prefix_length=4):
        super(ClipGPT, self).__init__()
        if generator == 'gpt2':
            gpt2_config = GPT2Config.from_pretrained('gpt2')
            gpt2_config.resid_pdrop=dropout
            gpt2_config.embd_pdrop=dropout
            # self.generator  = GPT2LMHeadModel(gpt2_config)
            self.generator = GPT2LMHeadModel.from_pretrained('gpt2', config=gpt2_config)
            self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.gen_embedding_size = self.generator.transformer.wte.weight.shape[1]
        else:
            raise ValueError('Generator not supported')
        


        clip_model, preprocess = clip.load("ViT-B/32", device=device)
        # use full precision for the model
        clip_model.type(torch.float32)
        self.clip = clip_model
        self.preprocess_clip = preprocess
        self.prefix_length = prefix_length

        self.adapted_layer = nn.Sequential(
            nn.Linear(self.clip.visual.transformer.width, self.gen_embedding_size),# * prefix_length),
            nn.Dropout(dropout)
        )

        self.generator_type = generator

        self.device = device

    def train_clip(self, images, captions):
        image_features = self.clip.encode_image(images)
        captions_clip = clip.tokenize(captions).to(images.device)
        text_features = self.clip.encode_text(captions_clip)
        return image_features, text_features
    
    def visual_clip(self, x: torch.Tensor):
        x = self.clip.visual.conv1(x.type(self.clip.dtype))  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.clip.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.clip.visual.positional_embedding.to(x.dtype)
        x = self.clip.visual.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.clip.visual.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.clip.visual.ln_post(x)
        return x
    
    def train_generator(self, captions, images):
    # Encode images or text to get CLIP embeddings
        with torch.no_grad():
            clip_embedding = self.visual_clip(images)
          
            tokens = self.tokenizer(captions, return_tensors='pt', truncation=True, padding="longest")
            input_ids = tokens.input_ids.to(self.device)
            att_mask = tokens.attention_mask.to(self.device)
            gen_embeddings = self.generator.transformer.wte(input_ids)

        clip_embedding = self.adapted_layer(clip_embedding.detach())
        # clip_embedding = clip_embedding.view(-1, self.prefix_length, self.gen_embedding_size)
  
        # Concatenate CLIP embeddings with generator embeddings
        emb_cat = torch.cat([clip_embedding, gen_embeddings], dim=1)

        # Create an attention mask for the concatenated embeddings
        clip_attention_mask = torch.ones(input_ids.shape[0], clip_embedding.shape[1]).to(self.device)
        combined_att_mask = torch.cat([clip_attention_mask, att_mask], dim=1)

        ones = torch.ones(input_ids.shape[0], clip_embedding.shape[1]).long().to(self.device)

        with torch.no_grad():
            # set the token corresponding to the CLIP embedding to -100
            labels = torch.cat([ones, input_ids], dim=1)
            labels_att_mask = torch.cat([ones, att_mask], dim=1)
            labels[~labels_att_mask.bool()] = -100

        # positional embedding are automatically added by the model
        return self.generator(
            inputs_embeds=emb_cat, 
            attention_mask=combined_att_mask,
            labels=labels
        ).loss

    
    def get_caption(self, images):
        clip_embedding = self.visual_clip(images)
        clip_embedding = self.adapted_layer(clip_embedding.detach())
        # clip_embedding = clip_embedding.view(-1, self.prefix_length, self.gen_embedding_size)

        # adding positional embeddings, I need it only here
        pos_emb = self.generator.transformer.wpe(torch.arange(clip_embedding.shape[1]).to(self.device))
        clip_embedding = clip_embedding + pos_emb.unsqueeze(0)
        
        # stopping_criteria = StoppingCriteriaList([StopOnTokens(stop_token_ids)])

        text = self.generator.generate(
            inputs_embeds=clip_embedding,
            max_new_tokens=32,
            pad_token_id=self.generator.config.eos_token_id,
            num_beams=5,
            #early_stopping=True,
            #temperature=0.7,  # adjust temperature to control randomness
            #top_k=20,  # consider the top 50 tokens at each step
            #top_p=0.90,  # nucleus sampling
            #repetition_penalty=1.2,  # encourage the model to generate different tokens
            # stopping_criteria=stopping_criteria
        )

        sents = []
        for i in range(len(text)):
            sent = self.tokenizer.decode(text[i], skip_special_tokens=True)
            if '.' in sent:
                sent = sent.split('.')[0] + ' .'
            # replace \xa0
            sent = sent.replace('\xa0', ' ')
            sents.append(sent)

        return sents
    

class StopOnTokens(StoppingCriteria):
    def __init__(self, stop_token_ids):
        self.stop_token_ids = stop_token_ids

    def __call__(self, input_ids, scores):
        if any(input_id in self.stop_token_ids for input_id in input_ids[0]):
            return True
        return False
    
