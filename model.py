import torch
import torch.nn as nn
import clip
from transformers import GPT2LMHeadModel, GPT2Config, T5ForConditionalGeneration, T5Config, AutoTokenizer
from transformers import StoppingCriteria, StoppingCriteriaList


class ClipGPT(nn.Module):
    def __init__(self, device, generator='gpt2', dropout=0.0):
        super(ClipGPT, self).__init__()
        if generator == 'gpt2':
            gpt2_config = GPT2Config.from_pretrained('gpt2')
            gpt2_config.resid_pdrop=dropout
            gpt2_config.embd_pdrop=dropout
            self.generator  = GPT2LMHeadModel.from_pretrained('gpt2', config=gpt2_config)
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.gen_embedding_size = self.generator.transformer.wte.weight.shape[1]
        else:
            raise ValueError('Generator not supported')
        
        self.adapted_layer = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(512, self.gen_embedding_size),
        )
        # self.dropout = nn.Dropout(dropout)

        clip_model, preprocess = clip.load("ViT-B/32", device=device)
        # use full precision for the model
        clip_model.type(torch.float32)
        self.clip = clip_model
        self.preprocess_clip = preprocess

        self.generator_type = generator

        self.device = device

    def train_clip(self, images, captions):
        image_features = self.clip.encode_image(images)
        captions_clip = clip.tokenize(captions).to(images.device)
        text_features = self.clip.encode_text(captions_clip)
        return image_features, text_features
    
    def train_generator(self, captions=None, images=None):
    # Encode images or text to get CLIP embeddings
        with torch.no_grad():
            if images is not None:
                clip_embedding = self.clip.encode_image(images)
            else:
                captions_clip = clip.tokenize(captions).to(self.device)
                clip_embedding = self.clip.encode_text(captions_clip)
          
            tokens = self.tokenizer(captions, return_tensors='pt', truncation=True, padding="longest")
            input_ids = tokens.input_ids.to(self.device)
            att_mask = tokens.attention_mask.to(self.device)
            gen_embeddings = self.generator.transformer.wte(input_ids)

        clip_embedding = self.adapted_layer(clip_embedding.detach())
  
        # Concatenate CLIP embeddings with generator embeddings
        emb_cat = torch.cat([clip_embedding, gen_embeddings], dim=1)

        # Create an attention mask for the concatenated embeddings
        clip_attention_mask = torch.ones(input_ids.shape[0], clip_embedding.shape[1]).to(self.device)
        combined_att_mask = torch.cat([clip_attention_mask, att_mask], dim=1)

        ones = torch.ones(input_ids.shape[0], clip_embedding.shape[1]).long().to(self.device)

        with torch.no_grad():
            # input_ids[~att_mask.bool()] = -100 
            labels = torch.cat([ones - 101, input_ids], dim=1)

        # positional embedding are automatically added by the model
        return self.generator(
            inputs_embeds=emb_cat, 
            attention_mask=combined_att_mask,
            labels=labels
        ).loss

    
    def get_caption(self, clip_embedding):
        clip_embedding = self.adapted_layer(clip_embedding.detach()).unsqueeze(1)
        clip_embedding = clip_embedding.view(-1, self.prefix_length, self.gen_embedding_size)

        # adding positional embeddings, I need it only here
        pos_emb = self.generator.transformer.wpe(torch.arange(clip_embedding.shape[1]).to(self.device))
        clip_embedding = clip_embedding + pos_emb.unsqueeze(0)
        
        # stopping_criteria = StoppingCriteriaList([StopOnTokens(stop_token_ids)])

        text = self.generator.generate(
            inputs_embeds=clip_embedding,
            max_new_tokens=32,
            pad_token_id=self.generator.config.eos_token_id,
            num_beams=10,
            early_stopping=True,
            temperature=0.7,  # adjust temperature to control randomness
            top_k=20,  # consider the top 50 tokens at each step
            top_p=0.90,  # nucleus sampling
            repetition_penalty=1.2,  # encourage the model to generate different tokens
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
    
