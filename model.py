import torch
import torch.nn as nn
import clip
from transformers import GPT2LMHeadModel, GPT2Config, T5ForConditionalGeneration, T5Config, AutoTokenizer


class ClipGPT(nn.Module):
    def __init__(self, device, generator='gpt2', prefix_size=512, prefix_length=10):
        super(ClipGPT, self).__init__()
        if generator == 'gpt2':
            gpt2_config = GPT2Config.from_pretrained('gpt2')
            self.generator  = GPT2LMHeadModel.from_pretrained('gpt2', config=gpt2_config)
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.gen_embedding_size = self.generator.transformer.wte.weight.shape[1]
        elif generator == 't5':
            t5_config = T5Config.from_pretrained('t5-small')
            self.generator = T5ForConditionalGeneration.from_pretrained('t5-small', config=t5_config)
            self.tokenizer = AutoTokenizer.from_pretrained("t5-small")
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.gen_embedding_size = self.generator.shared.weight.shape[1]
        else:
            raise ValueError('Generator not supported')
        
        self.adapted_layer = nn.Linear(prefix_size, self.gen_embedding_size)


        clip_model, preprocess = clip.load("ViT-B/32", device=device)
        # use full precision for the model
        clip_model.type(torch.float32)
        self.clip = clip_model
        self.preprocess_clip = preprocess

        self.generator_type = generator
        self.prefix_length = prefix_length
        self.prefix_size = prefix_size

        self.device = device

        self.gen_loss = torch.nn.CrossEntropyLoss(ignore_index=-100)

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
    
    def train_generator(self, captions):
        with torch.no_grad():
            captions_clip = clip.tokenize(captions).to(self.device)
            clip_embedding = self.clip.encode_text(captions_clip)
            # clip_embedding = self.vision_transformer(images)
            # clip_embedding = clip_embedding / clip_embedding.norm(2, dim=-1, keepdim=True)
            clip_embedding = self.adapted_layer(clip_embedding.detach()).unsqueeze(1)
            # clip_embedding = clip_embedding.view(-1, self.prefix_length, self.gen_embedding_size)
            
        tokens = self.tokenizer(captions, return_tensors='pt', truncation=True, padding="longest")
        input_ids = tokens.input_ids.to(self.device)
        att_mask = tokens.attention_mask.to(self.device)

        # add a one at the first zero of the attention mask to include on EOF token
        for i in range(att_mask.shape[0]):
            first_zero = torch.where(att_mask[i] == 0)[0]
            if len(first_zero) > 0:
                att_mask[i, first_zero[0]] = 1

        # set padding tokens to -100
        input_ids[~att_mask.bool()] = -100

        # clip_embedding = clip_embedding.unsqueeze(1)
        if self.generator_type == 'gpt2':
            gen_embeddings = self.generator.transformer.wte(input_ids)
            # add start token
        elif self.generator_type == 't5':
            gen_embeddings = self.generator.shared(input_ids)

        emb_cat = torch.cat([clip_embedding, gen_embeddings], dim=1)
        ones = torch.ones(input_ids.shape[0], clip_embedding.shape[1]).to(self.device)
        labels = torch.cat([ones - 101, input_ids], dim=1)

        att_mask = torch.cat([ones, att_mask], dim=1)

        output = self.generator(
            inputs_embeds=emb_cat, 
            # decoder_inputs_embeds=emb_cat,
            attention_mask=att_mask,
            labels=labels
        )

        return output.loss
    
    def get_caption(self, clip_embedding):
        clip_embedding = self.adapted_layer(clip_embedding.detach()).unsqueeze(1)

        text = self.generator.generate(
            inputs_embeds=clip_embedding,
            max_new_tokens=32,
            pad_token_id=self.generator.config.eos_token_id
        )

        return [ self.tokenizer.decode(text[i], skip_special_tokens=True) for i in range(len(text)) ]
    
