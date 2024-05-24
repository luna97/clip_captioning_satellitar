import torch
import torch.nn as nn
import clip
from transformers import GPT2LMHeadModel, GPT2Config, GPT2Tokenizer
from torchvision import transforms
from huggingface_hub import hf_hub_download

CLIP = 'clip'
VGG = 'vgg'
REMOTE_CLIP = 'remote_clip'

class ClipGPT(nn.Module):
    def __init__(self, device, encoder=CLIP, dropout=0.1):
        super(ClipGPT, self).__init__()

        # Load the GPT2 model and tokenizer
        gpt2_config = GPT2Config.from_pretrained('gpt2')
        gpt2_config.resid_pdrop=dropout
        gpt2_config.embd_pdrop=dropout
        self.generator = GPT2LMHeadModel.from_pretrained('gpt2', config=gpt2_config)
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.gen_embedding_size = self.generator.transformer.wte.weight.shape[1]
        
        if encoder == CLIP: 
            clip_model, preprocess = clip.load("ViT-B/32", device=device)
            if device == 'mps':   
                clip_model.type(torch.float32)
            self.clip = clip_model
            self.preprocess = preprocess
            self.clip.load_state_dict(torch.load('data/models/clip.pth', map_location=device))
            self.encoder_type = CLIP
            self.encoder_final_dim = self.clip.visual.transformer.width
        elif encoder == REMOTE_CLIP:
            clip_model, preprocess = clip.load("ViT-B/32", device=device)
            if device == 'mps':
                clip_model.type(torch.float32)
            self.clip = clip_model
            self.preprocess = preprocess
            checkpoint_path = hf_hub_download("chendelong/RemoteCLIP", "RemoteCLIP-ViT-B-32.pt", cache_dir='checkpoints')
            self.clip.load_state_dict(torch.load(checkpoint_path, map_location=device))
            self.encoder_type = CLIP
            self.encoder_final_dim = self.clip.visual.transformer.width
            
        elif encoder == VGG:
            self.vgg = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=True)
            self.preprocess = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            self.encoder_final_dim = 512
            self.encoder_type = VGG
        else:
            raise ValueError('Encoder must be either clip or vgg')
        

        self.adapted_layer = nn.Sequential(
            nn.Linear(self.encoder_final_dim, self.gen_embedding_size),
            nn.Dropout(dropout)
        )

        self.device = device
    
    def visual_clip(self, x: torch.Tensor):
        assert self.encoder_type == CLIP

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
        if self.encoder_type == VGG:
            decoder_embedding = self.vgg.features(images)
            decoder_embedding = decoder_embedding.permute(0, 2, 3, 1)
            decoder_embedding = decoder_embedding.view(decoder_embedding.shape[0], -1, decoder_embedding.shape[-1])
            # decoder_embedding = decoder_embedding.view(decoder_embedding.shape[0], 16, -1)
            # print(decoder_embedding.shape)
        elif self.encoder_type == CLIP:
            with torch.no_grad():
                decoder_embedding = self.visual_clip(images)
          
        tokens = self.tokenizer(captions, return_tensors='pt', truncation=True, padding="longest")
        input_ids = tokens.input_ids.to(self.device)
        att_mask = tokens.attention_mask.to(self.device)
        gen_embeddings = self.generator.transformer.wte(input_ids)
        decoder_embedding = self.adapted_layer(decoder_embedding.detach())
  
        # Concatenate CLIP embeddings with generator embeddings
        emb_cat = torch.cat([decoder_embedding, gen_embeddings], dim=1)

        # Create an attention mask for the concatenated embeddings
        clip_attention_mask = torch.ones(input_ids.shape[0], decoder_embedding.shape[1]).to(self.device)
        combined_att_mask = torch.cat([clip_attention_mask, att_mask], dim=1)

        ones = torch.ones(input_ids.shape[0], decoder_embedding.shape[1]).long().to(self.device)
        zeros = torch.zeros(input_ids.shape[0], decoder_embedding.shape[1]).long().to(self.device)

        with torch.no_grad():
            # set the token corresponding to the CLIP embedding to -100
            labels = torch.cat([ones, input_ids], dim=1)
            labels_att_mask = torch.cat([zeros, att_mask], dim=1)
            labels[~labels_att_mask.bool()] = -100

        # positional embedding are automatically added by the model
        return self.generator(
            inputs_embeds=emb_cat, 
            attention_mask=combined_att_mask,
            labels=labels
        ).loss

    def get_caption(self, images):
        """
        Generate captions for the images using the generator
        
        Args:
            images (torch.Tensor): a batch of images
        
        Returns:
            List[str]: captions for the images
        """
        if self.encoder_type == VGG:
            decoder_embedding = self.vgg.features(images)
            decoder_embedding = decoder_embedding.permute(0, 2, 3, 1)
            decoder_embedding = decoder_embedding.view(decoder_embedding.shape[0], -1, decoder_embedding.shape[-1])
        elif self.encoder_type == CLIP:
            decoder_embedding = self.visual_clip(images)

        decoder_embedding = self.adapted_layer(decoder_embedding.detach())

        # adding positional embeddings, I need them only here
        pos_emb = self.generator.transformer.wpe(torch.arange(decoder_embedding.shape[1]).to(self.device))
        decoder_embedding = decoder_embedding + pos_emb.unsqueeze(0)
        
        text = self.generator.generate(
            inputs_embeds=decoder_embedding,
            max_new_tokens=32,
            pad_token_id=self.generator.config.eos_token_id,
            num_beams=5,
        )

        # removed unwanted tokens and cut the text at the first dot
        sents = []
        for i in range(len(text)):
            sent = self.tokenizer.decode(text[i], skip_special_tokens=True)
            if '.' in sent:
                sent = sent.split('.')[0] + ' .'
            # replace \xa0
            sent = sent.replace('\xa0', ' ')
            sents.append(sent)

        return sents
    
