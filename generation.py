import torch

from dataset.audio_processor import AudioTokenizer
from utils.script_util import create_model_and_diffusion, create_multi_conditioner, load_checkpoint
from utils.config import Config

class Jen1:
    def __init__(self, ckpt_path, device='cuda' if torch.cuda.is_available else 'cpu',
                 cross_attn_cond_ids=['prompt'], global_cond_ids= [],
                 input_concat_ids= []):
        self.ckpt_path = ckpt_path
        self.config = Config
        self.audio_encoder = AudioTokenizer(device=device)
        self.conditioner = create_multi_conditioner(self.config.conditioner_config)     
        self.cross_attn_cond_ids = cross_attn_cond_ids
        self.global_cond_ids = global_cond_ids
        self.input_concat_ids = input_concat_ids
        
    def generate(self, prompt, steps=100, batch_size=1, seconds=10):
        sample_length = seconds * self.audio_encoder.sample_rate
        shape = (batch_size, 1, sample_length)
        wav = torch.randn(shape)
        model, diffusion = create_model_and_diffusion(self.config, steps)
        model, _, _, _ = load_checkpoint(self.ckpt_path, model)
        model.eval()
        diffusion.eval()
        with torch.no_grad():
            metadata = {'prompt': prompt}
            conditioning = self.conditioner(metadata)
            conditioning = self.get_conditioning(conditioning)
            emb, _, _ = self.audio_encoder(wav)
            shape = emb.shape
            sample_embs = diffusion.sample(model, shape, conditioning)
            samples = self.audio_encoder.model.decoder(sample_embs)
        
        return samples
            
    def get_conditioning(self, cond):
        cross_attention_input = None
        cross_attention_masks = None
        global_cond = None
        input_concat_cond = None

        if len(self.cross_attn_cond_ids) > 0:
            # Concatenate all cross-attention inputs over the sequence dimension
            # Assumes that the cross-attention inputs are of shape (batch, seq, channels)
            cross_attention_input = torch.cat([cond[key][0] for key in self.cross_attn_cond_ids], dim=1)
            cross_attention_masks = torch.cat([cond[key][1] for key in self.cross_attn_cond_ids], dim=1)

        if len(self.global_cond_ids) > 0:
            # Concatenate all global conditioning inputs over the channel dimension
            # Assumes that the global conditioning inputs are of shape (batch, channels)
            global_cond = torch.cat([cond[key][0] for key in self.global_cond_ids], dim=-1)
            if len(global_cond.shape) == 3:
                global_cond = global_cond.squeeze(1)
        
        if len(self.input_concat_ids) > 0:
            # Concatenate all input concat conditioning inputs over the channel dimension
            # Assumes that the input concat conditioning inputs are of shape (batch, channels, seq)
            input_concat_cond = torch.cat([cond[key][0] for key in self.input_concat_ids], dim=1)
            
        return {
            "cross_attn_cond": cross_attention_input,
            "cross_attn_masks": cross_attention_masks,
            "global_cond": global_cond,
            "input_concat_cond": input_concat_cond
        }