import torch
import random

from utils.script_util import create_multi_conditioner, load_checkpoint
from utils.config import Config

from encodec import EncodecModel

from jen1.diffusion.gdm import GaussianDiffusion
from jen1.model.model import UNetCFG1d
from jen1.noise_schedule import get_beta_schedule

class Jen1():
    def __init__(self, 
                 ckpt_path, 
                 device='cuda' if torch.cuda.is_available else 'cpu',
                 sample_rate = 48000, 
                 cross_attn_cond_ids=['prompt'], 
                 global_cond_ids= [],
                 input_concat_ids= ['masked_input', 'mask']):
        self.ckpt_path = ckpt_path
        self.device = device
        self.sample_rate = sample_rate
        self.config = Config
        self.conditioner = create_multi_conditioner(self.config.conditioner_config)     
        self.cross_attn_cond_ids = cross_attn_cond_ids
        self.global_cond_ids = global_cond_ids
        self.input_concat_ids = input_concat_ids 
        
        self.audio_encoder = EncodecModel.encodec_model_48khz()
        
    def generate(self, prompt, steps=100, batch_size=1, seconds=30, task='text_guided'):
        sample_length = seconds * self.sample_rate
        shape = (batch_size, self.audio_encoder.channels, sample_length)
        wav = torch.randn(shape)
        diffusion_config = self.config.diffusion_config.gaussian_diffusion
        model_config = self.config.model_config
        betas = get_beta_schedule(diffusion_config.noise_schedule, diffusion_config.steps)
        
        diffusion = GaussianDiffusion(steps=diffusion_config.steps, betas=betas,
                                      objective=diffusion_config.objective, loss_type=diffusion_config.loss_type,
                                      device=self.device, cfg_dropout_proba=diffusion_config.cfg_dropout_proba,
                                      embedding_scale=diffusion_config.embedding_scale,
                                      batch_cfg=diffusion_config.batch_cfg, scale_cfg=diffusion_config.scale_cfg,
                                      sampling_timesteps=steps, use_fp16=False)
        
        config_dict = {k: v for k, v in model_config.__dict__.items() if not k.startswith('__') and not callable(v)}
        context_embedding_features = config_dict.pop('context_embedding_features', None)
        context_embedding_max_length = config_dict.pop('context_embedding_max_length', None)
        
        model = UNetCFG1d(context_embedding_features=context_embedding_features, 
                     context_embedding_max_length=context_embedding_max_length, 
                     **config_dict).to(self.device)
        
        model, _, _, _ = load_checkpoint(self.ckpt_path, model)
        model.eval()
        diffusion.eval()
        
        with torch.no_grad():
            encoded_frames = self.audio_encoder.encode(wav)
            codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1)
            codes = codes.transpose(0, 1)
            emb = self.audio_encoder.quantizer.decode(codes)
            
            batch_metadata = [{'prompt': prompt} for _ in range(batch_size)]
            conditioning = self.conditioner(batch_metadata, self.device)
            masked_input, mask, causal = self.random_mask(emb, emb.shape[2], task)
            conditioning['masked_input'] = masked_input
            conditioning['mask'] = mask
            conditioning = self.get_conditioning(conditioning)
            shape = emb.shape
            sample_embs = diffusion.sample(model, shape, conditioning)
            samples = self.audio_encoder.decoder(sample_embs)
        
        return samples
    
    def random_mask(self, sequence, max_mask_length, task):
        b, _, sequence_length = sequence.size()
        
        masks = []
        if task.lower() == 'text_guided':
            mask = torch.zeros((1, 1, sequence_length)).to(sequence.device)
            masks.append(mask)
            causal = random.choices([True, False])
            causal = False
        elif task.lower() == 'music_inpaint':
            mask_length = random.randint(sequence_length*0.2, sequence_length*0.8)
            mask_start = random.randint(0, sequence_length-mask_length)
            
            mask = torch.ones((1, 1, sequence_length))
            mask[:, :, mask_start:mask_start+mask_length] = 0
            mask = mask.to(sequence.device)
            
            masks.append(mask)
            causal = False
        elif task.lower() == 'music_cont':
            mask_length = random.randint(sequence_length*0.2, sequence_length*0.8)
            
            mask = torch.onse((1, 1, sequence_length))
            mask[:, :, -mask_length:] = 0
            masks.append(mask)
            causal = True
        
        masks = masks * b
        mask = torch.cat(masks, dim=0).to(sequence.device)
        
        masked_sequence = sequence * mask
        
        return masked_sequence, mask, causal
            
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
        
if __name__ == '__main__':
    jen1 = Jen1(ckpt_path=None)
    prompt = 'a beautiful song'
    samples = jen1.generate(prompt=prompt)