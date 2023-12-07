import torch
import random
import numpy as np
import math

from utils.script_util import create_multi_conditioner, load_checkpoint
from utils.config import Config

from encodec import EncodecModel
from encodec.utils import convert_audio

from jen1.diffusion.gdm.gdm import GaussianDiffusion
from jen1.diffusion.vdm.vdm import VDM
from jen1.model.model import UNetCFG1d
from jen1.diffusion.gdm.noise_schedule import get_beta_schedule

class Jen1():
    def __init__(self, 
                 ckpt_path, 
                 device='cuda' if torch.cuda.is_available() else 'cpu',
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
        
    def get_model_and_diffusion(self, steps, use_gdm):
        if use_gdm:
            diffusion_config = self.config.diffusion_config.gaussian_diffusion
        else: 
            diffusion_config = self.config.diffusion_config.variational_diffusion
        model_config = self.config.model_config
        
        if use_gdm:
            betas, alphas = get_beta_schedule(diffusion_config.noise_schedule, diffusion_config.steps)
            betas = betas.to(self.device)
            betas = betas.to(torch.float32)
            if alphas is not None:
                alphas.to(self.device)
                alphas = alphas.to(torch.float32)
            diffusion = GaussianDiffusion(steps=diffusion_config.steps, betas=betas, alphas=alphas,
                                          objective=diffusion_config.objective, loss_type=diffusion_config.loss_type,
                                          device=self.device, cfg_dropout_proba=diffusion_config.cfg_dropout_proba,
                                          embedding_scale=diffusion_config.embedding_scale,
                                          batch_cfg=diffusion_config.batch_cfg, scale_cfg=diffusion_config.scale_cfg,
                                          sampling_timesteps=steps, use_fp16=False)
        else:   
            diffusion = VDM(loss_type=diffusion_config.loss_type, device=self.device, cfg_dropout_proba=diffusion_config.cfg_dropout_proba,
                            embedding_scale=diffusion_config.embedding_scale, 
                            batch_cfg=diffusion_config.batch_cfg, scale_cfg=diffusion_config.scale_cfg,
                            use_fp16=False)
        
        config_dict = {k: v for k, v in model_config.__dict__.items() if not k.startswith('__') and not callable(v)}
        context_embedding_features = config_dict.pop('context_embedding_features', None)
        context_embedding_max_length = config_dict.pop('context_embedding_max_length', None)
        
        model = UNetCFG1d(context_embedding_features=context_embedding_features, 
                     context_embedding_max_length=context_embedding_max_length, 
                     **config_dict).to(self.device)
        
        #model, _, _, _ = load_checkpoint(self.ckpt_path, model)
        model.eval()
        diffusion.eval()
        
        return diffusion, model
        
    def generate(self, prompt, seed=-1, steps=100, batch_size=1, seconds=30, use_gdm=False,
                 task='text_guided', init_audio=None, init_audio_sr=None, inpainting_scope=None):
        
        seed = seed if seed != -1 else np.random.randint(0, 2**32 -1)
        torch.manual_seed(seed)
        self.batch_size = batch_size
        
        diffusion, model = self.get_model_and_diffusion(steps, use_gdm)
                
        if init_audio is not None and init_audio.size() != 3:
            init_audio = init_audio.repeat(batch_size, 1, 1)
        
        if init_audio is None:
            flag = True
            sample_length = seconds * self.sample_rate
            shape = (batch_size, self.audio_encoder.channels, sample_length)
            init_audio = torch.zeros(shape)
            init_audio_sr = self.sample_rate
        
        init_audio = convert_audio(init_audio, init_audio_sr, self.sample_rate, self.audio_encoder.channels)

        if task == 'text_guided':
            mask = self.get_mask(sample_length, 0, seconds, batch_size)
            masked_input = init_audio * mask
            causal = False
        elif task == 'music_inpaint':
            mask = self.get_mask(sample_length, inpainting_scope[0], inpainting_scope[1], batch_size)
            inpaint_input = init_audio * mask
            masked_input = inpaint_input
            causal = False
        elif task == 'music_cont':
            cont_length = sample_length - init_audio.size(2)
            cont_start = init_audio.size(2)
            mask = self.get_mask(sample_length, cont_start/self.sample_rate, seconds, batch_size)
            cont_audio = torch.randn(batch_size, self.audio_encoder.channels, cont_length, device=self.device)
            cont_audio = cont_audio * mask[:, cont_start:]
            masked_input = torch.cat([init_audio, cont_audio], dim=2)     
            causal = True   
        
        with torch.no_grad():
            init_emb = self.get_emb(init_audio).to(self.device)
            emb_shape = init_emb.shape
            if flag:
                init_emb = None
            masked_emb = self.get_emb(masked_input).to(self.device)
            mask = mask.to(self.device)
            
            mask = torch.nn.functional.interpolate(mask, size=(emb_shape[2]))
            
            batch_metadata = [{'prompt': prompt} for _ in range(batch_size)]
            conditioning = self.conditioner(batch_metadata, self.device)
            conditioning['masked_input'] = masked_emb
            conditioning['mask'] = mask
            conditioning = self.get_conditioning(conditioning)
            
            sample_embs = diffusion.sample(model, emb_shape, conditioning, causal, init_data=init_emb)
            sample_embs = sample_embs.to('cpu')
            samples = self.audio_encoder.decoder(sample_embs)
        
        return samples
    
    def get_mask(self, sample_size, start, end, batch_size):
        masks = []
        maskstart = math.floor(start * self.sample_rate)
        maskend = math.ceil(end * self.sample_rate)
        mask = torch.ones((1, 1, sample_size))
        mask[:, :, maskstart:maskend] = 0
        masks.append(mask)
        mask = torch.concat(masks * batch_size, dim=0)
    
        return mask
    
    def get_emb(self, audio):
        encoded_frames = self.audio_encoder.encode(audio)
        codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1)
        codes = codes.transpose(0, 1)
        emb = self.audio_encoder.quantizer.decode(codes)
        return emb    
            
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
            concated_tensors = []
            for key in self.input_concat_ids:
                tensor = cond[key][0]
                
                if tensor.ndim == 2:
                    tensor = tensor.unsqueeze(0)
                    tensor = tensor.expand(self.batch_size, -1, -1)
                
                concated_tensors.append(tensor)
            # Concatenate all input concat conditioning inputs over the channel dimension
            # Assumes that the input concat conditioning inputs are of shape (batch, channels, seq)
            #input_concat_cond = torch.cat([cond[key][0] for key in self.input_concat_ids], dim=1)
            #For some reason, the BATCH component is removed. I don't know why.
            input_concat_cond = torch.cat(concated_tensors, dim=1)
            
        return {
            "cross_attn_cond": cross_attention_input,
            "cross_attn_masks": cross_attention_masks,
            "global_cond": global_cond,
            "input_concat_cond": input_concat_cond
        }
        
def save_audio_tensor(audio_tensor: torch.Tensor, file_path: str, sample_rate: int = 48000):
    print(f'Saving audio to {file_path}')
    """
    Saves an audio tensor to a file.
    Params:
        audio_tensor: torch.Tensor, The audio data to save.
        file_path: str, The path to the file where the audio will be saved.
        sample_rate: int, The sample rate of the audio data.
    Returns:
        None
    """
    # Ensure the tensor is on the CPU before saving
    audio_tensor = audio_tensor.detach()
    print(f'audio_tensor.shape: {audio_tensor.shape}')
    if audio_tensor.ndim == 3:
        audio_tensor = audio_tensor.squeeze(0)  # Remove the batch dimension
    # Use torchaudio to save the tensor as an audio file
    import torchaudio
    torchaudio.save(file_path, audio_tensor, sample_rate)
    print(f'Saved audio to {file_path}')
        
if __name__ == '__main__':
    jen1 = Jen1(ckpt_path=None)
    prompt = 'a beautiful song'
    samples = jen1.generate(prompt=prompt)
    save_audio_tensor(samples, 'samples.wav')