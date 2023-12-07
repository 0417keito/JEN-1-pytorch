import torch
import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from einops import reduce
from tqdm import tqdm

class VDM(nn.Module):
    def __init__(
            self,
            *,
            loss_type,
            device,
            cfg_dropout_proba=0.1,
            embedding_scale=0.8,
            batch_cfg=False,
            scale_cfg=False,
            use_fp16=False,
    ):
        super().__init__()
        self.device = device
        self.cfg_dropout_proba = cfg_dropout_proba
        self.embedding_scale = embedding_scale
        self.batch_cfg = batch_cfg
        self.scale_cfg = scale_cfg
        self.use_fp16 = use_fp16
        
        assert loss_type in {'l1', 'l2'}
        if loss_type == 'l1':
            self.loss_fn = F.l1_loss
        elif loss_type == 'l2':
            self.loss_fn = F.mse_loss
        
    def get_alpha_sigma(self, t):
        self.alphas = torch.cos(t * math.pi / 2)
        self.sigmas = torch.sin(t * math.pi / 2) 
    
    @torch.no_grad()
    def p_sample(self, x, time, time_next, model, conditioning, causal):
        v_pred = model(x, time, embedding=conditioning['cross_attn_cond'],
                            embedding_mask=conditioning['cross_attn_masks'],
                            embedding_scale=self.embedding_scale,
                            embedding_mask_proba=self.cfg_dropout_proba,
                            features=conditioning['global_cond'],
                            channels_list=[conditioning['input_concat_cond']],
                            batch_cfg=self.batch_cfg, scale_cfg=self.scale_cfg,
                            causal=causal)
        x_pred = self.alphas[time] * x - self.sigmas[time] * v_pred
        noise_pred = self.sigmas[time] * x + self.alphas[time] * v_pred
        x = self.alphas[time_next] * x_pred + self.sigmas[time_next] * noise_pred
         
        return x
    
    @torch.no_grad()
    def p_sample_loop(self, model, shape, conditioning, step=1000, return_all_timesteps=False, init_data=None, causal=False):
        audio = torch.randn(shape, device=self.device)
        if init_data is not None:
            audio = audio + init_data
        audios = [audio]
        steps = torch.linspace(1., 0., step + 1, device=self.device)
        self.get_alpha_sigma(steps)
        
        for i in tqdm(range(step), desc = 'sampling loop time step', total=step):
            times = steps[i]
            times_next = steps[i + 1]
            audio = self.p_sample(audio, times, times_next, model, conditioning, causal=causal)
            audios.append(audio)
        
        ret = audio if not return_all_timesteps else torch.stack(audios, dim=1)
        return ret
    
    @torch.no_grad()
    def sample(self, model, shape, conditioning, step=100, return_all_timesteps=False, causal=False, init_data=None):
        return self.p_sample_loop(model, shape, conditioning, step, return_all_timesteps=return_all_timesteps, init_data=init_data, causal=causal)
        
    def q_sample(self, x_start, times, noise=None):
        '''
        calculating q(x_t | x_0).
        '''
        if noise is None:
            noise = torch.rand_like(x_start)
        alphas, sigmas = torch.cos(times * math.pi / 2), torch.sin(times * math.pi / 2) 
        x_noised = x_start * alphas + noise * sigmas
        
        return x_noised, alphas, sigmas
        
    def training_loosses(self, model, x_start, conditioning, noise=None, causal=False):
        if noise is None:
            noise = torch.rand_like(x_start)
        
        times = torch.rand(x_start.shape[0]).requires_grad_(True).to(device=self.device)
        x_t, alphas, sigmas = self.q_sample(x_start, times, noise=noise)
        with autocast(enabled=self.use_fp16):
            model_out = model(x_t, times, embedding=conditioning['cross_attn_cond'],
                            embedding_mask=conditioning['cross_attn_masks'],
                            embedding_scale=self.embedding_scale,
                            embedding_mask_proba=self.cfg_dropout_proba,
                            features=conditioning['global_cond'],
                            channels_list=[conditioning['input_concat_cond']],
                            batch_cfg=self.batch_cfg, scale_cfg=self.scale_cfg,
                            causal=causal)
            target = noise * alphas - x_t * sigmas
            loss = self.loss_fn(model_out, target, reduction='none')
            loss = reduce(loss, 'b ... -> b', 'mean')
        return loss.mean()
            