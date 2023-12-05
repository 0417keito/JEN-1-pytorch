import torch
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from einops import reduce
from tqdm import tqdm

from utils.script_util import extract, default, identity


class GaussianDiffusion(nn.Module):
    def __init__(
            self,
            *,
            steps,
            betas,
            objective,
            loss_type,
            device,
            cfg_dropout_proba=0.1,
            embedding_scale=0.8,
            batch_cfg=False,
            scale_cfg=False,
            sampling_timesteps=None,
            ddim_sampling_eta=1.,
            use_fp16=False,
            alphas=None
    ):
        super().__init__()
        self.objective = objective
        self.device = device
        self.cfg_dropout_proba = cfg_dropout_proba
        self.embedding_scale = embedding_scale
        self.batch_cfg = batch_cfg
        self.scale_cfg = scale_cfg
        self.use_fp16 = use_fp16
        assert objective in {'noise', 'x0',
                             'v'}, 'objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v)'
        assert loss_type in {'l1', 'l2'}
        if loss_type == 'l1':
            self.loss_fn = F.l1_loss
        elif loss_type == 'l2':
            self.loss_fn = F.mse_loss

        self.num_timesteps = steps
        self.sampling_timesteps = default(sampling_timesteps, self.num_timesteps)

        assert self.sampling_timesteps <= self.num_timesteps
        self.is_ddim_sampling = self.sampling_timesteps < self.num_timesteps
        self.ddim_sampling_eta = ddim_sampling_eta
        self.betas = betas
        assert len(betas.shape) == 1, 'betas must be 1-D'
        assert (betas > 0).all() and (betas <= 1).all()

        if alphas is not None:
            alphas = alphas
        else:
            alphas = 1 - betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # calculations for diffusion q(x_t | x_{t_1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t_1} | x_t, x_0)
        self.posterior_variance = (
                betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = torch.log(
            torch.cat([self.posterior_variance[1].unsqueeze(0), self.posterior_variance[1:]])
        )
        self.posterior_mean_coef1 = (
                betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
                (1.0 - self.alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - self.alphas_cumprod)
        )

    def predict_start_from_noise(self, x_t, t, noise):
        return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
                (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
                extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x, t, model, conditioning=None, clip_x_start=False, causal=False):
        with autocast(enabled=self.use_fp16):
            model_out = model(x, t, embedding=conditioning['cross_attn_cond'],
                            embedding_mask=conditioning['cross_attn_masks'],
                            embedding_scale=self.embedding_scale,
                            embedding_mask_proba=self.cfg_dropout_proba,
                            features=conditioning['global_cond'],
                            channels_list=[conditioning['input_concat_cond']],
                            batch_cfg=self.batch_cfg, scale_cfg=self.scale_cfg,
                            causal=causal)
        maybe_clip = partial(torch.clamp, min=-1, max=1.) if clip_x_start else identity

        if self.objective == 'noise':
            pred_noise = model_out
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)
        elif self.objective == 'x0':
            x_start = model_out
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)
        elif self.objective == 'v':
            v = model_out
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return pred_noise, x_start

    def p_mean_variance(self, x, t, model, conditioning=None, cliped_denoised=True):
        pred_noise, x_start = self.model_predictions(x, t, model, conditioning)

        if cliped_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def p_sample(self, x, t, model, conditioning):
        b = x.shape[0]
        batched_times = torch.full((b,), t, device=self.device, dtype=torch.long)
        model_mean, variance, model_log_variance, x_start = self.p_mean_variance(
            x=x, t=batched_times, model=model, conditioning=conditioning, cliped_denoised=True
        )

        noise = torch.rand_like(x) if t > 0 else 0.
        pred_audio = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_audio, x_start

    @torch.no_grad()
    def p_sample_loop(self, model, shape, conditioning, return_all_timesteps=False, init_data=None):
        audio = torch.randn(shape, device=self.device)
        if init_data is not None:
            audio = audio + init_data
        audios = [audio]

        x_start = None

        for t in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            audio, x_start = self.p_sample(audio, t, model, conditioning)
            audios.append(audio)

        ret = audio if not return_all_timesteps else torch.stack(audios, dim=1)
        return ret

    @torch.no_grad()
    def ddim_sample(self, model, shape, conditioning, return_all_timesteps=False, causal=False, init_data=None):
        batch = shape[0]
        device = self.device
        total_timesteps = self.num_timesteps
        sampling_timesteps = self.sampling_timesteps
        eta = self.ddim_sampling_eta
        objective = self.objective

        times = torch.linspace(-1, total_timesteps - 1,
                               steps=sampling_timesteps + 1)  # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))

        audio = torch.randn(shape, device=device)
        if init_data is not None:
            audio = audio + init_data
        audios = [audio]

        x_start = None

        for time, time_next in tqdm(time_pairs, desc='sampling loop time stes'):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            pred_noise, x_start = self.model_predictions(audio, time_cond, model, conditioning, clip_x_start=True, causal=causal)

            audios.append(audio)

            if time_next < 0:
                audio = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(audio)

            audio = x_start * alpha_next.sqrt() + \
                    c * pred_noise + \
                    sigma * noise

        ret = audio if not return_all_timesteps else torch.stack(audios, dim=1)
        return ret

    @torch.no_grad()
    def sample(self, model, shape, conditioning, return_all_timesteps=False, causal=False, init_data=None):
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        return sample_fn(model, shape, conditioning, return_all_timesteps=return_all_timesteps, causal=causal, init_data=init_data)

    def q_sample(self, x_start, t, noise=None):
        '''
        calculating q(x_t | x_0).
        '''
        if noise is None:
            noise = torch.rand_like(x_start)
        assert noise.shape == x_start.shape

        return (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def training_loosses(self, model, x_start, t, conditioning, noise=None, causal=False):
        if noise is None:
            noise = torch.rand_like(x_start)
        x_t = self.q_sample(x_start, t, noise=noise)
        
        with autocast(enabled=self.use_fp16):
            model_out = model(x_t, t, embedding=conditioning['cross_attn_cond'],
                            embedding_mask=conditioning['cross_attn_masks'],
                            embedding_scale=self.embedding_scale,
                            embedding_mask_proba=self.cfg_dropout_proba,
                            features=conditioning['global_cond'],
                            channels_list=[conditioning['input_concat_cond']],
                            batch_cfg=self.batch_cfg, scale_cfg=self.scale_cfg,
                            causal=causal)

        if self.objective == 'noise':
            target = noise
        elif self.objective == 'x0':
            target = x_start
        elif self.objective == 'v':
            target = (extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
                      extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start)
        else:
            raise ValueError(f'unknown objective {self.objective}')

        loss = self.loss_fn(model_out, target, reduction='none')
        loss = reduce(loss, 'b ... -> b', 'mean')
        return loss.mean()
