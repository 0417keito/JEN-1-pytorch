import enum
import math
import torch
import numpy as np


def get_beta_schedule(schedule_name, num_diffusion_timesteps):
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return torch.linspace(
            beta_start, beta_end, num_diffusion_timesteps), None
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    elif schedule_name == 'angle':
        return angle_schedule(num_diffusion_timesteps)
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return torch.tensor(betas), None

def angle_schedule(num_diffusion_timesteps, vmax=1.0, vmin=0.0):
    scale = 1000 / num_diffusion_timesteps
    
    t = (vmax - vmin) * torch.rand(num_diffusion_timesteps) + vmin
    angle = t * math.pi / 2
    alpha, beta = torch.cos(angle), torch.sin(angle)
    return beta, alpha