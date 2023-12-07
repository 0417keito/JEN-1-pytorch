import glob
import os
from inspect import isfunction

import torch

from jen1.conditioners import MultiConditioner
from jen1.diffusion.gdm.noise_schedule import get_beta_schedule
from utils.conditioner_config import ConditionerConfig
from utils.config import Config

def exists(x):
    return x is not None


def default(x, y):
    if exists(x):
        return x
    return y() if isfunction(y) else y


def identity(t, *args, **kwargs):
    return t


def group_dict_by_prefix(prefix: str, d):
    return_dicts = ({}, {})
    for key in d.keys():
        no_prefix = int(not key.startswith(prefix))
        return_dicts[no_prefix][key] = d[key]
    return return_dicts


def groupby(prefix, d, keep_prefix):
    kwargs_with_prefix, kwargs = group_dict_by_prefix(prefix, d)
    if keep_prefix:
        return kwargs_with_prefix, kwargs
    kwargs_no_prefix = {k[len(prefix):]: v for k, v in kwargs_with_prefix.items()}
    return kwargs_no_prefix, kwargs


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def get_latest_checkpoint(dir_path, regex='Jen1_step_*.pth'):
    f_list = glob.glob(os.path.join(dir_path, regex))
    f_list.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    if len(f_list) == 0:
        return 'checkpoint is None'
    x = f_list[-1]
    return x


def save_checkpoint(model, optimizer, lr, iteration, checkpoint_path, logger):
    logger.info(f'Saving model and optimizer state at iteration {iteration} to {checkpoint_path}')
    if hasattr(model, 'module'):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    torch.save({'model': state_dict,
                'epoch': iteration,
                'optimizer': optimizer.state_dict(),
                'learning_rate': lr}, checkpoint_path)


def load_checkpoint(checkpoint_path, model, logger=None, optimizer=None):
    assert os.path.isfile(checkpoint_path)
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    epoch = checkpoint_dict['epoch']
    learning_rate = checkpoint_dict['learning_rate']
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint_dict['optimizer'])
    saved_state_dict = checkpoint_dict['model']
    if hasattr(model, 'module'):
        state_dict = model.module_state_dict()
    else:
        state_dict = model.state_dict()
    new_state_dict = {}
    for k, v in state_dict.items():
        try:
            new_state_dict[k] = saved_state_dict[k]
        except:
            try:
                new_state_dict[k] = saved_state_dict[f'_orig_mod.{k}']
            except:
                if logger is not None:
                    logger.info("%s is not in the checkpoint" % k)
                new_state_dict[k] = v
    if hasattr(model, 'module'):
        model.module.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(new_state_dict)
    if logger is not None:
        logger.info(f"Loaded checkpoint '{checkpoint_path}' (epoch {epoch})")
    return model, optimizer, learning_rate, epoch


def load_model_diffsize(checkpoint_path, model):
    assert os.path.isfile(checkpoint_path)
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')['model']

    if hasattr(model, 'module'):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()

    for k, v in checkpoint_dict.items():
        if k in state_dict and state_dict[k].size() == v.size():
            state_dict[k] = v
        else:
            k = k.replace('_orig_mod.', '')
            if k in state_dict and state_dict[k].size() == v.size():
                state_dict[k] = v
            else:
                print('[WARNING] Parameter mismatch :', k)

    if hasattr(model, 'module'):
        model.module.load_state_dict(state_dict, strict=False)
    else:
        model.load_state_dict(state_dict, strict=False)
    return model


def create_multi_conditioner(config: ConditionerConfig):
    conditioners = {}
    cond_dim = config.cond_dim
    default_keys = config.default_keys

    for conditioner_type in config.conditioning_type:
        if conditioner_type == 't5':
            from jen1.conditioners import T5Conditioner
            conditioner_config = config.t5_config
            config_dict = {k: v for k, v in conditioner_config.__dict__.items() if not k.startswith('__') and not callable(v)}
            id = config_dict.pop('id', None)
            conditioners[id] = T5Conditioner(output_dim=cond_dim, **config_dict)
        elif conditioner_type == 'int':
            from jen1.conditioners import IntConditioner
            conditioner_config = config.int_config
            config_dict = {k: v for k, v in conditioner_config.__dict__.items() if not k.startswith('__') and not callable(v)}
            id = config_dict.pop('id', None)
            conditioners[id] = IntConditioner(output_dim=cond_dim, **config_dict)
        elif conditioner_type == 'number':
            from jen1.conditioners import NumberConditioner
            conditioner_config = config.number_config
            config_dict = {k: v for k, v in conditioner_config.__dict__.items() if not k.startswith('__') and not callable(v)}
            id = config_dict.pop('id', None)
            conditioners[id] = NumberConditioner(output_dim=cond_dim, **config_dict)
        else:
            NotImplementedError('Invalid conditioner type!')

        return MultiConditioner(conditioners, default_keys=default_keys)


def create_model_and_diffusion(config: Config, sampling_steps=None):
    device = config.device
    model = create_model(config).to(device)
    diffusion = create_diffusion(config, sampling_steps).to(device)

    return model, diffusion


def create_diffusion(config: Config, sampling_steps=None):
    use_fp16 = config.use_fp16
    diffusion_type = config.diffusion_type
    if diffusion_type.lower() == 'gdm':
        diffusion_config = config.diffusion_config.gaussian_diffusion
        return create_gaussian_diffusion(steps=diffusion_config.steps,
                                         noise_schedule=diffusion_config.noise_schedule,
                                         objective=diffusion_config.objective,
                                         loss_type=diffusion_config.loss_type,
                                         device=diffusion_config.device,
                                         cfg_dropout_proba=diffusion_config.cfg_dropout_proba,
                                         embedding_scale=diffusion_config.embedding_scale,
                                         batch_cfg=diffusion_config.batch_cfg,
                                         scale_cfg=diffusion_config.scale_cfg,
                                         sampling_steps=sampling_steps,
                                         use_fp16=use_fp16,
                                         composer=diffusion_config.composer,
                                         demix_list=diffusion_config.demix_list)
    elif diffusion_type.lower() == 'vdm':
        use_fp16 = config.use_fp16
        diffusion_config = config.diffusion_config.variational_diffusion
        return create_variational_diffusion(diffusion_config.loss_type,
                                            diffusion_config.device,
                                            diffusion_config.cfg_dropout_proba,
                                            embedding_scale=diffusion_config.embedding_scale,
                                            batch_cfg=diffusion_config.batch_cfg,
                                            scale_cfg=diffusion_config.scale_cfg,
                                            use_fp16=use_fp16)

def create_gaussian_diffusion(steps=1000,
                              noise_schedule='linear',
                              objective='v',
                              loss_type='l2',
                              device='cuda',
                              cfg_dropout_proba=0.1,
                              embedding_scale=1,
                              batch_cfg=False,
                              scale_cfg=False,
                              sampling_steps=None,
                              use_fp16=False,
                              composer=False,
                              demix_list=[],
                              ):
    from jen1.diffusion.gdm.gdm import GaussianDiffusion
    betas_dict = None
    if composer and demix_list is not None:
        betas_dict = {}
        for demix in demix_list:
            betas, alphas = get_beta_schedule(noise_schedule, steps)
            betas = betas.to(device)
            betas = betas.to(torch.float32)
            if alphas is not None:
                alphas = alphas.to(device)
                alphas = alphas.to(torch.float32)
            betas_dict[demix] = {'betas': betas, 'alphas': alphas}
    
    betas, alphas = get_beta_schedule(noise_schedule, steps)
    betas = betas.to(device)
    betas = betas.to(torch.float32)
    if alphas is not None:
        alphas = alphas.to(device)
        alphas = alphas.to(torch.float32)
    return GaussianDiffusion(
        steps=steps,
        betas=betas,
        alphas=alphas,
        objective=objective,
        loss_type=loss_type,
        device=device,
        cfg_dropout_proba=cfg_dropout_proba,
        embedding_scale=embedding_scale,
        batch_cfg=batch_cfg,
        scale_cfg=scale_cfg,
        sampling_timesteps=sampling_steps,
        use_fp16=use_fp16,
        betas_dict=betas_dict,
        composer=composer,
    ).to(device)
    
def create_variational_diffusion(loss_type='l2',
                                 device='cuda',
                                 cfg_dropout_proba=0.1,
                                 embedding_scale=1,
                                 batch_cfg=False,
                                 scale_cfg=False,
                                 use_fp16=False,
                                 ):
    from jen1.diffusion.vdm.vdm import VDM
    return VDM(
        loss_type=loss_type,
        device=device,
        cfg_dropout_proba=cfg_dropout_proba,
        embedding_scale=embedding_scale,
        batch_cfg=batch_cfg,
        scale_cfg=scale_cfg,
        use_fp16=use_fp16
    ).to(device)


def create_model(config: Config):
    from jen1.model.model import UNetCFG1d
    model_config = config.model_config
    device = config.device
    config_dict = {k: v for k, v in model_config.__dict__.items() if not k.startswith('__') and not callable(v)}
    context_embedding_features = config_dict.pop('context_embedding_features', None)
    context_embedding_max_length = config_dict.pop('context_embedding_max_length', None)
    
    assert context_embedding_features is not None, "context_embedding_features is not set in model_config"
    assert context_embedding_max_length is not None, "context_embedding_max_length is not set in model_config"

    return UNetCFG1d(context_embedding_features=context_embedding_features, 
                     context_embedding_max_length=context_embedding_max_length, 
                     **config_dict).to(device)
