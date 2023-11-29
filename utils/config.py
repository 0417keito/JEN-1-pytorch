import torch
from dataclasses import dataclass
from .conditioner_config import ConditionerConfig

@dataclass
class DataConfig:
    sr = 48000 #audio sampling rate
    channels = 1 #audio channels
    batch_size = 32 
    num_workers = 1 
    dataset_dir = '/home/keito/JEN-1-pytorch/data'
    train_test_split = 0.9
    min_duration = 30 # lower limit on the duration of the audio file to be used
    max_duration = 90 # maximum duration of the audio file to be used
    num_buckets = 10 # Number of buckets used for sampling. Putting together audio files of similar duration.
    sample_length = 480000 # sample length = sampling rate × audio_duration(s)
    aug_shift =  False # whether using data augmentation
    cache_dir = '/home/keito/JEN-1-pytorch/data' # to save the duration of the audio file
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

@dataclass
class GDM_Config:
    steps = 1000 #num timesteps
    noise_schedule = 'linear' #noise scheduler
    objective = 'v' # training objectives optional['noise', 'x0', 'v']
    loss_type='l2' #loss type
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg_dropout_proba = 0.2 
    embedding_scale = 0.8
    batch_cfg = True
    scale_cfg = True

@dataclass
class VDM_Config: ...

@dataclass
class EDM_Config: ...

@dataclass
class DiffusionConfig:
    gaussian_diffusion = GDM_Config
    variational_diffusion = VDM_Config
    elucidated_diffusion = EDM_Config
    
@dataclass
class ModelConfig: 
    in_channels = 128 #number of potential embedded channels to be entered
    channels = 128
    multipliers = [1, 1, 1, 1, 2, 2, 4, 4, 8, 8] # indicates how many times the input channels of each block of UNet are in_channels. 
    factors = [1, 2, 1, 2, 1, 2, 1, 2, 1] # convolution layer parameters for each block
    num_blocks = [1, 3, 3, 3, 3, 3, 3, 3, 1] # number of ResNet Blocks in each block
    attentions = [0, 0, 0, 0, 1, 0, 1, 0, 0] # number of Attention layers in each block
    patch_size = 1
    resnet_groups = 8
    use_context_time = True
    kernel_multiplier_downsample: int = 2
    use_nearest_upsample = False
    use_skip_scale = True
    use_snake = False
    use_stft = False
    use_stft_context = False
    use_xattn_time = True
    out_channels = 128
    context_features = 1024
    context_features_multiplier = 4
    context_channels = [129]
    context_embedding_features = 1
    context_embedding_max_length = 1
    attention_heads = 8
    attention_multiplier = 1

@dataclass
class OptimizerConfig: 
    lr = 3e-5
    beta_1 = 0.9
    beta_2 = 0.95
    weight_decay = 0.1
    grad_clip = 0.7

@dataclass
class Config:
    save_dir = '/home/keito/JEN-1-pytorch/data'
    log_dir = '/home/keito/JEN-1-pytorch/data'
    use_ddp = True
    use_fp16 = False
    use_ema = False
    seed = 4996
    tasks = ['text_guided', 'music_inpaint', 'music_cont']
    num_epoch = 100
    log_interval = 10
    eval_interval = 20
    is_fintuning = False
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    diffusion_type = 'gdm'
    dataset_config = DataConfig
    diffusion_config = DiffusionConfig
    model_config = ModelConfig
    optimizer_config = OptimizerConfig
    conditioner_config = ConditionerConfig