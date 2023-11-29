import torch.distributed as dist
import torch.multiprocessing as mp
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from dataset.data_processor import DataProcessor
from trainer import UnifiedMultiTaskTrainer
from utils.config import OptimizerConfig, DataConfig
from utils.logger import get_logger
from utils.script_util import *


def main(config: Config):
    n_gpus = torch.cuda.device_count()

    if config.use_ddp:
        mp.spawn(run, nprocs=n_gpus, args=(n_gpus, config))
    else:
        run(rank=0, n_gpus=1, config=config)


def run(rank, n_gpus, config: Config):
    if rank == 0:
        logger = get_logger(config.log_dir)
        logger.info(config)
        writer = SummaryWriter(log_dir=config.log_dir)
        writer_val = SummaryWriter(log_dir=os.path.join(config.log_dir, 'val'))

    dist.init_process_group(backend='nccl', init_method='env://',
                                world_size=n_gpus, rank=rank)

    torch.manual_seed(config.seed)
    torch.cuda.set_device(rank)

    # create dataset
    logger.info('creating data loader...')
    data_config = Config.dataset_config
    dataprocessor = DataProcessor(batch_size=data_config.batch_size,
                                  num_workers=data_config.num_workers,
                                  dataset_dir=data_config.dataset_dir,
                                  sr=data_config.sr,
                                  channels=data_config.channels,
                                  train_test_split=data_config.train_test_split,
                                  min_duration=data_config.min_duration,
                                  max_duration=data_config.max_duration,
                                  sample_length=data_config.sample_length,
                                  aug_shift=data_config.aug_shift,
                                  cache_dir=data_config.cache_dir,
                                  device=data_config.device,
                                  n_gpus=n_gpus,
                                  rank=rank)

    train_dl = dataprocessor.train_loader
    if rank == 0:
        valid_dl = dataprocessor.valid_loader

    # create model and diffusion
    logger.info('creating model and diffusion...')
    model, diffusion = create_model_and_diffusion(config)
    conditioner = create_multi_conditioner(config.conditioner_config)

    # create optimizer
    optimizer_config: OptimizerConfig = config.optimizer_config
    grad_clip = optimizer_config.grad_clip
    optim = torch.optim.AdamW(params=model.parameters(),
                              lr=optimizer_config.lr,
                              betas=(optimizer_config.beta_1, optimizer_config.beta_2),
                              weight_decay=optimizer_config.weight_decay
                              )

    # confirm checkpoint
    latest_checkpoint_path = get_latest_checkpoint(config.save_dir)
    if os.path.isfile(latest_checkpoint_path):
        try:
            model, optim, _, epoch_str = load_checkpoint(latest_checkpoint_path, model,
                                                         logger, optim)
            global_step = (epoch_str - 1) * len(train_dl)
            print('Resume learning mode')
        except:
            model = load_model_diffsize(latest_checkpoint_path, model)
            epoch_str = 1
            global_step = 0.
            print('Resume learning mode')
    elif config.is_fintuning:
        pass
    else:
        epoch_str = 1
        global_step = 0
        print('Initial state mode')
    global_step = (epoch_str - 1) * len(train_dl)

    # creating scheduler
    lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer=optim, last_epoch=epoch_str - 2)

    scaler = GradScaler(enabled=config.use_fp16)

    if config.use_ddp:
        model = DDP(model, device_ids=[rank])

    logger.info('training...')
    if rank == 0:
        trainer = UnifiedMultiTaskTrainer(
            config=config,
            rank=rank,
            epoch_str=epoch_str,
            global_step=global_step,
            # config=config,
            model=model,
            diffusion=diffusion,
            conditioner=conditioner,
            dls=[train_dl, valid_dl],
            optimizer=optim,
            lr_scheduler=lr_scheduler,
            scaler=scaler,
            logger=logger,
            writers=[writer, writer_val],
            grad_clip=grad_clip
        )
    else:
        trainer = UnifiedMultiTaskTrainer(
            config=config,
            rank=rank,
            model=model,
            diffusion=diffusion,
            conditioner=conditioner,
            dls=[train_dl, None],
            optimizer=optim,
            lr_scheduler=lr_scheduler,
            scaler=scaler,
            logger=logger,
            writers=[writer, None],
            grad_clip=grad_clip
        )
    trainer.train_loop()


if __name__ == '__main__':
    config = Config()
    config.num_epoch = 1000
    config.save_dir = './checkpoints'
    config.log_dir = './logs'
    config.use_ddp = False
    config.dataset_config = DataConfig()
    config.dataset_config.dataset_dir = './dataset'
    config.dataset_config.batch_size = 1
    main(config=Config)
