import torch
import numpy as np
import logging
from torch.utils.data import DataLoader, Dataset
from .files_dataset import FilesAudioDataset
from .collate import collate
from .sampler import DistributedBucketSampler

class OffsetDataset(Dataset):
    def __init__(self, dataset, start, end, test=False):
        super().__init__()
        self.dataset = dataset
        self.start = start
        self.end = end
        self.test = test
        assert 0 <= self.start < self.end <= len(self.dataset)

    def __len__(self):
        return self.end - self.start

    def __getitem__(self, item):
        return self.dataset.get_item(self.start + item, test=self.test)
    
    def get_dur(self, item):
        self.dataset.get_dur(item)

class DataProcessor():
    def __init__(self, batch_size: int, num_workers: int, 
                 dataset_dir: str, sr: int, channels: int, train_test_split: float,  
                 min_duration: int, max_duration: int, sample_length: int, 
                 aug_shift: bool, cache_dir: str, device: str, n_gpus: int, rank: int):
        self.dataset = FilesAudioDataset(sr=sr, channels=channels, min_duration=min_duration,
                                         max_duration=max_duration, sample_length=sample_length,
                                         cache_dir=cache_dir, aug_shift=aug_shift, device=device, 
                                         dataset_dir=dataset_dir)
        self.create_datasets(train_test_split=train_test_split)
        self.create_samplers(batch_size=batch_size, n_gpus=n_gpus, rank=rank)
        self.create_data_loaders(batch_size=batch_size, num_workers=num_workers)
        self.print_stats()

    def set_epoch(self, epoch):
        self.train_sampler.set_epoch(epoch)
        self.valid_sampler.set_epoch(epoch)

    def create_datasets(self, train_test_split):
        train_len = int(len(self.dataset) * train_test_split)
        self.train_dataset = OffsetDataset(self.dataset, 0, train_len, test=False)
        self.valid_dataset = OffsetDataset(self.dataset, train_len, len(self.dataset), test=True)

    def create_samplers(self, batch_size, n_gpus, rank):
        self.train_sampler = DistributedBucketSampler(self.train_dataset, batch_size=batch_size,
                                                      boundaries=[], num_replicas=n_gpus,
                                                      rank=rank, shuffle=True)
        self.valid_sampler = DistributedBucketSampler(self.valid_dataset, batch_size=batch_size,
                                                      boundaries=[], num_replicas=n_gpus,
                                                      rank=rank, shuffle=True)

    def create_data_loaders(self, batch_size, num_workers):

        logging.info('Creating Data Loader')
        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, 
                                       num_workers=num_workers, sampler=self.train_sampler, 
                                       pin_memory=False, drop_last=True)
        self.valid_loader = DataLoader(self.valid_dataset, batch_size=batch_size, 
                                       num_workers=num_workers, sampler=self.valid_sampler, 
                                       pin_memory=False, drop_last=False)

    def print_stats(self):
        logging.info(f"Train {len(self.train_dataset)} samples. Test {len(self.valid_dataset)} samples")