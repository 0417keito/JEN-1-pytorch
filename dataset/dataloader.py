import os
import time
import math
import random
import json
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader, random_split

from encodec import EncodecModel
from encodec.utils import convert_audio


class MusicDataset(Dataset):
    def __init__(self, dataset_dir, sr, channels, min_duration, max_duration,
                 sample_duration, aug_shift, device, same_folder=False):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.sr = sr
        self.channels = channels
        self.min_duration = min_duration 
        self.max_duration = max_duration
        self.sample_duration = sample_duration
        self.aug_shift = aug_shift
        self.device = device
        if same_folder:
            self.audio_files_dir = dataset_dir
            self.metadatas_dir = dataset_dir
        else:
            self.audio_files_dir = f'{dataset_dir}/audios'
            self.metadatas_dir = f'{dataset_dir}/metadata'
        self.init_dataset()
    
    def get_duration_sec(self, file): 
        wav, sr = torchaudio.load(file)
        duration_sec = wav.shape[-1] / sr
        return duration_sec
    
    def filter(self, audio_files, durations):
        keep = []
        self.audio_files = []
        for i in range(len(audio_files)):
            filepath = audio_files[i]
            if durations[i] / self.sr < self.min_duration:
                continue
            if durations[i] / self.sr >= self.max_duration:
                continue
            keep.append(i)
            self.audio_files.append(filepath)
        self.durations = [durations[i] for i in keep] # in (s)
        duration_tensor = torch.tensor(self.durations)
        self.cumsum = torch.cumsum(duration_tensor, dim=0) # in (s)
    
    def init_dataset(self):
        audio_files = os.listdir(self.audio_files_dir)
        audio_files = [f'{self.audio_files_dir}/{file}' for file in audio_files if file.endswith('.wav') or file.endswith('.mp3')]
        durations = [self.get_duration_sec(file) for file in audio_files]
        self.filter(audio_files=audio_files, durations=durations)
    
    def get_index_offset(self, item):
        half_interval = self.sample_duration // 2
        shift = random.randint(-half_interval, half_interval) if self.aug_shift else 0
        offset = item * self.sample_duration + shift
        midpoint = offset + half_interval
        assert 0 <= midpoint <= self.cumsum[-1], f'Midpoint {midpoint} of item beyond total length {self.cumsum[-1]}'
        index = torch.searchsorted(self.cumsum, midpoint)
        start, end = self.cumsum[index-1] if index > 0 else 0.0, self.cumsum[index]
        assert start <= midpoint <= end, f'Midpoint {midpoint} not inside interval [{start}, {end}] for index {index}'
        if offset > end - self.sample_duration:
            offset = max(start, offset - half_interval)
        elif offset < start:
            offset = min(end - self.sample_duration, offset + half_interval)
        assert start <= offset <= end - self.sample_duration, f'Offset {offset} not in [{start}, {end} for index {index}]'
        offset = offset - start
        return index, offset
    
    def get_song_chunk(self, index, offset):
        audio_file_path = self.audio_files[index]
        wav, sr = torchaudio.load(audio_file_path)
        
        start_sample = int(offset * sr)
        end_sample = start_sample + int(self.sample_duration * sr)
        chunk = wav[:, start_sample:end_sample]
        #chunk = chunk.unsqueeze(0)
        
        return chunk, sr
    
    def __len__(self):
        return len(self.durations)
        
    def __getitem__(self, item):
        index, offset = self.get_index_offset(item)
        chunk, sr = self.get_song_chunk(item, offset)
        song_name = os.path.splitext(os.path.basename(self.audio_files[index]))[0]
        if os.path.exists(f'{self.metadatas_dir}/{song_name}.json'):
            with open(f'{self.metadatas_dir}/{song_name}.json', 'r') as file:
                metadata = json.load(file)
        model = EncodecModel.encodec_model_48khz()
        chunk = convert_audio(chunk, sr, model.sample_rate, model.channels)
        chunk = chunk.unsqueeze(0)
        with torch.no_grad():
            encoded_frames = model.encode(chunk)
        chunk = chunk.mean(0, keepdim=True)
        codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1)
        codes = codes.transpose(0, 1)
        emb = model.quantizer.decode(codes)
        emb = emb.to(self.device)
        
        return chunk, metadata, emb

def collate(batch):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    audio, data, emb = zip(*batch)
    audio = torch.cat(audio, dim=0)
    emb = torch.cat(emb, dim=0)

    metadata = [d for d in data]
    return (emb, metadata)

def get_dataloader(dataset_folder, batch_size: int = 50, shuffle: bool = True):
    dataset = MusicDataset(dataset_folder)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate)
    return dataloader


def get_dataloaders(dataset_dir, sr, channels, min_duration, max_duration, sample_duration, 
                    aug_shift, batch_size: int = 50, shuffle: bool = True, split_ratio=0.8, device='cpu', same_folder=False):
    dataset = MusicDataset(dataset_dir=dataset_dir, sr=sr, channels=channels,
                           min_duration=min_duration, max_duration=max_duration, sample_duration=sample_duration,
                           aug_shift=aug_shift, device=device, same_folder=same_folder)

    # Split the dataset into train and validation
    train_size = int(split_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate)

    return train_dataloader, val_dataloader