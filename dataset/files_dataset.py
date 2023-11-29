import logging
import librosa
import math
import os
import json
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from .utils import get_duration_sec, load_audio
from .audio_processor import AudioTokenizer, tokenize_audio


class FilesAudioDataset(Dataset):
    def __init__(self, sr, channels, min_duration, max_duration, sample_length, cache_dir,
                 aug_shift, device, dataset_dir):
        super().__init__()
        self.sr = sr
        self.channels = channels
        self.min_duration = min_duration or math.ceil(sample_length / sr)
        self.max_duration = max_duration or math.inf
        self.sample_length = sample_length
        assert sample_length / sr < self.min_duration, f'Sample length {sample_length} per sr {sr} ({sample_length / sr:.2f}) should be shorter than min duration {self.min_duration}'
        self.aug_shift = aug_shift
        self.device = device
        self.audio_files_dir = f'{dataset_dir}/audios'
        self.metadatas_dir = f'{dataset_dir}/metadata'
             
        self.init_dataset(cache_dir)

    def filter(self, files, durations):
        # Remove files too short or too long[]
        keep = []
        for i in range(len(files)):
            filepath = files[i]
            if durations[i] / self.sr < self.min_duration:
                continue
            if durations[i] / self.sr >= self.max_duration:
                continue
            keep.append(i)
        logging.info(f'self.sr={self.sr}, min: {self.min_duration}, max: {self.max_duration}')
        logging.info(f"Keeping {len(keep)} of {len(files)} files")
        self.files = [files[i] for i in keep]
        self.durations = [int(durations[i]) for i in keep] #サンプル長
        self.cumsum = np.cumsum(self.durations) #サンプル長の累積和

    def init_dataset(self, cache_dir):
        # Load list of files and starts/durations
        files = librosa.util.find_files(directory=f'{self.audio_files_dir}', ext=['mp3', 'opus', 'm4a', 'aac', 'wav'])
        logging.info(f"Found {len(files)} files. Getting durations")
        cache = cache_dir
        durations = np.array([get_duration_sec(file, cache=cache) * self.sr for file in files])  # Could be approximate　duration in sample_length
        self.filter(files, durations)

    def get_index_offset(self, item):
        # For a given dataset item and shift, return song index and offset within song
        half_interval = self.sample_length//2
        shift = np.random.randint(-half_interval, half_interval) if self.aug_shift else 0
        offset = item * self.sample_length + shift # Note we centred shifts, so adding now
        midpoint = offset + half_interval
        assert 0 <= midpoint < self.cumsum[-1], f'Midpoint {midpoint} of item beyond total length {self.cumsum[-1]}'
        index = np.searchsorted(self.cumsum, midpoint)  # index <-> midpoint of interval lies in this song
        start, end = self.cumsum[index - 1] if index > 0 else 0.0, self.cumsum[index] # start and end of current song
        assert start <= midpoint <= end, f"Midpoint {midpoint} not inside interval [{start}, {end}] for index {index}"
        if offset > end - self.sample_length: # Going over song
            offset = max(start, offset - half_interval)  # Now should fit
        elif offset < start: # Going under song
            offset = min(end - self.sample_length, offset + half_interval)  # Now should fit
        assert start <= offset <= end - self.sample_length, f"Offset {offset} not in [{start}, {end - self.sample_length}]. End: {end}, SL: {self.sample_length}, Index: {index}"
        offset = offset - start
        return index, offset
    
    def get_song_chunk(self, index, offset, test=False):
        filepath, total_length = self.files[index], self.durations[index]
        song_id = os.path.splitext(os.path.basename(filepath))[0]
        data, sr = load_audio(filepath, sr=self.sr, offset=offset, duration=self.sample_length)
        assert data.shape == (self.channels, self.sample_length), f'Expected {(self.channels, self.sample_length)}, got {data.shape}'
        
        audio_emcoder = AudioTokenizer(self.device)
        if data.size(0) == 2:
            data = data.mean(0, keepdim=True)
        embs, _ = tokenize_audio(audio_emcoder, (data, sr))
        
        if os.path.exists(f'{self.metadatas_dir}/{song_id}.json'):
                with open(f'{self.metadatas_dir}/{song_id}.json', 'r') as file:
                    metadata_for_t2m = json.load(file)
        
        return embs, metadata_for_t2m
    
    def get_dur(self, item):
        return self.durations[item]

    def get_item(self, item, test=False):
        index, offset = self.get_index_offset(item)
        return self.get_song_chunk(index, offset, test)

    def __len__(self):
        return int(np.floor(self.cumsum[-1] / self.sample_length))

    def __getitem__(self, item):
        return self.get_item(item)
