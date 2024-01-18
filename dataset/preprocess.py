import os
import torch
import torchaudio

def get_duration_sec(file): 
        wav, sr = torchaudio.load(file)
        duration_sec = wav.shape[-1] / sr
        return duration_sec

def filter(audio_files, durations):
        keep = []
        audio_files = []
        for i in range(len(audio_files)):
            filepath = audio_files[i]
            if durations[i] < min_duration:
                continue
            if durations[i] >= max_duration:
                continue
            keep.append(i)
            audio_files.append(filepath)
        durations = [durations[i] for i in keep] # in (s)
        duration_tensor = torch.tensor(durations)
        cumsum = torch.cumsum(duration_tensor, dim=0) # in (s)
        return cumsum , audio_files
    
def init_dataset(audio_files_dir, min_duration, max_duration, audio_file_txt_path,):
    audio_files = os.listdir(audio_files_dir)
    audio_files = [f'{audio_files_dir}/{file}' for file in audio_files if file.endswith('.wav') or file.endswith('.mp3')]
    durations = [get_duration_sec(file) for file in audio_files]
    cumsum, filterd_audio_files = filter(audio_files=audio_files, durations=durations, 
                    min_duration=min_duration, max_duration=max_duration)
    with open(audio_file_txt_path, 'w') as file:
        for item in filterd_audio_files:
            file.write(f"{item}\n")
    return cumsum, durations


if __name__ == '__main__':
    cumsum_path = ...
    durations_path = ...
    audio_file_txt_path = ...
    min_duration, max_duration = ... # please type min_duration, max_duration
    audio_files_dir = 'please type your dir '
    cumsum, durations = init_dataset(audio_files_dir=audio_files_dir, min_duration=min_duration,
                                     max_duration=max_duration, audio_file_txt_path=audio_file_txt_path)
    
    torch.save(cumsum, cumsum_path)
    torch.save(durations, durations_path)