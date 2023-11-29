import torch
import torchaudio
from typing import Any
from einops import rearrange, pack, unpack
from encodec import EncodecModel
from encodec.utils import convert_audio

class AudioTokenizer:
    """EnCodec audio."""

    def __init__(
        self,
        device: Any = None,
    ) -> None:
        # Instantiate a pretrained EnCodec model
        model = EncodecModel.encodec_model_48khz()
        model.set_target_bandwidth(6.0)
        remove_encodec_weight_norm(model)

        if not device:
            device = torch.device("cpu")
            if torch.cuda.is_available():
                device = torch.device("cuda:0")

        self._device = device

        self.codec = model.to(device)
        self.sample_rate = model.sample_rate
        self.channels = model.channels

    @property
    def device(self):
        return self._device

    def encode(self, wav: torch.Tensor, return_encoded=True) -> torch.Tensor:
        print("wav.shape:", wav.shape)
        wav, ps = pack([wav], '* n')
        print("wav.shape:", wav.shape)
        #wav = wav.unsqueeze(0)
        wav = rearrange(wav, f'b t -> b {self.codec.channels} t')

        encoded_frames = self.codec.encode(wav.to(self.device))
        codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1)
        codes = rearrange(codes, 'b q n -> b n q')

        emb = None
        if return_encoded:
            emb = self.get_emb_from_indices(codes)
            emb, = unpack(emb, ps, '* n c')

        codes, = unpack(codes, ps, ('* n q'))

        return emb, codes, None

    def get_emb_from_indices(self, indices):
        codes = rearrange(indices, 'b t q -> q b t')
        emb = self.codec.quantizer.decode(codes)
        return rearrange(emb, 'b n c -> b c n')

    def decode(self, frames: torch.Tensor) -> torch.Tensor:
        return self.codec.decode(frames)

def remove_encodec_weight_norm(model):
    from encodec.modules import SConv1d
    from encodec.modules.seanet import SConvTranspose1d, SEANetResnetBlock
    from torch.nn.utils import remove_weight_norm

    encoder = model.encoder.model
    for key in encoder._modules:
        if isinstance(encoder._modules[key], SEANetResnetBlock):
            remove_weight_norm(encoder._modules[key].shortcut.conv.conv)
            block_modules = encoder._modules[key].block._modules
            for skey in block_modules:
                if isinstance(block_modules[skey], SConv1d):
                    remove_weight_norm(block_modules[skey].conv.conv)
        elif isinstance(encoder._modules[key], SConv1d):
            remove_weight_norm(encoder._modules[key].conv.conv)

    decoder = model.decoder.model
    for key in decoder._modules:
        if isinstance(decoder._modules[key], SEANetResnetBlock):
            remove_weight_norm(decoder._modules[key].shortcut.conv.conv)
            block_modules = decoder._modules[key].block._modules
            for skey in block_modules:
                if isinstance(block_modules[skey], SConv1d):
                    remove_weight_norm(block_modules[skey].conv.conv)
        elif isinstance(decoder._modules[key], SConvTranspose1d):
            remove_weight_norm(decoder._modules[key].convtr.convtr)
        elif isinstance(decoder._modules[key], SConv1d):
            remove_weight_norm(decoder._modules[key].conv.conv)

def tokenize_audio(tokenizer: AudioTokenizer, audio):
    # Load and pre-process the audio waveform
    if isinstance(audio, str):
        wav, sr = torchaudio.load(audio)
    else:
        wav, sr = audio
    wav = convert_audio(wav, sr, tokenizer.sample_rate, tokenizer.channels)
    print("wav.shape:", wav.shape)
    #wav = wav.unsqueeze(0)
    print("wav.shape:", wav.shape)

    # Extract discrete codes from EnCodec
    with torch.no_grad():
        emb, codes, _ = tokenizer.encode(wav)
    return emb, codes
def preprocess_audio(audio_path, model, duration:int):
    wav, sr = torchaudio.load(audio_path)
    wav = torchaudio.functional.resample(wav, sr, model.sample_rate)
    wav = wav.mean(dim=0, keepdim=True)
    
    target_sample_length = int(model.sample_rate * duration)
    current_sample_length = wav.shape[1]
    if current_sample_length > target_sample_length:
        end_sample = target_sample_length
