import numpy as np
import torch


def collate(batch):
    utt_id_s = [b['utt_id'] for b in batch]
    text_s = [b['text'] for b in batch]

    audio_s = [b['audio'] for b in batch]
    audio_lens_s = [b['audio_lens'] for b in batch]

    audio_features_lens_s = [b['audio_features_lens'] for b in batch]
    # create an empty tensor with maximum audio feature length
    audio_features_s = torch.zeros([len(batch), max(audio_features_lens_s), 8],
                                   dtype=torch.int64) - 1  # audio pad with -1

    text_tokens_lens_s = [b['text_tokens_lens'] for b in batch]
    # create an empty tensor with maximum text tokens length
    text_tokens_s = torch.zeros([len(batch), max(text_tokens_lens_s)], dtype=torch.int64) + 3  # [PAD] token id 3

    language_s = [b['language'] for b in batch]

    for i, b in enumerate(batch):
        audio_features = b['audio_features']
        audio_features_lens = b['audio_features_lens']
        audio_features_s[i, :audio_features_lens, :] = torch.LongTensor(audio_features)

        text_tokens = b['text_tokens']
        text_tokens_lens = b['text_tokens_lens']
        text_tokens_s[i, :text_tokens_lens] = torch.LongTensor(text_tokens)

    batch = {
        'utt_id': utt_id_s,
        'text': text_s,
        'audio': audio_s,
        'audio_lens': audio_lens_s,
        'audio_features': audio_features_s,
        'audio_features_lens': torch.LongTensor(np.array(audio_features_lens_s)),
        'text_tokens': text_tokens_s,
        'text_tokens_lens': torch.LongTensor(np.array(text_tokens_lens_s)),
        'languages': torch.LongTensor(np.array(language_s)),
    }
    return batch
