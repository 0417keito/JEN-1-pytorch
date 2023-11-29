import numpy as np

from dataset.text_processor import TextProcessor, get_text_token_collater

text_collater = get_text_token_collater()

language_dict = {
    'en': 0,
    'zh': 1,
    'ja': 2,
}


def get_relevant_lyric_tokens(full_tokens, n_tokens, total_length, offset, duration):
    if len(full_tokens) < n_tokens:
        tokens = [0] * (n_tokens - len(full_tokens)) + full_tokens
        indices = [-1] * (n_tokens - len(full_tokens)) + list(range(0, len(full_tokens)))
    else:
        assert 0 <= offset < total_length
        midpoint = int(len(full_tokens) * (offset + duration / 2.0) / total_length)
        midpoint = min(max(midpoint, n_tokens // 2), len(full_tokens) - n_tokens // 2)
        tokens = full_tokens[midpoint - n_tokens // 2:midpoint + n_tokens // 2]
        indices = list(range(midpoint - n_tokens // 2, midpoint + n_tokens // 2))
    assert len(tokens) == n_tokens, f"Expected length {n_tokens}, got {len(tokens)}"
    assert len(indices) == n_tokens, f"Expected length {n_tokens}, got {len(indices)}"
    assert tokens == [full_tokens[index] if index != -1 else 0 for index in indices]
    return tokens, indices


class Labeller:
    def __init__(self, n_tokens, sample_length):
        self.text_processor = TextProcessor()
        self.n_tokens = n_tokens
        self.sample_length = sample_length

    def get_label(self, lyrics, genres, info, lang, metadata, tags, total_length, offset):
        # Code designed to use the music4all dataset
        lyrics, _ = self.text_processor.clean(text=f"{lyrics}".strip())
        artist, song_name = info['artist'], info['song']
        (popularity,
         release_danceability,
         energy,
         key_mode,
         valence_tempo,
         ) = (metadata['popularity'],
              metadata['release_danceablity'],
              metadata['energy'],
              metadata['key mode'],
              metadata['valence tempo'],)
        full_tokens = self.text_processor.tokenise(lyrics)
        # tokens, _ = get_relevant_lyric_tokens(full_tokens, self.n_tokens, total_length, offset, self.sample_length)

        cptpho_tokens, enroll_x_lens = text_collater([full_tokens])
        cptpho_tokens = cptpho_tokens.squeeze(0)
        lyrics_token_lens = enroll_x_lens[0]
        prompts = []
        prompts['artist_name'] = artist
        prompts['song_name'] = song_name
        prompts['genre'] = genres
        prompts['language'] = lang
        prompts['popularity'] = popularity
        prompts['release_daceability'] = release_danceability
        prompts['energy'] = energy
        prompts['key_mode'] = key_mode
        prompts['valence_tempo'] = valence_tempo
        prompts['tags'] = tags

        return {
            'text': lyrics,
            'audio': None,
            'audio_lens': total_length,
            'audio_features': None,
            'audio_features_lens': None,
            'text_tokens': np.array(cptpho_tokens),
            'text_tokens_lens': lyrics_token_lens,
            'prompts': prompts
        }
