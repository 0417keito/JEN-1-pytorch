# Heavily influenced by https://github.com/facebookresearch/audiocraft/blob/main/audiocraft/modules/conditioners.py

import logging
import typing as tp
import warnings

import torch
import torch.nn as nn

# from utils.module import NumberEmbedder


class Conditioner(nn.Module):
    def __init__(
            self,
            dim: int,
            output_dim: int,
            cond_len: int,
            project_out: bool = False,
    ):
        super().__init__()

        self.dim = dim
        self.output_dim = output_dim
        self.cond_len = cond_len
        self.proj_out = nn.Linear(dim, output_dim) if (dim != output_dim or project_out) else nn.Identity()

    def forward(self, x: tp.Any) -> tp.Any:
        raise NotImplementedError()


class T5Conditioner(Conditioner):
    T5_MODELS = ["t5-small", "t5-base", "t5-large", "t5-3b", "t5-11b",
                 "google/flan-t5-small", "google/flan-t5-base", "google/flan-t5-large",
                 "google/flan-t5-xl", "google/flan-t5-xxl"]

    T5_MODEL_DIMS = {
        "t5-small": 512,
        "t5-base": 768,
        "t5-large": 1024,
        "t5-3b": 1024,
        "t5-11b": 1024,
        "google/flan-t5-small": 512,
        "google/flan-t5-base": 768,
        "google/flan-t5-large": 1024,
        "google/flan-t5-3b": 1024,
        "google/flan-t5-11b": 1024,
    }

    def __init__(
            self,
            output_dim: int,
            t5_model_name: str = "t5-base",
            max_length: str = 128,
            enable_grad: bool = False,
            project_out: bool = False,
    ):
        assert t5_model_name in self.T5_MODELS, f"Unknown T5 model name: {t5_model_name}"
        super().__init__(self.T5_MODEL_DIMS[t5_model_name], output_dim, max_length, project_out=project_out)

        from transformers import T5EncoderModel, AutoTokenizer

        self.max_length = max_length
        self.enable_grad = enable_grad

        # Suppress logging from transformers
        previous_level = logging.root.manager.disable
        logging.disable(logging.ERROR)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                # self.tokenizer = T5Tokenizer.from_pretrained(t5_model_name, model_max_length = max_length)
                # model = T5EncoderModel.from_pretrained(t5_model_name, max_length=max_length).train(enable_grad).requires_grad_(enable_grad)
                self.tokenizer = AutoTokenizer.from_pretrained(t5_model_name)
                model = T5EncoderModel.from_pretrained(t5_model_name).train(enable_grad).requires_grad_(enable_grad)
            finally:
                logging.disable(previous_level)

        if self.enable_grad:
            self.model = model
        else:
            self.__dict__["model"] = model

    def forward(self, texts: tp.List[str], device: tp.Union[torch.device, str]) -> tp.Tuple[torch.Tensor, torch.Tensor]:

        self.model.to(device)
        self.proj_out.to(device)

        encoded = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device).to(torch.bool)

        self.model.eval()

        with torch.cuda.amp.autocast(enabled=False) and torch.set_grad_enabled(self.enable_grad):
            embeddings = self.model(
                input_ids=input_ids, attention_mask=attention_mask
            )["last_hidden_state"]

        embeddings = self.proj_out(embeddings)

        embeddings = embeddings * attention_mask.unsqueeze(-1).float()

        return embeddings, attention_mask


class IntConditioner(Conditioner):
    def __init__(self,
                 output_dim: int,
                 min_val: int = 0,
                 max_val: int = 512
                 ):
        super().__init__(output_dim, output_dim, 1)

        self.min_val = min_val
        self.max_val = max_val
        self.int_embedder = nn.Embedding(max_val - min_val + 1, output_dim).requires_grad_(True)

    def forward(self, ints: tp.List[int], device=None) -> tp.Any:
        ints = torch.tensor(ints).to(device)
        ints = ints.clamp(self.min_val, self.max_val)

        int_embeds = self.int_embedder(ints).unsqueeze(1)

        return [int_embeds, torch.ones(int_embeds.shape[0], 1).to(device)]


class NumberConditioner(Conditioner):
    '''
        Conditioner that takes a list of floats, normalizes them for a given range, and returns a list of embeddings
    '''

    def __init__(self,
                 output_dim: int,
                 min_val: float = 0,
                 max_val: float = 1
                 ):
        super().__init__(output_dim, output_dim, 1)

        self.min_val = min_val
        self.max_val = max_val
        from utils.module import NumberEmbedder
        self.embedder = NumberEmbedder(features=output_dim)

    def forward(self, floats: tp.List[float], device=None) -> tp.Any:
        # Cast the inputs to floats
        floats = [float(x) for x in floats]

        floats = torch.tensor(floats).to(device)

        floats = floats.clamp(self.min_val, self.max_val)

        normalized_floats = (floats - self.min_val) / (self.max_val - self.min_val)

        float_embeds = self.embedder(normalized_floats).unsqueeze(1)

        return [float_embeds, torch.ones(float_embeds.shape[0], 1).to(device)]


class MultiConditioner(nn.Module):
    """
    A module that applies multiple conditioners to an input dictionary based on the keys

    Args:
        conditioners: a dictionary of conditioners with keys corresponding to the keys of the conditioning input dictionary (e.g. "prompt")
        default_keys: a dictionary of default keys to use if the key is not in the input dictionary (e.g. {"prompt_t5": "prompt"})
    """

    def __init__(self, conditioners: tp.Dict[str, Conditioner], default_keys: tp.Dict[str, str] = {}):
        super().__init__()

        self.conditioners = nn.ModuleDict(conditioners)
        self.default_keys = default_keys

    def forward(self, batch_metadata: tp.List[tp.Dict[str, tp.Any]], device: tp.Union[torch.device, str]) -> tp.Dict[
        str, tp.Any]:
        output = {}

        for key, conditioner in self.conditioners.items():
            condition_key = key

            conditioner_inputs = []

            for x in batch_metadata:

                if condition_key not in x:
                    if condition_key in self.default_keys:
                        condition_key = self.default_keys[condition_key]
                    else:
                        raise ValueError(f"Conditioner key {condition_key} not found in batch metadata")

                # Unwrap the condition info if it's a single-element list or tuple, this is to support collation functions that wrap everything in a list
                if isinstance(x[condition_key], list) or isinstance(x[condition_key], tuple) and len(
                        x[condition_key]) == 1:
                    conditioner_inputs.append(x[condition_key][0])
                else:
                    conditioner_inputs.append(x[condition_key])

            output[key] = conditioner(conditioner_inputs, device)

        return output
