import torch
import torch.nn as nn

class Normalizer(nn.Module): 
    def __init__(self, dim):
        super().__init__()
    
    def forward(self, x): ...