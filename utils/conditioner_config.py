import torch
from dataclasses import dataclass, field

'''
The id of each Config refers to the key in the dictionary of metadata used. 
The value corresponding to that key is used for conditioning.
example metadata: {"prompt": "a beautiful song", "seconds_start": 22, "seconds_total": 193}
'''

@dataclass
class T5Config:
    id = 'prompt'
    t5_model_name = 'google/flan-t5-large'
    max_length = 128
    project_out = True

@dataclass
class IntConfig:
    id = 'seconds_start'
    min_val = 0
    max_val = 512

@dataclass
class NumberConfig: 
    id = 'seconds_total'
    min_val = 0
    max_val = 512

@dataclass
class ConditionerConfig:
    cond_dim = 1024
    default_keys = {}
    #conditioning type you want but The same, id things cannot be used.
    conditioning_type = ['t5', 'int', 'number']  
    t5_config = T5Config
    int_config = IntConfig
    number_config = NumberConfig