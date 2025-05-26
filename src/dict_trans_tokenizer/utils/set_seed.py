import torch
from transformers import set_seed as set_transformers_seed


def set_seed(seed: int):
    torch.manual_seed(seed)
    set_transformers_seed(seed)
