import random

import torch
import numpy as np

from genetic_chemalactica.utils.mol import is_valid_smiles, compute_fingerprint
from genetic_chemalactica.utils.file import load_csv, load_yamls


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def select_dtype(dtype: str):
    return {
        "bf16": torch.bfloat16,
        "fp32": torch.float32
    }[dtype]


__all__ = [
    "is_valid_smiles",
    "compute_fingerprint",
    "load_yamls",
    "load_csv",
    "RandomParser",
    "parse_rand_string",
    "parse_rand_strings"
]