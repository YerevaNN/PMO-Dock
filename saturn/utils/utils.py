import logging
import os
import random
import torch
import numpy as np


def _cuda_device_index(device: str):
    """Return CUDA device index from device string (e.g. 'cuda:0' -> 0)."""
    if isinstance(device, torch.device):
        return device.index if device.type == "cuda" else None
    if isinstance(device, str) and "cuda" in device:
        return int(device.split(":")[-1]) if ":" in device else 0
    return None


def get_gpu_memory_stats(device: str):
    """
    Return current GPU memory stats for the given device.

    :param device: e.g. "cuda:0" or "cuda"
    :return: dict with keys allocated_gb, reserved_gb, total_gb, free_gb (total - reserved),
             or None if not a CUDA device.
    """
    idx = _cuda_device_index(device)
    if idx is None or not torch.cuda.is_available():
        return None
    torch.cuda.synchronize(idx)
    allocated = torch.cuda.memory_allocated(idx)
    reserved = torch.cuda.memory_reserved(idx)
    total = torch.cuda.get_device_properties(idx).total_memory
    return {
        "allocated_gb": allocated / (1024**3),
        "reserved_gb": reserved / (1024**3),
        "total_gb": total / (1024**3),
        "free_gb": (total - reserved) / (1024**3),
    }


def format_gpu_memory(device: str) -> str:
    """
    Format current GPU memory as a short string: allocated / total, free.
    Returns empty string if not CUDA.
    """
    stats = get_gpu_memory_stats(device)
    if stats is None:
        return ""
    return "mem allocated=%.2fGiB total=%.2fGiB free=%.2fGiB" % (
        stats["allocated_gb"],
        stats["total_gb"],
        stats["free_gb"],
    )


def set_seed_everywhere(seed: int, device: str):
    """Set the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setup_logging(logging_path: str, level=logging.INFO):
    """Add file handler to root logger. Creates parent dir of logging_path. Does not add console if one exists."""
    root = logging.getLogger("")
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    os.makedirs(os.path.dirname(logging_path) or ".", exist_ok=True)
    fh = logging.FileHandler(logging_path, mode="a")
    fh.setLevel(level)
    fh.setFormatter(fmt)
    root.addHandler(fh)
    if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        ch = logging.StreamHandler()
        ch.setLevel(level)
        ch.setFormatter(fmt)
        root.addHandler(ch)

def to_tensor(array: np.array, device: str) -> torch.Tensor:
    """Convert np.array to torch.Tensor."""
    return torch.tensor(array, device=device)

def generate_causal_mask(size: int, device: str) -> torch.Tensor:
    """
    Generates the Causal Mask for input to Self-Attention. 
    Masked postions = float("-inf")
    Unmasked positions = float(0.0)

    :param size: Size of the (square) mask
    :return: A (size, size) mask
    """
    mask = (torch.triu(torch.ones(size, size, device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0))
    return mask
