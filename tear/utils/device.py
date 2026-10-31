from __future__ import annotations


def get_device(prefer: str = "mps") -> str:
    """'mps' / 'cuda' if requested and available, otherwise 'cpu'. Imports torch lazily."""
    import torch
    if prefer == "mps" and torch.backends.mps.is_available():
        return "mps"
    if prefer == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"
