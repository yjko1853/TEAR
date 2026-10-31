"""Small generic helpers: JSON I/O, step timing, torch device selection."""
from .io import load_json, save_json
from .timing import timed

__all__ = ["load_json", "save_json", "timed"]
