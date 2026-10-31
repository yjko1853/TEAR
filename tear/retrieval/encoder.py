"""Sentence encoding (default all-mpnet-base-v2; any sentence-transformers model by name), MPS -> CPU fallback,
L2-normalised float32 outputs."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import MODEL
from ..utils.device import get_device
from ..utils.timing import timed


def load_encoder(device: str = "mps", threads: int | None = None, model_name: str | None = None):
    """threads caps torch's intra-op CPU threads (useful on a fanless machine); model_name defaults to MODEL."""
    if threads:
        import torch
        torch.set_num_threads(threads)
    from sentence_transformers import SentenceTransformer
    dev = get_device(device)
    return SentenceTransformer(model_name or MODEL, device=dev), dev


def l2_normalize(emb: np.ndarray) -> np.ndarray:
    emb = emb.astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
    return emb


def encode_resilient(model, texts, batch, device_holder):
    """encode() that halves the batch on MPS / out-of-memory errors and finally moves the model to CPU."""
    import torch
    while True:
        try:
            emb = model.encode(texts, batch_size=batch, show_progress_bar=False, convert_to_numpy=True,
                               normalize_embeddings=False)
            return emb, batch
        except RuntimeError as e:
            msg = str(e).lower()
            if "out of memory" in msg or "mps" in msg:
                if batch >= 32:
                    batch //= 2
                    print(f"  RuntimeError ({msg[:60]}...), retrying with batch={batch}", flush=True)
                    if device_holder[0] == "mps":
                        torch.mps.empty_cache()
                    continue
                if device_holder[0] != "cpu":
                    print("  falling back to CPU", flush=True)
                    model.to("cpu")
                    device_holder[0] = "cpu"
                    batch = 64
                    continue
            raise


def encode_sentences_chunked(texts, out_path: Path, parts_dir: Path, dataset: str, tag: str = "fact",
                             batch: int = 256, chunk: int = 16384, device: str = "mps",
                             model_name: str | None = None) -> None:
    """Resumable chunked encoding: each chunk is saved under parts_dir; out_path is their concatenation."""
    parts_dir.mkdir(parents=True, exist_ok=True)
    n = len(texts)
    model, dev = load_encoder(device, model_name=model_name)
    holder = [dev]
    print(f"model on {dev}, max_seq_length={model.max_seq_length}", flush=True)
    with timed("S1_encode", dataset, f"{tag} n={n} device={dev} batch={batch}"):
        n_parts = (n + chunk - 1) // chunk
        for pi in range(n_parts):
            pp = parts_dir / f"part_{pi:04d}.npy"
            if pp.exists():
                continue
            lo, hi = pi * chunk, min(n, (pi + 1) * chunk)
            with timed("S1_encode_chunk", dataset, f"{tag} part {pi}/{n_parts}"):
                emb, batch = encode_resilient(model, texts[lo:hi], batch, holder)
            np.save(pp, l2_normalize(emb))
        parts = [np.load(parts_dir / f"part_{pi:04d}.npy") for pi in range(n_parts)]
        full = np.concatenate(parts, axis=0) if len(parts) > 1 else parts[0]
        assert full.shape[0] == n, full.shape
        np.save(out_path, full)
        print(f"saved {out_path} shape={full.shape}", flush=True)


def encode_questions(texts, out_path: Path, dataset: str, batch: int = 256, device: str = "mps",
                     threads: int | None = None, model_name: str | None = None) -> None:
    model, dev = load_encoder(device, threads, model_name)
    with timed("S2_encode_questions", dataset, f"n={len(texts)} device={dev}"):
        qemb = model.encode(texts, batch_size=batch, show_progress_bar=False, convert_to_numpy=True)
        np.save(out_path, l2_normalize(qemb))
