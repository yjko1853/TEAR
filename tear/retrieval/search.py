"""Exact inner-product top-N search over L2-normalised fact embeddings."""
from __future__ import annotations

import numpy as np

from ..utils.timing import timed


def faiss_topn(qemb: np.ndarray, femb: np.ndarray, N: int, dataset: str, qbatch: int = 2048, threads: int = 8):
    """FAISS IndexFlatIP (exact). Do not call from a process that has imported torch (macOS hang)."""
    import faiss
    faiss.omp_set_num_threads(threads)
    ids = np.zeros((len(qemb), N), np.int32)
    scores = np.zeros((len(qemb), N), np.float32)
    with timed("S2_faiss_build", dataset, f"IndexFlatIP n={femb.shape[0]}"):
        index = faiss.IndexFlatIP(femb.shape[1])
        index.add(np.ascontiguousarray(femb, dtype=np.float32))
    with timed("S2_faiss_search", dataset, f"top{N} queries={len(qemb)}"):
        for lo in range(0, len(qemb), qbatch):
            hi = min(len(qemb), lo + qbatch)
            D, I = index.search(np.ascontiguousarray(qemb[lo:hi]), N)
            ids[lo:hi] = I
            scores[lo:hi] = D
            print(f"  searched {hi}/{len(qemb)}", flush=True)
    return ids, scores


def numpy_topn(qemb: np.ndarray, femb: np.ndarray, N: int, dataset: str, qb: int = 256):
    """Chunked matmul + argpartition; the same exact result without faiss."""
    F = np.ascontiguousarray(femb, dtype=np.float32)
    ids = np.zeros((len(qemb), N), np.int32)
    scores = np.zeros((len(qemb), N), np.float32)
    with timed("S2_numpy_search", dataset, f"top{N} queries={len(qemb)}"):
        for lo in range(0, len(qemb), qb):
            hi = min(len(qemb), lo + qb)
            S = qemb[lo:hi] @ F.T
            part = np.argpartition(-S, N - 1, axis=1)[:, :N]
            ps = np.take_along_axis(S, part, axis=1)
            o = np.argsort(-ps, axis=1, kind="stable")
            ids[lo:hi] = np.take_along_axis(part, o, axis=1)
            scores[lo:hi] = np.take_along_axis(ps, o, axis=1)
            if (lo // qb) % 20 == 0:
                print(f"  searched {hi}/{len(qemb)}", flush=True)
    return ids, scores
