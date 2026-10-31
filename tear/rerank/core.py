"""Reranker registry. rerank(name, sims, F) returns a (Q, N) permutation of the candidate pool."""
from __future__ import annotations

import numpy as np

from .baselines import chrono_perm, hardwindow_scores, timer4_scores

RERANKERS = ["none", "timer4", "timer4_sym", "chrono", "hardwindow", "atc", "tr", "tear", "tear_window"]


def perm_from_scores(scores):
    return np.argsort(-scores, axis=1, kind="stable")


def tear_scores(sims, F, name: str, lam1: float, lam2: float):
    """atc: x lambda1 on ATC-penalised candidates; tr: x lambda2 on non-representatives; tear = both;
    tear_window additionally moves candidates outside the before/after window behind the valid ones."""
    sc = sims.copy()
    if name in ("atc", "tear", "tear_window"):
        sc = sc * np.where(F["atc"], lam1, 1.0).astype(np.float32)
    if name in ("tr", "tear", "tear_window"):
        sc = sc * np.where(F["tr"], lam2, 1.0).astype(np.float32)
    if name == "tear_window":
        sc[~F["window"]] -= 10.0
    return sc


def rerank(name: str, sims, F, lam1: float = 0.8, lam2: float = 0.8, mu: float = 0.4, P: int = 100):
    Q, N = sims.shape
    if name == "none":
        return np.tile(np.arange(N), (Q, 1))
    if name in ("timer4", "timer4_sym"):
        stores = [F["phi_before"]] if name == "timer4" else [F["phi_before"], F["phi_after"]]
        return perm_from_scores(timer4_scores(sims, stores, mu))
    if name.startswith("chrono"):
        return chrono_perm(F["chrono_dir"], F["cand_date"], Q, N, P)
    if name == "hardwindow":
        return perm_from_scores(hardwindow_scores(sims, F["window"]))
    if name in ("atc", "tr", "tear", "tear_window"):
        return perm_from_scores(tear_scores(sims, F, name, lam1, lam2))
    raise ValueError(name)
