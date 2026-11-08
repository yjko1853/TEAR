"""Reranker registry. rerank(name, sims, F) returns a (Q, N) permutation of the candidate pool."""
from __future__ import annotations

import numpy as np

from .baselines import chrono_perm, hardwindow_scores, timer4_scores

RERANKERS = ["none", "timer4", "timer4_sym", "chrono", "hardwindow", "atc", "tr", "tear", "tear_window"]
# TR-promote variant (diagnostic, 2026-10-31): the group representative is lifted to the rank of its group's
# best-scoring member; every other candidate keeps its score, so the reader still sees the other occurrences.
PROMOTE = ["tr_promote", "tear_promote", "tear_window_promote"]
PROMOTE_EPS = 1e-6   # breaks the tie with the member whose score the representative takes (float32 sims ~0.3-0.9)


def perm_from_scores(scores):
    return np.argsort(-scores, axis=1, kind="stable")


def promote_scores(sc, tr_rep):
    """TR-promote: each representative's score becomes the maximum over its group (+ PROMOTE_EPS); others unchanged."""
    sc = sc.copy()
    for qi in np.nonzero((tr_rep >= 0).any(axis=1))[0]:
        rep = tr_rep[qi]; m = rep >= 0
        row = sc[qi]
        np.maximum.at(row, rep[m], row[m])
        row[np.unique(rep[m])] += PROMOTE_EPS
    return sc


def tear_scores(sims, F, name: str, lam1: float, lam2: float):
    """atc: x lambda1 on ATC-penalised candidates; tr: x lambda2 on non-representatives; tear = both;
    tear_window additionally moves candidates outside the before/after window behind the valid ones.
    *_promote: the TR step is replaced by promote_scores (representative lifted, nothing down-weighted)."""
    sc = sims.copy()
    if name in ("atc", "tear", "tear_window", "tear_promote", "tear_window_promote"):
        sc = sc * np.where(F["atc"], lam1, 1.0).astype(np.float32)
    if name in ("tr", "tear", "tear_window"):
        sc = sc * np.where(F["tr"], lam2, 1.0).astype(np.float32)
    if name in PROMOTE:
        sc = promote_scores(sc, F["tr_rep"])
    if name in ("tear_window", "tear_window_promote"):
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
    if name in ("atc", "tr", "tear", "tear_window") or name in PROMOTE:
        return perm_from_scores(tear_scores(sims, F, name, lam1, lam2))
    raise ValueError(name)
