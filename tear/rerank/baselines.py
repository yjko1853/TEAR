"""Baseline rerankers compared with TEAR on the same candidate pool.

  timer4      TimeR4 (Qian et al., EMNLP 2024) Eq. 9-10 for 'before <date>' questions:
              phi_t = 1 - |t_q - t| / max(t_q - t) if t_q - t > 0 else -100; final = mu*sim + (1-mu)*phi_t
  timer4_sym  the same rule also applied to 'after <date>' (extension, not in TimeR4)
  chrono      first -> top-P by similarity sorted by date ascending; last -> descending (PoK Rank / SAR style)
  hardwindow  candidates outside the before/after (or from-to) window are moved behind the valid ones
"""
from __future__ import annotations

import numpy as np


def timer4_phi(tdays, typ: str, cand_date):
    """phi_t for the first constraint of type typ ('before' | 'after'); None if the question has none."""
    td = next((b for a, b in tdays if a == typ), None)
    if td is None:
        return None
    diff = (td - cand_date) if typ == "before" else (cand_date - td)
    pos = diff > 0
    ph = np.full(len(cand_date), -100.0, np.float32)
    if pos.any():
        ph[pos] = 1.0 - np.abs(diff[pos]) / diff[pos].max()
    return ph


def chrono_direction(cues) -> int:
    """+1 first (ascending), -1 last (descending), 0 otherwise."""
    if "first" in cues:
        return 1
    if "last" in cues:
        return -1
    return 0


def timer4_scores(sims, phi_stores, mu: float):
    sc = sims.copy()
    for store in phi_stores:
        has = ~np.isnan(store[:, 0])
        sc[has] = mu * sims[has] + (1 - mu) * store[has]
    return sc


def chrono_perm(chrono_dir, cand_date, Q: int, N: int, P: int):
    perm = np.tile(np.arange(N), (Q, 1))
    for qi in np.nonzero(chrono_dir)[0]:
        d = cand_date[qi, :P]
        o = np.argsort(d if chrono_dir[qi] > 0 else -d, kind="stable")
        perm[qi, :P] = o
    return perm


def hardwindow_scores(sims, window):
    sc = sims.copy()
    sc[~window] -= 10.0   # behind every valid candidate; order among them stays by similarity
    return sc
