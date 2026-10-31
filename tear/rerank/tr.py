"""Temporal reranking (TR): temporal-constraint validity and one representative fact per group.

Ported from the original TEAR code (retrieval_temporal.group_facts_hits / select_representative_fact /
apply_temporal_penalty). For a first/last question the candidates are grouped by the slots that must stay fixed
(answer slot subject -> (r, o); object -> (s, r); timestamp -> (s, r, o)); among the candidates that satisfy the
before/after constraint, each group keeps its earliest (first) or latest (last) fact; every other candidate is a
non-representative and gets score x lambda2.
"""
from __future__ import annotations

import numpy as np

from ..temporal import NEG_INF_DAYS, POS_INF_DAYS, date_to_days, extract_from_to_dates


def before_after_validity(tdays, start_c, end_c, date_c, interval: bool):
    """Mask of candidates satisfying every (before|after, day) constraint; intervals use end < d / start > d."""
    valid = np.ones(len(date_c), bool)
    for typ, td in tdays:
        if typ == "before":
            valid &= (end_c if interval else date_c) < td
        else:
            valid &= (start_c if interval else date_c) > td
    return valid


def period_validity(bounds, date_c, start_c=None, end_c=None, interval: bool = False):
    """Mask of candidates inside an inclusive [lo, hi] day range (same-period constraint); intervals must overlap it."""
    lo, hi = bounds
    if interval:
        return (start_c <= hi) & (end_c >= lo)
    return (date_c >= lo) & (date_c <= hi)


def from_to_window(text, start_c, end_c):
    """TimelineKGQA 'from X to Y': facts whose interval lies inside [X, Y]; None if the question has no such span."""
    ft = extract_from_to_dates(text)
    if not ft:
        return None
    lo = NEG_INF_DAYS if ft[0] == "beginning of time" else date_to_days(ft[0])
    hi = POS_INF_DAYS if ft[1] == "end of time" else date_to_days(ft[1])
    if lo is None or hi is None:
        return None
    return (start_c >= lo) & (end_c <= hi)


def group_keys(slot, s_c, r_c, o_c, E: int, R: int):
    """Integer group key per candidate; the answer slot is the free position."""
    if slot == "subject":
        return r_c.astype(np.int64) * E + o_c
    if slot == "object":
        return s_c.astype(np.int64) * R + r_c
    return (s_c.astype(np.int64) * R + r_c) * E + o_c   # timestamp (and the original's fallback for relation)


def group_representatives(keys, valid, cand_date, cues):
    """Indices of the representative candidate of each group (earliest for 'first', latest for 'last')."""
    idx = np.nonzero(valid)[0]
    reps = np.array([], np.int64)
    if len(idx) and ("first" in cues or "last" in cues):
        k_ = keys[idx]; dt = cand_date[idx]
        order = np.lexsort((idx, dt if "first" in cues else -dt, k_))
        ks = k_[order]
        firsts = np.r_[True, ks[1:] != ks[:-1]]
        reps = idx[order[firsts]]
    return reps


def non_representative_mask(n: int, reps):
    """True = not a representative (gets lambda2)."""
    return ~np.isin(np.arange(n), reps)
