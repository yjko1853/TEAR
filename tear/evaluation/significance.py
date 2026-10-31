"""Paired significance tests for per-question correctness vectors (same questions, two systems)."""
from __future__ import annotations

import math

import numpy as np


def mcnemar(a, b, exact: bool | None = None) -> dict:
    """McNemar test on paired booleans a (system A correct) and b (system B correct).

    exact=None chooses the exact binomial test when the number of discordant pairs is < 25, otherwise the
    chi-square test with continuity correction (df = 1).
    """
    a = np.asarray(a, bool); b = np.asarray(b, bool)
    n01 = int((~a & b).sum())   # A wrong, B right
    n10 = int((a & ~b).sum())   # A right, B wrong
    n = n01 + n10
    if n == 0:
        return {"n01": 0, "n10": 0, "statistic": 0.0, "p_value": 1.0, "method": "none"}
    if exact is None:
        exact = n < 25
    if exact:
        k = min(n01, n10)
        p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
        return {"n01": n01, "n10": n10, "statistic": float(k), "p_value": min(1.0, 2 * p), "method": "exact"}
    stat = (abs(n01 - n10) - 1) ** 2 / n
    return {"n01": n01, "n10": n10, "statistic": stat, "p_value": math.erfc(math.sqrt(stat / 2)), "method": "chi2_cc"}


def paired_bootstrap(a, b, n_boot: int = 10000, seed: int = 0, alpha: float = 0.05) -> dict:
    """Bootstrap over questions of mean(b) - mean(a): point estimate, percentile CI, two-sided p-value.

    p_value = 0.0 means no bootstrap replicate crossed zero, i.e. p < 1 / n_boot.
    """
    a = np.asarray(a, float); b = np.asarray(b, float)
    d = b - a
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    chunk = max(1, 2_000_000 // max(1, len(d)))   # bounded memory for large question sets
    for lo in range(0, n_boot, chunk):
        hi = min(n_boot, lo + chunk)
        boots[lo:hi] = d[rng.integers(0, len(d), size=(hi - lo, len(d)))].mean(1)
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    p = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
    return {"diff": float(d.mean()), "ci_low": float(lo), "ci_high": float(hi), "p_value": float(min(1.0, p)),
            "n": int(len(d)), "n_boot": n_boot}
