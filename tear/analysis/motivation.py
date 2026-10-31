"""Cosine similarity of fact pairs that share (s, r, o) and differ only in time, against two references
(pairs sharing only the subject, random pairs). Motivates temporal reranking: the encoder cannot separate them."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..utils.io import load_json


def summarize(name, sims):
    return {"pair_type": name, "n_pairs": len(sims), "mean": float(sims.mean()), "median": float(np.median(sims)),
            "std": float(sims.std()), "p10": float(np.percentile(sims, 10)), "p90": float(np.percentile(sims, 90)),
            "share_gt_0.8": float((sims > 0.8).mean()), "share_gt_0.9": float((sims > 0.9).mean()),
            "share_gt_0.95": float((sims > 0.95).mean())}


def same_sro_similarity(d, dataset: str, n_pairs: int = 20000, seed: int = 0):
    """Returns (summary DataFrame, histogram DataFrame). Sampling order is fixed by the seed."""
    f = np.load(d / "facts.npz")
    emb = np.load(d / "fact_emb.npy", mmap_mode="r")
    rng = np.random.default_rng(seed)
    if dataset == "multitq":
        s, r, o, t = f["s_id"], f["r_id"], f["o_id"], f["t_id"]
    else:
        s, r, o = f["s_id"], f["p_id"], f["o_id"]
        t = f["start_days"].astype(np.int64) * 10**6 + (f["end_days"].astype(np.int64) % 10**6)
    E = len(load_json(d / "vocab.json")["entities"]); R = int(r.max()) + 1
    key = (s.astype(np.int64) * R + r) * E + o
    order = np.argsort(key, kind="stable")
    ks = key[order]
    starts = np.r_[0, np.nonzero(ks[1:] != ks[:-1])[0] + 1]
    ends = np.r_[starts[1:], len(ks)]
    good = []   # groups with >= 2 distinct timestamps
    for a, b in zip(starts, ends):
        if b - a >= 2 and len(np.unique(t[order[a:b]])) >= 2:
            good.append((a, b))
    print(f"groups={len(starts)} groups_with_2+_distinct_times={len(good)} "
          f"facts_in_such_groups={sum(b-a for a,b in good)}")
    pairs = []
    for g in rng.integers(len(good), size=n_pairs):
        a, b = good[g]
        members = order[a:b]
        for _ in range(10):
            i, j = rng.choice(members, 2, replace=False)
            if t[i] != t[j]:
                pairs.append((i, j)); break
    pairs = np.array(pairs)

    def cos(p):
        A = np.asarray(emb[p[:, 0]]); B = np.asarray(emb[p[:, 1]])
        return (A * B).sum(1)
    same = cos(pairs)
    rnd = cos(rng.integers(len(s), size=(len(pairs), 2)))
    so = np.argsort(s, kind="stable"); ss = s[so]
    st = np.r_[0, np.nonzero(ss[1:] != ss[:-1])[0] + 1]; en = np.r_[st[1:], len(ss)]
    big = [(a, b) for a, b in zip(st, en) if b - a >= 2]
    sp = []
    for g in rng.integers(len(big), size=len(pairs)):
        a, b = big[g]; i, j = rng.choice(so[a:b], 2, replace=False); sp.append((i, j))
    subj = cos(np.array(sp))
    df = pd.DataFrame([summarize("same_(s,r,o)_different_t", same), summarize("same_subject_only", subj),
                       summarize("random_pairs", rnd)])
    bins = np.linspace(0, 1, 21)
    hist = pd.DataFrame({"bin_lo": bins[:-1], "bin_hi": bins[1:],
                         "same_sro": np.histogram(same, bins)[0], "same_subject": np.histogram(subj, bins)[0],
                         "random": np.histogram(rnd, bins)[0]})
    sizes = np.array([b - a for a, b in zip(starts, ends)])
    print("group size: mean %.2f, share of facts in groups with >=2 members %.3f, max %d" %
          (sizes.mean(), (sizes[sizes >= 2].sum() / sizes.sum()), sizes.max()))
    return df, hist
