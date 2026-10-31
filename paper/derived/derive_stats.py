"""Derived statistics for the manuscript (read-only on cache/ and results/; writes CSVs next to this script).

Run from the repository root:  .venv/bin/python -I paper/derived/derive_stats.py
  group_structure.csv   share of KG facts whose (s, r, o) recurs with another timestamp (MultiTQ, ICEWS-Actor, Cron)
  top15_redundancy.csv  share of top-15 candidates (reranker 'none') that repeat an (s, r, o) already ranked higher
  rule_firing_shares.csv  question-weighted share of questions with a cue / parsed date / window / TR change / ATC change
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
C = Path("cache")
rows = []
for ds in ("multitq", "icews", "cron"):
    f = np.load(C / ds / "facts.npz")
    r = f["r_id"] if "r_id" in f else f["p_id"]
    s, o = f["s_id"].astype(np.int64), f["o_id"].astype(np.int64)
    R = int(r.max()) + 1; E = int(max(s.max(), o.max())) + 1
    key = (s * R + r) * E + o
    _, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    rows.append({"dataset": ds, "n_facts": len(key), "n_relations": R, "n_sro_groups": len(cnt),
                 "share_facts_in_repeated_sro": float((cnt[inv] >= 2).mean()),
                 "mean_facts_per_sro": float(len(key) / len(cnt)), "max_facts_per_sro": int(cnt.max())})
pd.DataFrame(rows).to_csv(OUT / "group_structure.csv", index=False)
print(pd.DataFrame(rows).to_string(index=False))

red = []
for ds, tag in (("multitq", "rule_exact_anydate"), ("icews", "oracle"), ("cron", "oracle")):
    f = np.load(C / ds / "facts.npz")
    r = f["r_id"] if "r_id" in f else f["p_id"]
    s, o = f["s_id"].astype(np.int64), f["o_id"].astype(np.int64)
    R = int(r.max()) + 1; E = int(max(s.max(), o.max())) + 1
    key = (s * R + r) * E + o
    for rk in ("none", "tear"):
        ids = np.load(C / ds / "rerank" / f"{rk}_{tag}_top30_ids.npy")[:, :15]
        k = key[ids]
        dup = np.zeros_like(k, bool)
        for j in range(1, 15):
            dup[:, j] = (k[:, :j] == k[:, j:j + 1]).any(1)
        red.append({"dataset": ds, "reranker": rk, "tag": tag, "n_questions": len(k),
                    "share_top15_repeating_sro": float(dup.mean()),
                    "share_questions_with_repeat": float(dup.any(1).mean())})
pd.DataFrame(red).to_csv(OUT / "top15_redundancy.csv", index=False)
print(pd.DataFrame(red).to_string(index=False))

# share of questions on which each rule changes something (weighted over the constraint_recognition rows)
fire = []
for ds, tag in (("multitq", "rule_exact_anydate"), ("icews", "oracle"), ("cron", "oracle")):
    cr = pd.read_csv(Path("results") / f"{ds}_constraint_recognition_{tag}.csv")
    n = cr["n"].sum()
    fire.append({"dataset": ds, "tag": tag, "n_questions": int(n),
                 **{f"share_{c}": float((cr["n"] * cr[c]).sum() / n) for c in
                    ("cue_first", "cue_last", "target_date", "window", "tr_fired", "atc_penalised_any")}})
pd.DataFrame(fire).to_csv(OUT / "rule_firing_shares.csv", index=False)
print(pd.DataFrame(fire).to_string(index=False))
