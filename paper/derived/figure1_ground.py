"""Figure 1 flags on the original wording with anchor grounding (question 3000150, dataset multitq_orig).
Run from the repository root:  .venv/bin/python -I paper/derived/figure1_ground.py > paper/derived/figure1_flags_ground.txt
Flags per candidate: penalised entity in the answer slot (topic Kuwait or anchor Council of Representatives), valid
(before the grounded date 2009-07-03), representative of its (r, o) group among valid candidates (latest)."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root (tear/)
from tear.temporal import date_to_days  # noqa: E402

d = "cache/multitq_orig"; k = "3000150"; TAG = "rule_v2_ground_exact_anydate"
order = [str(x) for x in json.load(open(f"{d}/rerank/question_order.json"))["question_ids"]]; i = order.index(k)
ids = np.load(f"{d}/top1000_ids.npy", mmap_mode="r")[i]
f = np.load(f"{d}/facts.npz"); E = json.load(open(f"{d}/vocab.json"))["entities"]
sent = open(f"{d}/sentences.txt").read().split("\n")
td = date_to_days("2009-07-03"); pen = {"Kuwait", "Council of Representatives"}
s, r, o, t = f["s_id"][ids], f["r_id"][ids], f["o_id"][ids], f["t_days"][ids]
valid = t < td; best = {}
for j in range(len(ids)):
    if valid[j]:
        g = (int(r[j]), int(o[j]))
        if g not in best or t[j] > t[best[g]]:
            best[g] = j
rep = np.zeros(len(ids), bool); rep[list(best.values())] = True
pos = {int(x): j for j, x in enumerate(ids)}
for rk in ("none", "atc", "tr", "tear"):
    top = np.load(f"{d}/rerank/{rk}_{TAG}_top30_ids.npy", mmap_mode="r")[i, :3]
    print(rk)
    for x in top:
        j = pos[int(x)]
        print(f"  {sent[int(x)]:70s} penalised={E[s[j]] in pen} valid={bool(valid[j])} rep={bool(rep[j])}")
