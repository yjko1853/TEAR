"""Reader statistics on the generated ICEWS18 questions (auxiliary experiment), with the scoring rules of
scripts/18_final_reader_report.py (reader, reader+date, direct = LLM-free top-1 fact, hybrid), and paired
tests with confidence intervals; must agree with results/qwen_icews18_final_summary.csv.

Run from the repository root:  .venv/bin/python -I paper/derived/derive_reader_icews18.py
Inputs : results/qwen_icews18/*.jsonl (Qwen2.5-7B-Instruct-4bit, prompt v3, k = 15, first 2,000 questions of the fixed
         shuffle), cache/icews18/{questions.json, slots_rule_v2.json, facts.npz, vocab.json, rerank/}
Outputs (paper/derived/):
  reader_icews18_summary.csv       per metric and setting: Hit@1 on the 2,000 questions, per question type
  reader_icews18_significance.csv  McNemar (continuity corrected) and paired bootstrap (10,000 draws, seed 0)
  The hybrid rule (first/last cue -> direct, otherwise reader+date) was fixed on 2026-10-17 on MultiTQ; every ICEWS18
  question is unseen, so it is evaluated on all 2,000 questions.
"""
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root (tear/)
from tear.reader.parse import parse_list  # noqa: E402
from tear.temporal import extract_temporal_cue  # noqa: E402
from tear.text import normalize  # noqa: E402

OUT = Path(__file__).resolve().parent
D = Path("cache/icews18")
PRED = Path("results/qwen_icews18")
TAG, MODEL = "rule_v2_ground_exact_anydate", "Qwen2.5-7B-Instruct-4bit"
SETTINGS = ["none", "timer4", "hardwindow", "tear", "tear_window"]

Q = {str(q["quid"]): q for q in json.load(open(D / "questions.json"))}
S = json.load(open(D / "slots_rule_v2.json"))
order = [str(x) for x in json.load(open(D / "rerank" / "question_order.json"))["question_ids"]]
pos = {k: i for i, k in enumerate(order)}
F = np.load(D / "facts.npz")
V = json.load(open(D / "vocab.json"))
E, T = V["entities"], V["timestamps"]


def gran(q):
    ql = q.lower()
    return 4 if re.search(r"\byear\b", ql) else (7 if re.search(r"\bmonth\b", ql) else 10)


def cut_date(p, q):
    m = re.search(r"(\d{4})(?:-(\d{2}))?(?:-(\d{2}))?", p)
    if not m:
        return p
    s = m.group(1) + ("-" + m.group(2) if m.group(2) else "") + ("-" + m.group(3) if m.group(3) else "")
    g = gran(q)
    return s[:g] if len(s) >= g else p


def ok(pred, q):
    return any(normalize(pred) == normalize(a) for a in q["answers"])


rows = []
for s in SETTINGS:
    R = [json.loads(line) for line in open(PRED / f"{s}_{TAG}__{MODEL}__k15__p-v3.jsonl")]
    top = np.load(D / "rerank" / f"{s}_{TAG}_top30_ids.npy", mmap_mode="r")[:, 0]
    for r in R:
        k = str(r["quid"]); q = Q[k]
        lst = parse_list(r["prediction"]); first = lst[0] if lst else ""
        fd = cut_date(first, q["question"]) if q["answer_type"] == "time" else first
        sl = S[k]["answer_slot"]; fid = int(top[pos[k]])
        p = (T[int(F["t_id"][fid])][:gran(q["question"])] if sl == "timestamp"
             else (E[int(F["o_id"][fid])] if sl == "object" else E[int(F["s_id"][fid])]))
        direct = ok(p, q)
        ordinal = bool(set(extract_temporal_cue(q["question"])) & {"first", "last"})
        rows.append({"setting": s, "quid": k, "qtype": q["qtype"], "reader": ok(first, q), "reader_date": ok(fd, q),
                     "direct": direct, "hybrid": direct if ordinal else ok(fd, q)})
df = pd.DataFrame(rows)


def sig(a, b, rng):
    a = np.asarray(a, bool); b = np.asarray(b, bool)
    b01 = int((~a & b).sum()); b10 = int((a & ~b).sum())
    p = math.erfc(math.sqrt(((abs(b01 - b10) - 1) ** 2 / max(1, b01 + b10)) / 2)) if b01 + b10 else 1.0
    d = b.astype(float) - a.astype(float)
    bs = d[rng.integers(0, len(d), (10000, len(d)))].mean(1)
    return d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5), p, b01, b10


summ, sigs = [], []
for col in ("reader", "reader_date", "direct", "hybrid"):
    for s in SETTINGS:
        sub = df[df.setting == s]
        summ.append({"metric": col, "setting": s, "n": len(sub), "hit1": sub[col].mean(),
                     **{f"qt_{t}": sub[sub.qtype == t][col].mean() for t in sorted(sub.qtype.unique())}})
        for base in ("none", "timer4"):
            if s == base:
                continue
            b = df[df.setting == base].set_index("quid").loc[sub.quid, col].values
            m, lo, hi, p, b01, b10 = sig(b, sub[col].values, np.random.default_rng(0))
            sigs.append({"metric": col, "setting": s, "baseline": base, "n": len(sub), "diff": m, "ci_low": lo,
                         "ci_high": hi, "mcnemar_p": p, "baseline_wrong_run_right": b01, "baseline_right_run_wrong": b10})
tw = df[df.setting == "tear_window"]
m, lo, hi, p, b01, b10 = sig(tw.reader_date.values, tw.hybrid.values, np.random.default_rng(0))
sigs.append({"metric": "hybrid_vs_reader_date", "setting": "tear_window", "baseline": "tear_window reader_date",
             "n": len(tw), "diff": m, "ci_low": lo, "ci_high": hi, "mcnemar_p": p,
             "baseline_wrong_run_right": b01, "baseline_right_run_wrong": b10})
SM, SG = pd.DataFrame(summ), pd.DataFrame(sigs)
SM.to_csv(OUT / "reader_icews18_summary.csv", index=False)
SG.to_csv(OUT / "reader_icews18_significance.csv", index=False)
ref = pd.read_csv("results/qwen_icews18_final_summary.csv")
chk = SM.merge(ref, on=["setting", "metric"], suffixes=("", "_ref"))
print("max abs difference to results/qwen_icews18_final_summary.csv:", float((chk.hit1 - chk.hit1_ref).abs().max()))
print(SM.round(4).to_string(index=False))
print(SG.round(4).to_string(index=False))
