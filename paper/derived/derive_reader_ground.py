"""Reader statistics for the manuscript, main setting (original questions + KG anchor grounding), complementing
results/qwen_ground_final_*.csv (scripts/18_final_reader_report.py), which they must agree with.

Run from the repository root:  .venv/bin/python -I paper/derived/derive_reader_ground.py
Inputs : results/qwen_ground/*.jsonl (Qwen2.5-7B-Instruct-4bit, prompt v3, k = 15, 6,000-question sample),
         results/qwen_ground_final_per_question.csv, cache/multitq_orig/questions.json
Outputs (paper/derived/):
  reader_rules_ground.csv     per setting: reader exact / lenient / list-level (any-match) with tear.evaluation.hits.
                              score_reader_record (the rules of the metrics section), sample mean and test re-weighted;
                              Multiple (mean of the three Multiple types) for reader and reader+date; agreement of the
                              exact rule with column 'reader' of the per-question file
  reader_vs_llmfree_ground.csv per setting: reader+date versus the LLM-free top-1 answer on the same questions
                              (both right / reader only / LLM-free only / neither), overall and by answer type;
                              time questions whose first answer is right only after the granularity cut
  reader_leaderboard_ground.csv per setting and metric (reader, reader_date, direct): overall, Multiple, entity and time,
                              re-weighted to the MultiTQ test distribution of question types (within each answer type
                              by the test counts of (qtype, answer type))
"""
import collections
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root (tear/)
from tear.evaluation.hits import score_reader_record  # noqa: E402

OUT = Path(__file__).resolve().parent
MODEL = "Qwen2.5-7B-Instruct-4bit"
TAG = "rule_v2_ground_exact_anydate"
SETTINGS = ["none", "timer4", "atc", "tr", "tear", "tear_window", "timer4_public"]
MULTI = ("equal_multi", "after_first", "before_last")

qs = json.load(open("cache/multitq_orig/questions.json"))
share = collections.Counter(q["qtype"] for q in qs)
N = sum(share.values())
pq = pd.read_csv("results/qwen_ground_final_per_question.csv", dtype={"quid": str})


def reweight(df, col):
    return sum(df.loc[df.qtype == t, col].mean() * share[t] / N for t in share)


rows, cmp_rows = [], []
for s in SETTINGS:
    f = (f"results/qwen_ground/{s}__{MODEL}__k15__p-v3.jsonl" if s == "timer4_public"
         else f"results/qwen_ground/{s}_{TAG}__{MODEL}__k15__p-v3.jsonl")
    recs = {}
    for line in open(f):
        r = json.loads(line)
        recs[str(r["quid"])] = score_reader_record(r)
    d = pq[pq.setting == s].copy()
    d["exact"] = [recs[k]["hit1_exact"] for k in d.quid]
    d["lenient"] = [recs[k]["hit1_lenient"] for k in d.quid]
    d["any_match"] = [recs[k]["hit_any_lenient"] for k in d.quid]
    row = {"setting": s, "n": len(d), "agree_exact_with_per_question": float((d.exact == d.reader).mean())}
    for c in ("exact", "lenient", "any_match", "reader", "reader_date"):
        row[f"{c}_sample"] = d[c].mean()
        row[f"{c}_reweighted"] = reweight(d, c)
    for c in ("reader", "reader_date"):
        row[f"{c}_Multiple_sample"] = d[d.qtype.isin(MULTI)][c].mean()
    rows.append(row)
    if s == "timer4_public":
        continue
    for grp, g in [("all", d)] + [(a, gg) for a, gg in d.groupby("answer_type")]:
        rd, di = g.reader_date.astype(bool), g.direct.astype(bool)
        cmp_rows.append({"setting": s, "group": grp, "n": len(g), "both_right": (rd & di).mean(),
                         "reader_only": (rd & ~di).mean(), "llmfree_only": (~rd & di).mean(),
                         "neither": (~rd & ~di).mean(), "reader_date": rd.mean(), "llmfree": di.mean()})
    t = d[d.answer_type == "time"]
    cmp_rows.append({"setting": s, "group": "time: right only after granularity cut", "n": len(t),
                     "reader_date": float((t.reader_date.astype(bool) & ~t.reader.astype(bool)).mean())})
pd.DataFrame(rows).to_csv(OUT / "reader_rules_ground.csv", index=False)

cell = collections.Counter((q["qtype"], q["answer_type"]) for q in qs)
lb = []
for s in SETTINGS:
    d = pq[pq.setting == s]
    for m in ("reader", "reader_date", "direct"):
        if d[m].isna().all():
            continue
        row = {"setting": s, "metric": m, "overall": reweight(d, m),
               "Multiple": sum(d.loc[d.qtype == t, m].mean() * share[t] for t in MULTI) / sum(share[t] for t in MULTI)}
        for a in ("entity", "time"):
            cells = [(t, c) for (t, aa), c in cell.items() if aa == a]
            row[a] = sum(d.loc[(d.qtype == t) & (d.answer_type == a), m].mean() * c for t, c in cells) / sum(c for _, c in cells)
        lb.append(row)
pd.DataFrame(lb).to_csv(OUT / "reader_leaderboard_ground.csv", index=False)
print(pd.DataFrame(lb).round(4).to_string(index=False))
pd.DataFrame(cmp_rows).to_csv(OUT / "reader_vs_llmfree_ground.csv", index=False)
print(pd.DataFrame(rows).round(4).to_string(index=False))
print(pd.DataFrame(cmp_rows).round(4).to_string(index=False))
