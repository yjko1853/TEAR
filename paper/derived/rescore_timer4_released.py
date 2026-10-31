"""Re-score TimeR4's released MultiTQ predictions (fine-tuned LLaMA2) with the three rules used for our reader.

Run from the repository root:  .venv/bin/python -I paper/derived/rescore_timer4_released.py
Input : external/TimeR4_results/unzipped/results/MultiTQ/finetuned_llama2/predictions.jsonl (test.json order)
Output: timer4_released_rescored.csv  rows = group (overall, qlabel, answer_type, qtype), columns = rules:
  any_match  TimeR4 repository eval_hit: some gold answer occurs anywhere in ' '.join(answer list) (list-level)
  hit1_lenient  first listed answer contains a gold answer (TEAR/TimeR4 substring rule)
  hit1_exact    first listed answer equals a gold answer after normalisation
Scoring uses tear.evaluation.hits.score_reader_record, the same function as for the Qwen reader.
"""
import ast
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root (tear/)
from tear.evaluation.hits import score_reader_record  # noqa: E402

OUT = Path(__file__).resolve().parent
qs = json.load(open("cache/multitq/questions.json"))
preds = [json.loads(line) for line in open("external/TimeR4_results/unzipped/results/MultiTQ/finetuned_llama2/predictions.jsonl")]
assert len(preds) == len(qs)
mismatch = sum(sorted(ast.literal_eval(p["ground_truth"])) != sorted(q["answers"]) for p, q in zip(preds, qs))
print("questions", len(qs), "ground-truth mismatches", mismatch)
rows = []
for p, q in zip(preds, qs):
    r = score_reader_record({"prediction": p["prediction"], "answers": q["answers"]})
    rows.append({"qtype": q["qtype"], "qlabel": q["qlabel"], "answer_type": q["answer_type"],
                 "any_match": r["hit_any_lenient"], "hit1_lenient": r["hit1_lenient"], "hit1_exact": r["hit1_exact"]})
df = pd.DataFrame(rows)
out = [{"group_type": "overall", "group": "all", "n": len(df), **df[["any_match", "hit1_lenient", "hit1_exact"]].mean().to_dict()}]
for gt in ("qlabel", "answer_type", "qtype"):
    for g, gg in df.groupby(gt):
        out.append({"group_type": gt, "group": g, "n": len(gg), **gg[["any_match", "hit1_lenient", "hit1_exact"]].mean().to_dict()})
res = pd.DataFrame(out); res.to_csv(OUT / "timer4_released_rescored.csv", index=False)
print(res.round(4).to_string(index=False))

# same three rules on the 6,000-question reader sample (cache/multitq/sample_6000.json), with test-set re-weighting
sample = set(str(x) for x in json.load(open("cache/multitq/sample_6000.json"))["quids"])
df["quid"] = [str(q["quid"]) for q in qs]
sm = df[df.quid.isin(sample)]
w = df.qtype.value_counts(normalize=True)
rec = {"n": len(sm)}
for c in ("any_match", "hit1_lenient", "hit1_exact"):
    by = sm.groupby("qtype")[c].mean()
    rec[f"{c}_sample"] = sm[c].mean(); rec[f"{c}_reweighted"] = float((by * w.reindex(by.index)).sum())
    rec[f"{c}_Multiple_sample"] = sm[sm.qlabel == "Multiple"][c].mean()
pd.DataFrame([rec]).to_csv(OUT / "timer4_released_rescored_sample6000.csv", index=False)
print(pd.DataFrame([rec]).round(4).T.to_string())
