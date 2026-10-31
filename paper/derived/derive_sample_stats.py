"""LLM-free Hit@1 on the 6,000-question reader sample, and wall-clock time of the rule labeller.

Run from the repository root:  .venv/bin/python -I paper/derived/derive_sample_stats.py
  llmfree_hit1_sample6000.csv   exact Hit@1 of the top-1 fact per reranker on cache/multitq/sample_6000.json
                                (overall = mean over the type-balanced sample; reweighted = test-set qtype weights)
  rule_labeller_timing.csv      seconds to label all 54,584 questions with tear.labeling.rules (NER topics preloaded), rule v1;
                                wall-clock seconds vary between runs
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root (tear/)
from tear.evaluation.hits import read_answer_from_fact  # noqa: E402
from tear.labeling.rules import label_multitq  # noqa: E402
from tear.text import exact_match  # noqa: E402

OUT = Path(__file__).resolve().parent
d = Path("cache/multitq")
qs = json.load(open(d / "questions.json")); qby = {str(q["quid"]): q for q in qs}
order = [str(k) for k in json.load(open(d / "rerank" / "question_order.json"))["question_ids"]]
pos = {k: i for i, k in enumerate(order)}
slots = json.load(open(d / "slots_rule.json"))
facts = np.load(d / "facts.npz"); voc = json.load(open(d / "vocab.json"))
sample = [str(x) for x in json.load(open(d / "sample_6000.json"))["quids"]]
test_w = pd.Series([q["qtype"] for q in qs]).value_counts(normalize=True)
rows = []
for rk in ("none", "timer4", "hardwindow", "atc", "tr", "tear", "tear_window"):
    top = np.load(d / "rerank" / f"{rk}_rule_exact_anydate_top30_ids.npy", mmap_mode="r")
    rec = []
    for k in sample:
        q = qby[k]
        pred = read_answer_from_fact(slots[k]["answer_slot"], int(top[pos[k], 0]), facts, voc["entities"],
                                     voc["timestamps"], q["rewrite"])
        rec.append((q["qtype"], q["qlabel"], q["answer_type"], exact_match(pred, q["answers"])))
    df = pd.DataFrame(rec, columns=["qtype", "qlabel", "answer_type", "exact"])
    by = df.groupby("qtype").exact.mean()
    row = {"reranker": rk, "n": len(df), "exact_sample_mean": df.exact.mean(),
           "exact_reweighted_to_test": float((by * test_w.reindex(by.index)).sum())}
    row.update({f"exact_{k}": v for k, v in by.items()})
    row.update({f"exact_{k}": v for k, v in df.groupby("answer_type").exact.mean().items()})
    rows.append(row)
res = pd.DataFrame(rows); res.to_csv(OUT / "llmfree_hit1_sample6000.csv", index=False)
pd.set_option("display.width", 250); print(res.round(4).to_string(index=False))

ner = json.load(open(d / "topic_ner_all.json"))
t0 = time.time(); lab = label_multitq(qs, voc["entities"], ner, version="v1"); el = time.time() - t0   # v1 rules (slots_rule.json); v2 is now the default
same = sum(lab[k]["answer_slot"] == slots[k]["answer_slot"] and lab[k]["topic_entity"] == slots[k]["topic_entity"] for k in lab)
pd.DataFrame([{"n_questions": len(qs), "seconds": el, "ms_per_question": 1000 * el / len(qs),
               "identical_to_cached_labels": same}]).to_csv(OUT / "rule_labeller_timing.csv", index=False)
print(f"rule labeller: {el:.1f}s for {len(qs)} questions; identical to cached slots_rule.json: {same}")
