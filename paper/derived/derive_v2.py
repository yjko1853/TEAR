"""Pool-dependent manuscript statistics for the current configuration v2 (tag rule_v2_exact_anydate:
answer-slot rules v2, query fallback for empty/truncated rewrites, date parser v2).

Run from the repository root:  .venv/bin/python -I paper/derived/derive_v2.py
Writes into paper/derived/ (suffix _v2):
  top15_redundancy_v2.csv           share of top-15 candidates (none, tear) repeating a higher-ranked (s, r, o)
  rule_firing_shares_v2.csv         question-weighted shares from results/multitq_constraint_recognition_rule_v2_exact_anydate.csv
  llmfree_hit1_sample6000_v2.csv    exact Hit@1 of the top-1 fact on the reader sample (mean and test-weighted)
  llmfree_hit1_tune120_v2.csv       the same on the prompt-development set
  query_and_dates_v2.csv            query fallback count, before_after questions with/without a parsed date (parser v2)
  rule_labeller_timing_v2.csv       wall-clock of the v2 labeller on all questions, agreement with slots_rule_v2.json
  figure1_example_v2.txt            Figure 1 question (3000150) under the v2 arrays
"""
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root (tear/)
from tear.data.multitq import stage_query  # noqa: E402
from tear.evaluation.hits import read_answer_from_fact  # noqa: E402
from tear.labeling.rules import label_multitq  # noqa: E402
from tear.temporal import target_days  # noqa: E402
from tear.text import exact_match  # noqa: E402

OUT = Path(__file__).resolve().parent
TAG = "rule_v2_exact_anydate"
d = Path("cache/multitq")
qs = json.load(open(d / "questions_v2.json")); qby = {str(q["quid"]): q for q in qs}
order = [str(k) for k in json.load(open(d / "rerank" / "question_order.json"))["question_ids"]]
pos = {k: i for i, k in enumerate(order)}
slots = json.load(open(d / "slots_rule_v2.json"))
f = np.load(d / "facts.npz"); voc = json.load(open(d / "vocab.json"))
s_, r_, o_ = f["s_id"].astype(np.int64), f["r_id"].astype(np.int64), f["o_id"].astype(np.int64)
key = (s_ * (int(r_.max()) + 1) + r_) * (int(max(s_.max(), o_.max())) + 1) + o_
top = {rk: np.load(d / "rerank" / f"{rk}_{TAG}_top30_ids.npy", mmap_mode="r")
       for rk in ("none", "timer4", "hardwindow", "atc", "tr", "tear", "tear_window")}

# ---- top-15 redundancy
red = []
for rk in ("none", "tear"):
    k = key[np.asarray(top[rk][:, :15])]
    dup = np.zeros_like(k, bool)
    for j in range(1, 15):
        dup[:, j] = (k[:, :j] == k[:, j:j + 1]).any(1)
    red.append({"reranker": rk, "tag": TAG, "n_questions": len(k), "share_top15_repeating_sro": float(dup.mean()),
                "share_questions_with_repeat": float(dup.any(1).mean())})
pd.DataFrame(red).to_csv(OUT / "top15_redundancy_v2.csv", index=False); print(pd.DataFrame(red).to_string(index=False))

# ---- rule firing shares
cr = pd.read_csv(Path("results") / f"multitq_constraint_recognition_{TAG}.csv"); n = cr["n"].sum()
fire = {"tag": TAG, "n_questions": int(n), **{f"share_{c}": float((cr["n"] * cr[c]).sum() / n) for c in
        ("cue_first", "cue_last", "target_date", "window", "tr_fired", "atc_penalised_any")}}
pd.DataFrame([fire]).to_csv(OUT / "rule_firing_shares_v2.csv", index=False); print(fire)

# ---- LLM-free Hit@1 on the reader sample and the prompt-development set
test_w = pd.Series([q["qtype"] for q in qs]).value_counts(normalize=True)
for name, file in (("sample6000", "sample_6000.json"), ("tune120", "tune_120.json")):
    keys = [str(x) for x in json.load(open(d / file))["quids"]]
    rows = []
    for rk, arr in top.items():
        rec = []
        for kq in keys:
            q = qby[kq]
            pred = read_answer_from_fact(slots[kq]["answer_slot"], int(arr[pos[kq], 0]), f, voc["entities"],
                                         voc["timestamps"], stage_query(q))
            rec.append((q["qtype"], q["answer_type"], exact_match(pred, q["answers"])))
        df = pd.DataFrame(rec, columns=["qtype", "answer_type", "exact"]); by = df.groupby("qtype").exact.mean()
        row = {"reranker": rk, "n": len(df), "exact_sample_mean": df.exact.mean(),
               "exact_reweighted_to_test": float((by * test_w.reindex(by.index)).sum())}
        row.update({f"exact_{k}": v for k, v in by.items()})
        rows.append(row)
    pd.DataFrame(rows).to_csv(OUT / f"llmfree_hit1_{name}_v2.csv", index=False)
    print(name); print(pd.DataFrame(rows)[["reranker", "exact_sample_mean", "exact_reweighted_to_test"]].round(4).to_string(index=False))

# ---- query fallback and date parsing (parser v2)
ba = [q for q in qs if q["qtype"] == "before_after"]
nod = [q for q in ba if not target_days(stage_query(q), "any", "v2")]
yr = [q for q in nod if re.search(r"\b(19|20)\d\d\b", stage_query(q))]
qd = {"n_questions": len(qs), "query_from_original_question": sum(q["query_source"] == "question" for q in qs),
      "n_before_after": len(ba), "before_after_no_parsed_date": len(nod), "no_parsed_date_but_year_present": len(yr),
      "examples_year_present": " || ".join(stage_query(q) for q in yr[:5])}
pd.DataFrame([qd]).to_csv(OUT / "query_and_dates_v2.csv", index=False); print(qd)

# ---- labeller timing (v2 rules, NER topics preloaded)
ner = json.load(open(d / "topic_ner_all_v2.json"))
qs_lab = [dict(q, rewrite=stage_query(q)) for q in qs]
t0 = time.time(); lab = label_multitq(qs_lab, voc["entities"], ner, version="v2"); el = time.time() - t0
same = sum(lab[k]["answer_slot"] == slots[k]["answer_slot"] and lab[k]["topic_entity"] == slots[k]["topic_entity"] for k in lab)
pd.DataFrame([{"n_questions": len(qs), "seconds": el, "ms_per_question": 1000 * el / len(qs),
               "identical_to_cached_labels": same}]).to_csv(OUT / "rule_labeller_timing_v2.csv", index=False)
print(f"labeller v2: {el:.1f}s, identical {same}/{len(qs)}")

# ---- Figure 1 under v2
sent = (d / "sentences.txt").read_text().split("\n"); i = pos["3000150"]
with open(OUT / "figure1_example_v2.txt", "w") as fo:
    fo.write(f"question: {stage_query(qby['3000150'])} | answers {qby['3000150']['answers']} | slot {slots['3000150']}\n")
    for rk in ("none", "atc", "tr", "tear"):
        fo.write(rk + "\n" + "".join(f"  {sent[int(x)]}\n" for x in top[rk][i, :3]))
print(open(OUT / "figure1_example_v2.txt").read())
