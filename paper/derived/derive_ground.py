"""Statistics for the main setting: original questions + KG anchor grounding (dataset multitq_orig, tag
rule_v2_ground_exact_anydate). Run from the repository root:  .venv/bin/python -I paper/derived/derive_ground.py
Writes into paper/derived/ (suffix _ground):
  top15_redundancy_ground.csv, rule_firing_shares_ground.csv, llmfree_hit1_sample6000_ground.csv,
  labels_ground.csv (slot / topic-source counts, timestamp decision vs gold answer type),
  rule_check_accuracy_ground.csv (author's blind check, questions 101-200, on the original wording),
  figure1_example_ground.txt (question 3000150 on the original wording)
The first author's check sheets are private working files and are not released; set TEAR_CHECK_DIR to the
folder that holds them (file, sheet and column names as in the original sheets) to recompute the check.
"""
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root (tear/)
from tear.data.multitq import stage_query  # noqa: E402
from tear.evaluation.hits import read_answer_from_fact  # noqa: E402
from tear.labeling.rules import rule_slot_fixes  # noqa: E402
from tear.text import exact_match  # noqa: E402

OUT = Path(__file__).resolve().parent
TAG = "rule_v2_ground_exact_anydate"
d = Path("cache/multitq_orig")
qs = json.load(open(d / "questions.json")); qby = {str(q["quid"]): q for q in qs}
order = [str(k) for k in json.load(open(d / "rerank" / "question_order.json"))["question_ids"]]
pos = {k: i for i, k in enumerate(order)}
slots = json.load(open(d / "slots_rule_v2.json"))
anchors = json.load(open(d / "anchors.json"))
f = np.load(d / "facts.npz"); voc = json.load(open(d / "vocab.json"))
s_, r_, o_ = f["s_id"].astype(np.int64), f["r_id"].astype(np.int64), f["o_id"].astype(np.int64)
key = (s_ * (int(r_.max()) + 1) + r_) * (int(max(s_.max(), o_.max())) + 1) + o_
top = {rk: np.load(d / "rerank" / f"{rk}_{TAG}_top30_ids.npy", mmap_mode="r")
       for rk in ("none", "timer4", "hardwindow", "atc", "tr", "tear", "tear_window")}

red = []
for rk in ("none", "tear"):
    k = key[np.asarray(top[rk][:, :15])]; dup = np.zeros_like(k, bool)
    for j in range(1, 15):
        dup[:, j] = (k[:, :j] == k[:, j:j + 1]).any(1)
    red.append({"reranker": rk, "share_top15_repeating_sro": float(dup.mean()), "share_questions_with_repeat": float(dup.any(1).mean())})
pd.DataFrame(red).to_csv(OUT / "top15_redundancy_ground.csv", index=False); print(pd.DataFrame(red).round(4).to_string(index=False))

cr = pd.read_csv(Path("results") / f"multitq_orig_constraint_recognition_{TAG}.csv"); n = cr["n"].sum()
cols = [c for c in ("cue_first", "cue_last", "target_date", "window", "tr_fired", "atc_penalised_any", "anchor_used", "has_period", "anchor_atc") if c in cr.columns]
fire = {"n_questions": int(n), **{f"share_{c}": float((cr["n"] * cr[c]).sum() / n) for c in cols}}
pd.DataFrame([fire]).to_csv(OUT / "rule_firing_shares_ground.csv", index=False); print(fire); print(cr.round(3).to_string(index=False))

test_w = pd.Series([q["qtype"] for q in qs]).value_counts(normalize=True)
keys = [str(x) for x in json.load(open(d / "sample_6000.json"))["quids"]]
same_sample = keys == [str(x) for x in json.load(open("cache/multitq/sample_6000.json"))["quids"]]
rows = []
for rk, arr in top.items():
    rec = [(qby[k]["qtype"], exact_match(read_answer_from_fact(slots[k]["answer_slot"], int(arr[pos[k], 0]), f, voc["entities"],
                                                               voc["timestamps"], stage_query(qby[k])), qby[k]["answers"])) for k in keys]
    df = pd.DataFrame(rec, columns=["qtype", "exact"]); by = df.groupby("qtype").exact.mean()
    rows.append({"reranker": rk, "n": len(df), "exact_sample_mean": df.exact.mean(),
                 "exact_reweighted_to_test": float((by * test_w.reindex(by.index)).sum()), "same_quids_as_rewrite_sample": same_sample})
pd.DataFrame(rows).to_csv(OUT / "llmfree_hit1_sample6000_ground.csv", index=False)
print(pd.DataFrame(rows).round(4).to_string(index=False))

import collections
C = collections.Counter
lab = {"slots": dict(C(v["answer_slot"] for v in slots.values())), "topic_source": dict(C(v["topic_source"] for v in slots.values())),
       "slot_x_answer_type": {f"{a}|{b}": c for (a, b), c in C((v["answer_slot"], qby[k]["answer_type"]) for k, v in slots.items()).items()},
       "anchored": len(anchors), "anchor_resolved": sum(bool(a.get("date")) for a in anchors.values()),
       "anchor_source": dict(C(a.get("anchor_source") for a in anchors.values() if a.get("date")))}
pd.DataFrame([{"field": k, "value": json.dumps(v)} for k, v in lab.items()]).to_csv(OUT / "labels_ground.csv", index=False); print(lab)

CHECK_DIR = Path(os.environ.get("TEAR_CHECK_DIR", "check"))   # first author's check sheets (not released)
CHECK_SHEET = CHECK_DIR / "규칙라벨_검수_2차_101-200.xlsx"     # blind sample, questions 101-200
CHECK_BASE = CHECK_DIR / "규칙라벨_검수_300.csv"               # the 300 sampled questions (no, quid, question)
SHEET, COL_JUDGED, COL_QUESTION = "검수", "판정_slot", "질문"     # sheet, judged-slot column, question column
if CHECK_SHEET.exists() and CHECK_BASE.exists():
    chk = pd.read_excel(CHECK_SHEET, sheet_name=SHEET)
    base = pd.read_csv(CHECK_BASE, encoding="utf-8-sig")
    m = chk.merge(base[["no", "quid"]], on="no"); m["judged"] = m[COL_JUDGED].astype(str).str.strip().str.lower()
    j = m[m["judged"].isin(["subject", "object", "timestamp"])]
    def wilson(k, n, z=1.96):
        p = k / n; den = 1 + z * z / n; c = (p + z * z / (2 * n)) / den; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
        return c - h, c + h
    res = []
    for name, fn in (("orig wording, rules v2 before fixes d-e (blind)", lambda k: rule_slot_fixes(stage_query(qby[k]), frozenset("abc"))),
                     ("orig wording, current rules v2 (not blind)", lambda k: slots[k]["answer_slot"])):
        ok = [fn(str(q)) == s for q, s in zip(j["quid"], j["judged"])]; kk, nn = sum(ok), len(ok); lo, hi = wilson(kk, nn)
        res.append({"rules": name, "judged": nn, "correct": kk, "accuracy": kk / nn, "wilson_low": lo, "wilson_high": hi})
    pd.DataFrame(res).to_csv(OUT / "rule_check_accuracy_ground.csv", index=False); print(pd.DataFrame(res).round(4).to_string(index=False))
else:
    print(f"rule_check_accuracy_ground.csv not recomputed: check sheets not found in {CHECK_DIR} (set TEAR_CHECK_DIR)")

sent = (d / "sentences.txt").read_text().split("\n"); i = pos["3000150"]
with open(OUT / "figure1_example_ground.txt", "w") as fo:
    fo.write(f"question: {stage_query(qby['3000150'])} | answers {qby['3000150']['answers']} | slot {slots['3000150']}\n")
    fo.write(f"anchor: {anchors.get('3000150')}\n")
    ids0 = np.load(d / "top1000_ids.npy", mmap_mode="r")[i]; gold = [r for r, x in enumerate(ids0) if "France Make a visit Kuwait in 2009-06-22" == sent[int(x)]]
    fo.write(f"similarity rank of 'France Make a visit Kuwait in 2009-06-22': {[g + 1 for g in gold]}\n")
    for rk in ("none", "atc", "tr", "tear"):
        fo.write(rk + "\n" + "".join(f"  {sent[int(x)]}\n" for x in top[rk][i, :3]))
print(open(OUT / "figure1_example_ground.txt").read())
