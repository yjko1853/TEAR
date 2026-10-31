"""LLM-free Hit@1 on the 120-question prompt-development set, and defects of the TimeR4 rewritten questions.

Run from the repository root:  .venv/bin/python -I paper/derived/derive_tune_and_rewrites.py
  llmfree_hit1_tune120.csv  exact Hit@1 of the top-1 fact on cache/multitq/tune_120.json per reranker
  rewrite_defects.csv       per qtype: empty rewrites (< 3 characters), rewrites that lost the first/last cue of an
                            after_first / before_last question although the original question has it
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root (tear/)
from tear.evaluation.hits import read_answer_from_fact  # noqa: E402
from tear.text import exact_match  # noqa: E402

OUT = Path(__file__).resolve().parent
d = Path("cache/multitq")
qs = json.load(open(d / "questions.json")); qby = {str(q["quid"]): q for q in qs}
order = [str(k) for k in json.load(open(d / "rerank" / "question_order.json"))["question_ids"]]
pos = {k: i for i, k in enumerate(order)}
slots = json.load(open(d / "slots_rule.json"))
facts = np.load(d / "facts.npz"); voc = json.load(open(d / "vocab.json"))
tune = [str(x) for x in json.load(open(d / "tune_120.json"))["quids"]]
rows = []
for rk in ("none", "timer4", "atc", "tr", "tear", "tear_window"):
    top = np.load(d / "rerank" / f"{rk}_rule_exact_anydate_top30_ids.npy", mmap_mode="r")
    ok = [exact_match(read_answer_from_fact(slots[k]["answer_slot"], int(top[pos[k], 0]), facts, voc["entities"],
                                            voc["timestamps"], qby[k]["rewrite"]), qby[k]["answers"]) for k in tune]
    rows.append({"reranker": rk, "n": len(ok), "exact": float(np.mean(ok))})
pd.DataFrame(rows).to_csv(OUT / "llmfree_hit1_tune120.csv", index=False)
print(pd.DataFrame(rows).round(4).to_string(index=False))

rec = []
for qt in sorted({q["qtype"] for q in qs}):
    g = [q for q in qs if q["qtype"] == qt]
    row = {"qtype": qt, "n": len(g), "empty_rewrite": sum(len(q["rewrite"].strip()) < 3 for q in g)}
    cue = {"after_first": "first", "before_last": "last"}.get(qt)
    if cue:
        miss = [q for q in g if cue not in q["rewrite"].lower()]
        row.update({"cue": cue, "cue_missing_in_rewrite": len(miss),
                    "cue_present_in_original_of_those": sum(cue in q["question"].lower() for q in miss),
                    "cue_missing_and_empty": sum(len(q["rewrite"].strip()) < 3 for q in miss)})
    rec.append(row)
df = pd.DataFrame(rec); df.to_csv(OUT / "rewrite_defects.csv", index=False)
print(df.to_string(index=False))
ents = set(voc["entities"])
print("entities exist:", {e: e in ents for e in ("Citizen (Iraq)", "Legislature (Iraq)", "Iraqi Islamic Party")})

# before_after rewrites without a parsed date, and those that still contain a 4-digit year (unparsed date forms)
import re  # noqa: E402
from tear.temporal import target_days  # noqa: E402
ba = [q for q in qs if q["qtype"] == "before_after"]
nod = [q for q in ba if not target_days(q["rewrite"], "any")]
yr = [q for q in nod if re.search(r"\b(19|20)\d\d\b", q["rewrite"])]
pd.DataFrame([{"n_before_after": len(ba), "no_parsed_date": len(nod), "no_parsed_date_but_year_present": len(yr),
               "examples": " || ".join(q["rewrite"] for q in yr[:5])}]).to_csv(OUT / "before_after_date_parsing.csv", index=False)
print("before_after:", len(ba), "no parsed date:", len(nod), "of which contain a year:", len(yr))

# preliminary 30-question check of the TimeR4 prompt (results/qwen/*p-timer4__bench.jsonl)
from tear.reader.parse import parse_list  # noqa: E402
bench = [json.loads(line) for line in open("results/qwen/tear_window_rule_exact_anydate__Qwen2.5-7B-Instruct-4bit__k15__p-timer4__bench.jsonl")]
empty = sum(1 for r in bench if not parse_list(r["prediction"]))
tq = [r for r in bench if r["answer_type"] == "time"]
gran = sum(1 for r in tq if parse_list(r["prediction"]) and len(parse_list(r["prediction"])[0].strip()) != len(r["answers"][0]))
pd.DataFrame([{"n": len(bench), "empty_list": empty, "time_questions": len(tq),
               "time_first_item_length_differs_from_gold": gran}]).to_csv(OUT / "timer4_prompt_bench_check.csv", index=False)
print("TimeR4 prompt bench:", len(bench), "empty lists:", empty, "time questions:", len(tq), "granularity mismatch:", gran)
