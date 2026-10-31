"""Qualitative cases and reader error analysis, main setting (original questions + KG anchor grounding, dataset
multitq_orig, tag rule_v2_ground_exact_anydate; Qwen2.5-7B-Instruct-4bit, prompt v3, k = 15, 6,000-question sample).
No model is loaded; plain numpy / pandas on cached arrays and the saved reader outputs.

Run from the repository root:  OMP_NUM_THREADS=1 .venv/bin/python -I paper/derived/qualitative_cases_ground.py
Inputs : results/qwen_ground_final_per_question.csv (scoring of scripts/18_final_reader_report.py),
         results/qwen_ground/{none,tear_window}_rule_v2_ground_exact_anydate__Qwen2.5-7B-Instruct-4bit__k15__p-v3.jsonl,
         cache/multitq_orig/{questions.json, slots_rule_v2.json, anchors.json, facts.npz, vocab.json, rerank/}
Outputs (paper/derived/):
  qualitative_cases_ground.csv / .txt  three successes and three failures, chosen by the fixed rules below in the order
                                       of the reader's shuffled sample (first match per rule)
  reader_errors_ground.csv             TEAR+window reader errors (first answer, exact, before the date cut) whose gold
                                       answer occurs in the top-15 facts, by category (rules below)

Case rules (S = success: None wrong and TEAR+window right, both for the LLM-free top-1 answer and for reader+date;
F = failure: TEAR+window reader+date wrong):
  S1 before_last, anchor grounded from a mention of the anchor (grounding + TR)
  S2 equal, entity answer, the unreranked top-1 fact holds the topic entity in the predicted answer slot (ATC)
  S3 first_last, time answer, the top-1 facts of both rankings share (s, r, o), i.e. TR picked another occurrence, and
     the unreranked reader copied the date of its top-1 fact
  F1 equal_multi 3002799, the case of the manual grounding check (addendum_anchor_grounding.md, #2) that lies in the
     reader sample: the reference event was grounded with the wrong relation
  F2 a 'visit' question with a subject answer whose TEAR+window top-1 fact is 'Host a visit' (relation direction)
  F3 after_first or before_last, a gold answer equals an element of the TEAR+window top-1 fact but in another slot
     (error class C; exact element match, so that substrings such as 'Sudan' in 'South Sudan' do not count)
Error classes (LLM-free, as in error_breakdown_ground.csv): A gold not in the top 15; B in the top 15 but not in the top-1
fact; C in the top-1 fact but read from another slot or at another granularity.
Reader error categories, applied in this order:
  empty or malformed   no answer, or the output is not a Python list (the parser fell back to splitting)
  wrong granularity    time question; the first answer and a gold date are the same date at different granularity
                       (finer: fixed by the date cut; coarser: not fixed)
  copied, wrong fact   the first answer equals an entity or date of a top-15 fact that does not answer the question
                       (sub-counts: from the top-ranked fact; the topic entity itself)
  other                the first answer occurs in none of the top-15 facts
"""
import ast
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root (tear/)
from tear.reader.parse import parse_list  # noqa: E402
from tear.text import normalize  # noqa: E402

OUT = Path(__file__).resolve().parent
D = Path("cache/multitq_orig")
TAG, MODEL = "rule_v2_ground_exact_anydate", "Qwen2.5-7B-Instruct-4bit"
Q = {str(q["quid"]): q for q in json.load(open(D / "questions.json"))}
SL = json.load(open(D / "slots_rule_v2.json"))
AN = json.load(open(D / "anchors.json"))
_F = np.load(D / "facts.npz")
F = {c: np.asarray(_F[c]) for c in ("s_id", "r_id", "o_id", "t_id")}   # load once (NpzFile re-reads on every access)
V = json.load(open(D / "vocab.json"))
E, R, T = V["entities"], V["relations"], V["timestamps"]
order = [str(x) for x in json.load(open(D / "rerank" / "question_order.json"))["question_ids"]]
pos = {k: i for i, k in enumerate(order)}
TOP = {s: np.load(D / "rerank" / f"{s}_{TAG}_top30_ids.npy", mmap_mode="r") for s in ("none", "tear_window")}
PRED = {s: {str(json.loads(l)["quid"]): json.loads(l)["prediction"]
            for l in open(f"results/qwen_ground/{s}_{TAG}__{MODEL}__k15__p-v3.jsonl")} for s in ("none", "tear_window")}
pq = pd.read_csv("results/qwen_ground_final_per_question.csv", dtype={"quid": str})
W = {s: pq[pq.setting == s].set_index("quid") for s in ("none", "tear_window")}
sample = list(W["tear_window"].index)          # order of the reader's fixed shuffle


def fact(fid):
    return E[int(F["s_id"][fid])], R[int(F["r_id"][fid])], E[int(F["o_id"][fid])], T[int(F["t_id"][fid])]


def ftext(fid):
    s, r, o, t = fact(fid)
    return f"{s} {r} {o}, {t}"


def top(s, k, n=15):
    return [int(x) for x in TOP[s][pos[k], :n]]


def contains_gold(fid, q):
    els = [normalize(x) for x in fact(fid)]
    return any(normalize(a) and any(normalize(a) in e for e in els) for a in q["answers"])


def gold_rank(s, k, n=15):
    q = Q[k]
    for i, fid in enumerate(top(s, k, n)):
        if contains_gold(fid, q):
            return i + 1
    return None


def exact_gold_in(fid, q):
    els = {normalize(x) for x in fact(fid)}
    return any(normalize(a) in els for a in q["answers"])


def err_class(k):
    q = Q[k]
    if W["tear_window"].loc[k, "direct"]:
        return "-"
    r = gold_rank("tear_window", k)
    return "A" if r is None else ("C" if r == 1 else "B")


def first_answer(s, k):
    lst = parse_list(PRED[s][k])
    return lst[0] if lst else ""


def ok(s, k, col):
    return bool(W[s].loc[k, col])


def slot_value(fid, slot):
    s, r, o, t = fact(fid)
    return {"subject": s, "object": o, "relation": r, "timestamp": t}.get(slot)


def pick(rule):
    for k in sample:
        try:
            if rule(k):
                return k
        except KeyError:
            continue
    return None


def success(k):
    return (not ok("none", k, "direct") and not ok("none", k, "reader_date")
            and ok("tear_window", k, "direct") and ok("tear_window", k, "reader_date"))


def tw_fail(k):
    return not ok("tear_window", k, "reader_date")


cases = {
    "S1": pick(lambda k: Q[k]["qtype"] == "before_last" and k in AN and AN[k].get("status") == "resolved"
               and AN[k].get("anchor_source") == "mention" and success(k)),
    "S2": pick(lambda k: Q[k]["qtype"] == "equal" and Q[k]["answer_type"] == "entity" and success(k)
               and normalize(slot_value(top("none", k)[0], SL[k]["answer_slot"]) or "") == normalize(SL[k].get("topic_entity") or "#")),
    "S3": pick(lambda k: Q[k]["qtype"] == "first_last" and Q[k]["answer_type"] == "time" and success(k)
               and fact(top("none", k)[0])[:3] == fact(top("tear_window", k)[0])[:3]
               and first_answer("none", k).startswith(fact(top("none", k)[0])[3][:len(Q[k]["answers"][0])])),
    "F1": "3002799" if tw_fail("3002799") else None,
    "F2": pick(lambda k: tw_fail(k) and SL[k]["answer_slot"] == "subject" and "visit" in Q[k]["question"].lower()
               and "host" not in Q[k]["question"].lower() and fact(top("tear_window", k)[0])[1] == "Host a visit"
               and Q[k]["qtype"] == "before_after"),
    "F3": pick(lambda k: tw_fail(k) and Q[k]["qtype"] in ("after_first", "before_last") and err_class(k) == "C"
               and exact_gold_in(top("tear_window", k)[0], Q[k])),
}
rows, txt = [], []
for cid, k in cases.items():
    if k is None:
        txt.append(f"== {cid}: no match")
        continue
    q = Q[k]; a = AN.get(k, {})
    row = {"case": cid, "quid": k, "qtype": q["qtype"], "answer_type": q["answer_type"], "question": q["question"],
           "gold": " | ".join(q["answers"]), "slot": SL[k]["answer_slot"], "topic": SL[k].get("topic_entity"),
           "anchor": a.get("anchor"), "anchor_relation": a.get("relation"), "anchor_date": a.get("date"),
           "anchor_type": a.get("type"),
           "none_top1": ftext(top("none", k)[0]), "tw_top1": ftext(top("tear_window", k)[0]),
           "none_gold_rank15": gold_rank("none", k), "tw_gold_rank15": gold_rank("tear_window", k),
           "none_reader": first_answer("none", k), "tw_reader": first_answer("tear_window", k),
           "none_reader_date_ok": ok("none", k, "reader_date"), "tw_reader_date_ok": ok("tear_window", k, "reader_date"),
           "none_llmfree_ok": ok("none", k, "direct"), "tw_llmfree_ok": ok("tear_window", k, "direct"),
           "tw_error_class": err_class(k)}
    rows.append(row)
    txt.append(f"== {cid} | {q['qtype']} | quid {k} | class {row['tw_error_class']}")
    for key in ("question", "gold", "slot", "topic", "anchor", "anchor_relation", "anchor_date", "none_reader",
                "tw_reader", "none_gold_rank15", "tw_gold_rank15"):
        txt.append(f"   {key:16s}: {row[key]}")
    for s in ("none", "tear_window"):
        txt.append(f"   {s:16s}: " + " || ".join(ftext(f) for f in top(s, k, 3)))
pd.DataFrame(rows).to_csv(OUT / "qualitative_cases_ground.csv", index=False)
(OUT / "qualitative_cases_ground.txt").write_text("\n".join(txt) + "\n")
print("\n".join(txt))

# ---- reader error analysis (TEAR+window, first answer before the date cut)
DATE = re.compile(r"(\d{4})(?:-(\d{2}))?(?:-(\d{2}))?")


def is_list(pred):
    m = re.search(r"\[.*?\]", pred.strip(), re.S)
    if not m:
        return False
    try:
        return isinstance(ast.literal_eval(m.group(0)), (list, tuple))
    except Exception:
        return False


cnt = {"errors_gold_in_top15": 0, "empty_or_malformed": 0, "wrong_granularity": 0, "wrong_granularity_finer": 0,
       "wrong_granularity_coarser": 0, "copied_wrong_fact": 0, "copied_from_top1": 0, "copied_topic_entity": 0,
       "other_not_in_top15": 0, "all_reader_errors": 0}
for k in sample:
    if ok("tear_window", k, "reader"):
        continue
    cnt["all_reader_errors"] += 1
    if gold_rank("tear_window", k) is None:
        continue
    cnt["errors_gold_in_top15"] += 1
    q = Q[k]; pred = PRED["tear_window"][k]; ans = first_answer("tear_window", k)
    if not ans.strip() or not is_list(pred):
        cnt["empty_or_malformed"] += 1
        continue
    if q["answer_type"] == "time":
        m = DATE.search(ans)
        if m:
            a = m.group(0)
            golds = [g for g in q["answers"] if DATE.fullmatch(g)]
            fin = any(a != g and a.startswith(g) for g in golds)
            coa = any(a != g and g.startswith(a) for g in golds)
            if fin or coa:
                cnt["wrong_granularity"] += 1
                cnt["wrong_granularity_finer" if fin else "wrong_granularity_coarser"] += 1
                continue
    na = normalize(ans)
    tops = top("tear_window", k)
    hit = [i for i, fid in enumerate(tops) if na and any(na == normalize(x) or (q["answer_type"] == "time" and normalize(x).startswith(na))
                                                     for x in (fact(fid)[0], fact(fid)[2], fact(fid)[3]))]
    if hit:
        cnt["copied_wrong_fact"] += 1
        cnt["copied_from_top1"] += int(0 in hit)
        cnt["copied_topic_entity"] += int(na == normalize(SL[k].get("topic_entity") or "#"))
        continue
    cnt["other_not_in_top15"] += 1
n_sample = len(sample)
rows = [{"category": c, "count": v,
         "share_of_errors_gold_in_top15": (v / cnt["errors_gold_in_top15"]) if c != "all_reader_errors" else np.nan,
         "n_sample": n_sample} for c, v in cnt.items()]
pd.DataFrame(rows).to_csv(OUT / "reader_errors_ground.csv", index=False)
print(pd.DataFrame(rows).round(4).to_string(index=False))
