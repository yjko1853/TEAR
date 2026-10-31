"""Hit@1 scoring.

LLM-free Hit@1: the answer is read directly off the top-1 reranked fact - the value in the (rule-predicted) answer
slot, or the fact date cut to the granularity the question asks for ('year' -> YYYY, 'month' -> YYYY-MM, otherwise
YYYY-MM-DD; read from the question text, not from the gold time_level field).
Reader Hit@1: first item of the LLM's answer list; exact (normalised equality) and lenient (TEAR/TimeR4 substring
rule); hit_any_lenient = TimeR4-style "Hit" over the whole generated list.
"""
from __future__ import annotations

import re

import pandas as pd

from ..data.multitq import stage_query
from ..reader.parse import parse_list
from ..text import exact_match, lenient_match, match


def answer_granularity(question: str) -> int:
    ql = question.lower()
    if re.search(r'\byear\b', ql):
        return 4
    if re.search(r'\bmonth\b', ql):
        return 7
    return 10


def read_answer_from_fact(slot: str, fid: int, facts, ents, tss, question: str) -> str:
    if slot == "timestamp":
        return tss[int(facts["t_id"][fid])][:answer_granularity(question)]
    if slot == "object":
        return ents[int(facts["o_id"][fid])]
    return ents[int(facts["s_id"][fid])]


def llmfree_rows(top1, keys, qby, slots, facts, ents, tss, reranker: str):
    """One row per question: (reranker, qtype, qlabel, answer_type, exact, lenient)."""
    rows = []
    for i, k in enumerate(keys):
        q = qby[k]
        slot = slots.get(k, {}).get("answer_slot", "subject")
        pred = read_answer_from_fact(slot, int(top1[i]), facts, ents, tss, stage_query(q))
        rows.append((reranker, q["qtype"], q["qlabel"], q["answer_type"],
                     exact_match(pred, q["answers"]), lenient_match(pred, q["answers"])))
    return rows


def summarize_llmfree(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for rk, g in df.groupby("reranker", sort=False):
        rec = {"reranker": rk, "n": len(g), "exact_all": g.exact.mean(), "lenient_all": g.lenient.mean()}
        for col in ("qtype", "qlabel", "answer_type"):
            for v, gg in g.groupby(col):
                rec[f"exact_{v}"] = gg.exact.mean()
        out.append(rec)
    return pd.DataFrame(out)


def score_reader_record(r: dict) -> dict:
    lst = parse_list(r["prediction"])
    first = lst[0] if lst else ""
    return {"hit1_exact": exact_match(first, r["answers"]),
            "hit1_lenient": lenient_match(first, r["answers"]),
            "hit_any_lenient": any(match(" ".join(lst), a) for a in r["answers"])}
