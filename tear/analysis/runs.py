"""Per-question metrics of a finished reranking run, for paired comparisons between runs.

A run is (dataset, question file, rule labels, tag, rerank directory). Answer coverage@15 and LLM-free Hit@1 (exact)
are recomputed from the saved top-30 arrays and can be checked against the stage 05/06 CSVs.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..evaluation.coverage import AnswerMasks
from ..evaluation.hits import llmfree_rows
from ..paths import CACHE, RESULTS
from ..utils.io import load_json

RERANKERS = ["none", "timer4", "hardwindow", "atc", "tr", "tear", "tear_window"]


def stage_text(q: dict) -> str:
    """The question text every stage reads: 'query' (questions_v2.json) when present, otherwise 'rewrite'."""
    return q["query"] if "query" in q else q["rewrite"]


_MASKS: dict = {}


def coverage15(top15, qs, facts, voc):
    """Per-question answer coverage@15 (TEAR match rule), as in tear.evaluation.coverage.answer_hit_features."""
    key = (tuple(voc["entities"]), tuple(voc["relations"]), tuple(voc["timestamps"]))
    if key not in _MASKS:                                  # the per-answer mask cache is shared across runs
        _MASKS[key] = AnswerMasks(voc["entities"], voc["relations"], voc["timestamps"])
    sm = _MASKS[key]
    sizes = (len(voc["entities"]), len(voc["relations"]), len(voc["timestamps"]))
    s, r, o, t = facts["s_id"], facts["r_id"], facts["o_id"], facts["t_id"]
    out = np.zeros(len(qs), bool)
    for i, q in enumerate(qs):
        em, rm, tm = sm.union(q["answers"], sizes)
        c = top15[i]
        out[i] = (em[s[c]] | em[o[c]] | rm[r[c]] | tm[t[c]]).any()
    return out


def load_run(ds, tag, questions="questions.json", slots_file="slots_rule.json", rerank_dir=None, rerankers=RERANKERS):
    """(question keys, questions in that order with 'rewrite' = stage text, {reranker: {top, cov15, hit1}})."""
    d = CACHE / ds
    rdir = Path(rerank_dir) if rerank_dir else d / "rerank"
    qs = [dict(q, rewrite=stage_text(q)) for q in load_json(d / questions)]
    order = [str(k) for k in load_json(rdir / "question_order.json")["question_ids"]]
    qby = {str(q["quid"]): q for q in qs}
    qs = [qby[k] for k in order]
    slots = load_json(d / slots_file)
    facts = np.load(d / "facts.npz"); voc = load_json(d / "vocab.json")
    per = {}
    for rk in rerankers:
        top = np.load(rdir / f"{rk}_{tag}_top30_ids.npy")
        hit = pd.DataFrame(llmfree_rows(top[:, 0], order, qby, slots, facts, voc["entities"], voc["timestamps"], rk),
                           columns=["reranker", "qtype", "qlabel", "answer_type", "exact", "lenient"])
        per[rk] = {"top": top, "cov15": coverage15(top[:, :15], qs, facts, voc), "hit1": hit.exact.values.astype(bool)}
    return order, qs, per


def check_against_csv(ds, tag, per, results=RESULTS):
    """Assert that the recomputed overall coverage@15 and Hit@1 equal the stage 05 / 06 CSVs."""
    cov = pd.read_csv(results / f"{ds}_coverage_main_{tag}.csv")
    hit = pd.read_csv(results / f"{ds}_llmfree_hit1_{tag}.csv")
    for rk, v in per.items():
        c = cov[(cov.reranker == rk) & (cov.k == 15) & (cov.group == "all")].coverage.item()
        h = hit[hit.reranker == rk].exact_all.item()
        assert abs(c - v["cov15"].mean()) < 1e-9 and abs(h - v["hit1"].mean()) < 1e-9, (ds, tag, rk, c, h)
