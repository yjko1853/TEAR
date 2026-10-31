"""The shared candidate pool: questions + fact table + the retriever's top-N candidates per question.

Every reranker and every metric reads the same pool, so differences between rerankers are not confounded by the
retriever.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import is_multitq
from ..paths import CACHE
from ..temporal import days_to_date
from ..utils.io import load_json


class CandidatePool:
    def __init__(self, ds: str, topn: int = 1000, cache_dir: Path | None = None,
                 questions_file: str = "questions.json", suffix: str = ""):
        """questions_file / suffix select a question version, e.g. questions_v2.json with top{N}_v2_{ids,scores}.npy."""
        d = Path(cache_dir) if cache_dir else CACHE / ds
        self.ds, self.d = ds, d
        f = np.load(d / "facts.npz")
        voc = load_json(d / "vocab.json")
        self.qs = load_json(d / questions_file)
        self.ids = np.load(d / f"top{topn}{suffix}_ids.npy")
        self.sims = np.load(d / f"top{topn}{suffix}_scores.npy").astype(np.float32)
        self.N = self.ids.shape[1]
        self.ents = voc["entities"]
        with open(d / "sentences.txt", encoding="utf-8") as fh:
            self.sent_lower = [line.rstrip("\n").lower() for line in fh]
        if is_multitq(ds):
            self.rels = voc["relations"]
            self.tss = voc["timestamps"]
            self.s, self.r, self.o = f["s_id"], f["r_id"], f["o_id"]
            self.ts_start = self.ts_end = f["t_id"]
            self.date = self.start = self.end = f["t_days"].astype(np.int64)
            self.qkey = "quid"
            # cues/dates come from the stage text: 'query' (v2 questions) or the TimeR4 rewrite (v1, multitq_orig)
            self.cue_field = "query" if self.qs and "query" in self.qs[0] else "rewrite"
            self.interval = False
        else:
            self.rels = voc["predicates"]
            self.s, self.r, self.o = f["s_id"], f["p_id"], f["o_id"]
            self.start = f["start_days"].astype(np.int64)
            self.end = f["end_days"].astype(np.int64)
            self.date = self.start
            uniq = np.unique(np.concatenate([self.start, self.end]))
            self.tss = [days_to_date(int(x)) for x in uniq]
            pos = {int(x): i for i, x in enumerate(uniq)}
            self.ts_start = np.array([pos[int(x)] for x in self.start], np.int32)
            self.ts_end = np.array([pos[int(x)] for x in self.end], np.int32)
            self.qkey = "qid"
            self.cue_field = "question"
            self.interval = True
        self.E, self.R = len(self.ents), len(self.rels)

    def key(self, q) -> str:
        return str(q[self.qkey])

    def answers(self, q):
        """(gold answer strings, whether answer coverage is defined for this question)."""
        if is_multitq(self.ds):
            return q["answers"], True
        at = q["answer_type"]
        return [q["answer"]], at in ("subject", "object", "timestamp_start", "timestamp_end")
