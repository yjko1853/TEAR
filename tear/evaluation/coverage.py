"""Retrieval-side metrics on the candidate pool.

answer coverage@k: share of questions with at least one top-k fact whose subject / relation / object / timestamp
(TimelineKGQA: start or end) matches a gold answer under TEAR's `match` rule.
gold-event recall@k (TimelineKGQA): all gold events in the top-k ("all", the dataset's strict Hits definition) or at
least one ("any").
"""
from __future__ import annotations

import numpy as np

from ..text import normalize

KS = [5, 10, 15, 20, 30]


class AnswerMasks:
    """answer string -> boolean masks over the entity / relation / timestamp vocabularies (TEAR match rule)."""

    def __init__(self, ents, rels, tss):
        self.en = [normalize(e) for e in ents]
        self.rn = [normalize(r) for r in rels]
        self.tn = [normalize(t) for t in tss]
        self.cache = {}

    def __call__(self, ans):
        a = normalize(ans)
        if a not in self.cache:
            if not a:
                self.cache[a] = tuple(np.zeros(len(x), bool) for x in (self.en, self.rn, self.tn))
            else:
                self.cache[a] = tuple(np.fromiter((a in x for x in L), bool, len(L)) for L in (self.en, self.rn, self.tn))
        return self.cache[a]

    def union(self, answers, sizes):
        em = np.zeros(sizes[0], bool); rm = np.zeros(sizes[1], bool); tm = np.zeros(sizes[2], bool)
        for a in answers:
            e_, r_, t_ = self(a); em |= e_; rm |= r_; tm |= t_
        return em, rm, tm


def answer_hit_features(pool) -> dict:
    """hit (Q, N): candidate contains a gold answer; cov_defined (Q,); gold_hit (Q, N) and n_gold (Q,) for
    datasets with gold events."""
    Q, N = pool.ids.shape
    hit = np.zeros((Q, N), bool)
    cov_defined = np.zeros(Q, bool)
    gold_hit = np.zeros((Q, N), bool)
    n_gold = np.zeros(Q, np.int32)
    sm = AnswerMasks(pool.ents, pool.rels, pool.tss)
    sizes = (pool.E, pool.R, len(pool.tss))
    for qi, q in enumerate(pool.qs):
        cand = pool.ids[qi]
        answers, defined = pool.answers(q)
        cov_defined[qi] = defined
        if defined:
            em, rm, tm = sm.union(answers, sizes)
            hit[qi] = (em[pool.s[cand]] | em[pool.o[cand]] | rm[pool.r[cand]] | tm[pool.ts_start[cand]]
                       | tm[pool.ts_end[cand]])
        if "gold_fact_ids" in q:
            g = [x for x in q["gold_fact_ids"] if x >= 0]
            n_gold[qi] = len(g)
            if g:
                gold_hit[qi] = np.isin(cand, np.array(g))
    return dict(hit=hit, cov_defined=cov_defined, gold_hit=gold_hit, n_gold=n_gold)


def coverage_rows(F, perm, name, groups, ks=KS, extra=None):
    """Long-format rows: reranker, k, group_type, group, n, n_cov_defined, coverage (+ gold recall)."""
    rows = []
    hit = F["hit"]; defined = F["cov_defined"]
    gh = F["gold_hit"]; ng = F["n_gold"]
    for k in ks:
        top = perm[:, :k]
        cov = np.take_along_axis(hit, top, axis=1).any(1)
        gtop = np.take_along_axis(gh, top, axis=1).sum(1)
        rec_all = (gtop == ng) & (ng > 0)
        rec_any = gtop > 0
        for gtype, gser in groups.items():
            for gval in sorted(gser.unique()):
                m = (gser == gval).values
                md = m & defined
                row = {"reranker": name, "k": k, "group_type": gtype, "group": gval, "n": int(m.sum()),
                       "n_cov_defined": int(md.sum()), "coverage": float(cov[md].mean()) if md.any() else np.nan}
                if ng.any():
                    mg = m & (ng > 0)
                    row["gold_recall_all"] = float(rec_all[mg].mean()) if mg.any() else np.nan
                    row["gold_recall_any"] = float(rec_any[mg].mean()) if mg.any() else np.nan
                if extra:
                    row.update(extra)
                rows.append(row)
    return rows


def timer4_public_coverage(pool) -> np.ndarray:
    """MultiTQ: coverage@15 of TimeR4's released top-15 (fine-tuned retriever); unmapped strings matched as text."""
    sm = AnswerMasks(pool.ents, pool.rels, pool.tss)
    sizes = (pool.E, pool.R, len(pool.tss))
    covs = []
    for q in pool.qs:
        em, rm, tm = sm.union(q["answers"], sizes)
        hit = False
        for fid, raw in zip(q["timer4_top15_ids"], q["timer4_top15_raw"]):
            if fid >= 0:
                if em[pool.s[fid]] or em[pool.o[fid]] or rm[pool.r[fid]] or tm[pool.ts_start[fid]]:
                    hit = True; break
            else:
                if any(normalize(a) in normalize(raw) for a in q["answers"]):
                    hit = True; break
        covs.append(hit)
    return np.array(covs)
