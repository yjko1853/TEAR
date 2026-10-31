"""Event-anchored interval constraints for TimelineKGQA (ICEWS-Actor, CronQuestions-KG): the reference event is looked
up in the KG and candidates are kept only if their interval has the asked Allen relation to it (no training, no LLM).

TimelineKGQA medium / complex questions fix the time of the asked event X by one or two other events Y, written out in
full: 'Who Affiliation To Far Left while Takis Kliridis Affiliation To Ministry of Finance?'. Facts are intervals
[start, end] (open ends are +-1e8 days). Each constraint is a connective followed by an anchor clause:
  connective -> a set of Allen relations of X to Y. The wording only fixes a family (validation split: 'during' /
               'while' / 'in the course of' are used for d, di, o and oi; 'at the same time' for =, s, si, f, fi and a
               few overlaps), so the family is kept as a set; or a day offset ('N days after Y': X.start - Y.end = N;
               'N days before Y': Y.start - X.end = N); or the relation printed in the question by the generator
               ("{'relation': 'X < Y', ...}", kept in the released text).
  anchor clause -> the KG fact whose subject and object (Cron: and predicate, when named) are both mentioned in it,
               taking the earliest-mentioned subject; when that (s, p, o) has several intervals, a candidate is valid
               if it relates to any of them (validation: the gold answer event passes the window for 96.9% / 87.7% of
               ICEWS / Cron questions this way, 88.1% / 76.8% when only the best-scored interval is kept).
The connective lexicon, the Allen definitions and the offset arithmetic were checked against the gold events of the
validation split (scripts/16_timeline_grounding.py --dev_design); the test split is only evaluated.
"""
from __future__ import annotations

import re

import numpy as np

from ..temporal import days_to_date

_DICT = r"\{[^{}]*'relation':\s*'(X [^']+ Y)'[^{}]*\}"
CONNECTIVES = [   # (kind, regex); earlier patterns claim their span first
    ("days_dict", r"(\d+) days " + _DICT),
    ("dict", _DICT),
    ("days_before", r"\b(\d+) days before\b"),
    ("days_after", r"\b(\d+) days after\b"),
    ("same_start_end", r"\bat the same start and end time\b"),
    ("same_time", r"\bat the same time\b"),
    ("during", r"\b(?:during|while|in the course of|the period when|in the midst of)\b"),
    ("before", r"\b(?:before|prior to|in advance of|earlier than|early than|preceding|ahead of)\b"),
    ("after", r"\b(?:after|later than|following)\b"),
    ("equal", r"\bequal\b"), ("finishedby", r"\bfinishedby\b"), ("finishes", r"\bfinishes\b"),
    ("startedby", r"\bstartedby\b"), ("starts", r"\bstarts\b"), ("metby", r"\bmetby\b"), ("meets", r"\bmeets\b"),
]
_OVERLAP = {"=", "s", "si", "f", "fi", "d", "di", "o", "oi"}
FAMILIES = {"before": {"<"}, "after": {">"}, "during": {"d", "di", "o", "oi"}, "same_time": _OVERLAP,
            "same_start_end": {"="}, "equal": {"="}, "finishes": {"f"}, "finishedby": {"fi"}, "starts": {"s"},
            "startedby": {"si"}, "metby": {"mi"}, "meets": {"m"}}
ALLEN = ("<", ">", "=", "m", "mi", "s", "si", "f", "fi", "d", "di", "o", "oi")


def allen_mask(rel: str, xs, xe, ys, ye):
    """Candidates (xs, xe arrays) standing in Allen relation `rel` to the interval [ys, ye] (strict definitions)."""
    if rel == "<":
        return xe < ys
    if rel == ">":
        return xs > ye
    if rel == "=":
        return (xs == ys) & (xe == ye)
    if rel == "m":
        return xe == ys
    if rel == "mi":
        return xs == ye
    if rel == "s":
        return (xs == ys) & (xe < ye)
    if rel == "si":
        return (xs == ys) & (xe > ye)
    if rel == "f":
        return (xe == ye) & (xs > ys)
    if rel == "fi":
        return (xe == ye) & (xs < ys)
    if rel == "d":
        return (xs > ys) & (xe < ye)
    if rel == "di":
        return (xs < ys) & (xe > ye)
    if rel == "o":
        return (xs < ys) & (ys < xe) & (xe < ye)
    if rel == "oi":
        return (ys < xs) & (xs < ye) & (ye < xe)
    raise ValueError(rel)


def allen_of(xs: int, xe: int, ys: int, ye: int) -> str:
    """The single Allen relation of [xs, xe] to [ys, ye]."""
    return next(r for r in ALLEN if allen_mask(r, np.array([xs]), np.array([xe]), ys, ye)[0])


def parse_constraints(question: str) -> list[dict]:
    """[{kind, relations (set) or None, offset ('before'|'after', days) or None, start, end, clause}] in text order;
    the clause of a connective runs to the next connective or the end of the question."""
    taken, found = [], []
    for kind, rx in CONNECTIVES:
        for m in re.finditer(rx, question, re.I):
            if any(m.start() < e and s < m.end() for s, e in taken):
                continue
            taken.append((m.start(), m.end()))
            rel, off = None, None
            if kind in ("days_dict", "dict"):
                r = m.group(2 if kind == "days_dict" else 1)
                code = r.split()[1]
                rel = {code} if code in ALLEN else None
                if kind == "days_dict" and code in ("<", ">"):
                    off = ("before" if code == "<" else "after", int(m.group(1)))
                    rel = None
            elif kind in ("days_before", "days_after"):
                off = (kind.split("_")[1], int(m.group(1)))
            else:
                rel = set(FAMILIES[kind])
            found.append({"kind": kind, "relations": rel, "offset": off, "start": m.start(), "end": m.end()})
    found.sort(key=lambda c: c["start"])
    for i, c in enumerate(found):
        stop = found[i + 1]["start"] if i + 1 < len(found) else len(question)
        c["clause"] = question[c["end"]:stop].strip(" ,?")
    return found


def _norm_tokens(s: str) -> list[str]:
    return re.sub(r"[^\w\s]", " ", s.lower()).split()


class MentionIndex:
    """Normalised n-gram lookup of KG names with token positions."""

    def __init__(self, names):
        self.ids = {}
        for i, n in enumerate(names):
            t = tuple(_norm_tokens(n))
            if t and len(" ".join(t)) >= 3:
                self.ids.setdefault(t, []).append(i)
        self.maxw = max(len(t) for t in self.ids)

    def find(self, text: str) -> list[tuple[int, int, int]]:
        """[(start token, end token, id)]."""
        w = _norm_tokens(text)
        out = []
        for i in range(len(w)):
            for j in range(i + 1, min(len(w), i + self.maxw) + 1):
                for e in self.ids.get(tuple(w[i:j]), ()):
                    out.append((i, j, e))
        return out


class IntervalAnchorResolver:
    """facts: npz arrays s_id, p_id, o_id, start_days, end_days; fact_emb (optional, rows = facts) for choosing among
    several intervals of the same (s, p, o)."""

    def __init__(self, facts, entities, predicates, fact_emb=None):
        self.S, self.P, self.O = facts["s_id"], facts["p_id"], facts["o_id"]
        self.ST = facts["start_days"].astype(np.int64)
        self.EN = facts["end_days"].astype(np.int64)
        self.ents, self.preds = entities, predicates
        self.fact_emb = fact_emb
        self.mentions = MentionIndex(entities)
        self.pred_tokens = [tuple(_norm_tokens(p)) for p in predicates]
        order = np.argsort(self.S, kind="stable")
        self._order, self._sorted = order, self.S[order]

    def facts_of_subject(self, e: int) -> np.ndarray:
        lo, hi = np.searchsorted(self._sorted, e, "left"), np.searchsorted(self._sorted, e, "right")
        return np.sort(self._order[lo:hi])

    def resolve_clause(self, clause: str) -> np.ndarray:
        """Fact ids of the anchor (s, p, o) mentioned in the clause (all its intervals); empty if none."""
        ms = self.mentions.find(clause)
        if not ms:
            return np.array([], np.int64)
        toks = _norm_tokens(clause)
        pos = {}
        for i, j, e in ms:
            pos.setdefault(e, []).append((i, j))
        best = None
        for i, j, e in sorted(ms, key=lambda m: (m[0], -(m[1] - m[0]))):
            fs = self.facts_of_subject(e)
            if not len(fs):
                continue
            later = {o for o, ps in pos.items() if any(a >= j for a, _ in ps)}
            fs = fs[np.isin(self.O[fs], list(later))]
            if not len(fs):
                continue
            # prefer a predicate named between the subject and its object (CronQuestions-KG has 203 predicates)
            scored = []
            for p in np.unique(self.P[fs]):
                pt = self.pred_tokens[p]
                named = any(tuple(toks[k:k + len(pt)]) == pt for k in range(j, len(toks) - len(pt) + 1))
                scored.append((not named, p))
            p = min(scored)[1]
            fs = fs[self.P[fs] == p]
            o_first = min(fs, key=lambda f: (min(a for a, _ in pos[int(self.O[f])] if a >= j), -len(self.ents[self.O[f]])))
            best = fs[(self.O[fs] == self.O[o_first])]
            break
        return best if best is not None else np.array([], np.int64)

    def resolve(self, question: str, q_vec=None, choose: str = "any") -> list[dict]:
        """Constraints of the question with their anchor facts ('anchors': chosen fact ids)."""
        cons = parse_constraints(question)
        for c in cons:
            fs = self.resolve_clause(c["clause"])
            if len(fs) > 1 and choose == "best" and q_vec is not None and self.fact_emb is not None:
                fs = fs[[int(np.argmax(np.asarray(self.fact_emb[fs]) @ q_vec))]]
            c["anchors"] = [int(f) for f in fs]
        return cons

    def window(self, cons: list[dict], xs, xe) -> np.ndarray | None:
        """Candidates satisfying every resolved constraint (any anchor interval per constraint); None if none
        resolved."""
        valid, used = np.ones(len(xs), bool), False
        for c in cons:
            if not c.get("anchors") or (c["relations"] is None and c["offset"] is None):
                continue
            m = np.zeros(len(xs), bool)
            for f in c["anchors"]:
                ys, ye = self.ST[f], self.EN[f]
                if c["offset"] is not None:
                    typ, n = c["offset"]
                    m |= (xs - ye == n) if typ == "after" else (ys - xe == n)
                else:
                    for r in c["relations"]:
                        m |= allen_mask(r, xs, xe, ys, ye)
            valid &= m
            used = True
        return valid if used else None

    def anchor_entities(self, cons: list[dict]) -> set:
        return {int(x) for c in cons for f in c.get("anchors", []) for x in (self.S[f], self.O[f])}


def read_answer(answer_type: str, fid: int, facts, entities) -> str | None:
    """LLM-free answer of a TimelineKGQA question from one fact: the answer-slot value (subject / object / start /
    end date); None for answer types that are not a slot value (duration, ranking, union, ...)."""
    if answer_type == "subject":
        return entities[int(facts["s_id"][fid])]
    if answer_type == "object":
        return entities[int(facts["o_id"][fid])]
    if answer_type == "timestamp_start":
        return days_to_date(int(facts["start_days"][fid]))
    if answer_type == "timestamp_end":
        return days_to_date(int(facts["end_days"][fid]))
    return None
