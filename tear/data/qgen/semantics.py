"""The answer procedure of each MultiTQ question type, executed over a TemporalKG.

Recovered from the released questions (align.py reproduces the gold answers of MultiTQ dev questions with these
procedures; the shares are in the addendum): answers are computed on the whole KG, before/after compare strictly,
an anchor event is dated by its *earliest* occurrence, the anchor itself is not an answer of a same-period question.

  qtype         variant    question shape (answer side = subject unless noted)          answers
  equal         period     Who visited {O} in {T}?                                      partners with a fact in T
  equal         unique     When did {S} visit {O}?  (answer type time)                  the one period of (S, r, O)
  before_after  date       Who visited {O} before {T}?                                  partners with a fact < T / > T
  before_after  anchor     Who visited {O} before {Y}?                                  partners with a fact < / > t0
  first_last    extreme    Who first visited {O}? / When did {S} first visit {O}?       partners on the first/last day
  equal_multi   same       Who visited {O} in the same month as {Y}?                    partners in period(t0), minus Y
  equal_multi   period_fl  In {T}, who first visited {O}?                               first/last day within T
  after_first   anchor     After {Y}, who first visited {O}?                            first day after t0
  before_last   anchor     Who visited {O} last before {Y} did?                         last day before t0
  after_first / before_last date: the same with a date instead of {Y} (part of MultiTQ's design; rare in the data)
t0 = earliest day of the anchor event (Y, r, O) (answer side object: (O, r, Y)). Entity answers are sorted by their
first fact day in the selection; time answers are written at the question's granularity ('2008', '2008-06',
'2008-06-20').
"""
from __future__ import annotations

import numpy as np

from .kg import EventList, TemporalKG
from .timeexpr import period_label, period_of

ENTITY_VARIANTS = {"equal": ("period",), "before_after": ("date", "anchor"), "first_last": ("extreme",),
                   "equal_multi": ("same", "period_fl"), "after_first": ("anchor", "date"),
                   "before_last": ("anchor", "date")}
TIME_VARIANTS = {"equal": ("unique",), "first_last": ("extreme",)}
QLABEL = {"equal": "Single", "before_after": "Single", "first_last": "Single", "equal_multi": "Multiple",
          "after_first": "Multiple", "before_last": "Multiple"}
NEEDS_ANCHOR = {("before_after", "anchor"), ("equal_multi", "same"), ("after_first", "anchor"),
                ("before_last", "anchor")}
NEEDS_TIME = {("equal", "period"), ("before_after", "date"), ("equal_multi", "period_fl"), ("after_first", "date"),
              ("before_last", "date")}


def _ordered(partners: np.ndarray, fids: np.ndarray, exclude=None):
    """Unique partners in order of first appearance (events are day-sorted) and the fact ids that support them."""
    seen, out, sup = set(), [], []
    for p, f in zip(partners.tolist(), fids.tolist()):
        if exclude is not None and p == exclude:
            continue
        sup.append(f)
        if p not in seen:
            seen.add(p); out.append(p)
    return out, sup


def _extreme(ev: EventList, sl: slice | np.ndarray, first: bool, exclude=None):
    days = ev.days[sl]
    if exclude is not None:
        keep = ev.partners[sl] != exclude
        days = days[keep]
    if not len(days):
        return [], []
    t = days.min() if first else days.max()
    m = (ev.days == t)
    if exclude is not None:
        m &= ev.partners != exclude
    return _ordered(ev.partners[m], ev.fids[m])


def entity_answers(ev: EventList, qtype: str, variant: str, *, first: bool | None = None, before: bool | None = None,
                   lo: int | None = None, hi: int | None = None, anchor: int | None = None, gran: str = "day",
                   exclude_anchor_after: bool = False):
    """(answer entity ids, supporting fact ids, info) for one event list. info has t0 for anchored variants
    (None when the anchor has no event in the list)."""
    info = {}
    if variant in ("anchor", "same"):
        t0 = ev.first_day_of(anchor)
        info["t0"] = t0
        if t0 is None:
            return [], [], info
        m = (ev.partners == anchor) & (ev.days == t0)
        info["anchor_fids"] = ev.fids[m].tolist()
        info["anchor_n_days"] = int(len(np.unique(ev.days[ev.partners == anchor])))
    if qtype == "equal" and variant == "period":
        sl = ev.in_range(lo, hi)
        a, s = _ordered(ev.partners[sl], ev.fids[sl])
        return a, s, info
    if qtype == "before_after":
        if variant == "date":
            m = ev.days < lo if before else ev.days > hi
            a, s = _ordered(ev.partners[m], ev.fids[m])
        else:
            m = ev.days < info["t0"] if before else ev.days > info["t0"]
            a, s = _ordered(ev.partners[m], ev.fids[m], exclude=anchor)
        return a, s, info
    if qtype == "first_last":
        a, s = _extreme(ev, slice(None), first)
        return a, s, info
    if qtype == "equal_multi":
        if variant == "same":
            plo, phi = period_of(info["t0"], gran)
            sl = ev.in_range(plo, phi)
            a, s = _ordered(ev.partners[sl], ev.fids[sl], exclude=anchor)
            info["period"] = (plo, phi)
        else:
            sl = ev.in_range(lo, hi)
            a, s = _extreme(ev, sl, first) if sl.stop > sl.start else ([], [])
        return a, s, info
    if qtype in ("after_first", "before_last"):
        aft = qtype == "after_first"
        ref = info["t0"] if variant == "anchor" else (hi if aft else lo)
        m = ev.days > ref if aft else ev.days < ref
        if not m.any():
            return [], [], info
        t = ev.days[m].min() if aft else ev.days[m].max()
        sel = ev.days == t
        if exclude_anchor_after and variant == "anchor":
            sel &= ev.partners != anchor
        a, s = _ordered(ev.partners[sel], ev.fids[sel])
        return a, s, info
    raise ValueError((qtype, variant))


def time_answers(days: np.ndarray, qtype: str, variant: str, gran: str, first: bool | None = None):
    """Time answers of (S, r, O) with sorted days: 'unique' -> the period label when all days fall in one period
    (otherwise []), 'extreme' -> the period of the first / last day."""
    if not len(days):
        return []
    if qtype == "equal" and variant == "unique":
        labels = sorted({period_label(int(d), gran) for d in np.unique(days)})
        return labels if len(labels) == 1 else []
    if qtype == "first_last" and variant == "extreme":
        return [period_label(int(days[0] if first else days[-1]), gran)]
    raise ValueError((qtype, variant))


def is_pseudo_temporal(ev: EventList, answers: list) -> bool:
    """The time constraint does not change the answer: the answers are every partner of the event list."""
    return set(answers) == ev.partner_set()


def evaluate(kg: TemporalKG, st: dict) -> dict:
    """Answers of a question structure st = {qtype, variant, answer_type, side, relation, topic, [other], [anchor],
    [lo, hi], [first], [before], gran}. Returns {answers: [names or labels], fact_ids, t0, ...}."""
    r = kg.rid[st["relation"]] if isinstance(st["relation"], str) else st["relation"]
    if st["answer_type"] == "time":
        days = kg.sro_days(st["subject"], r, st["object"])
        ans = time_answers(days, st["qtype"], st["variant"], st["gran"], st.get("first"))
        return {"answers": ans, "n_periods": len({period_label(int(d), st["gran"]) for d in np.unique(days)}),
                "n_days": int(len(np.unique(days)))}
    ev = kg.events(st["side"], r, st["topic"])
    a, sup, info = entity_answers(ev, st["qtype"], st["variant"], first=st.get("first"), before=st.get("before"),
                                  lo=st.get("lo"), hi=st.get("hi"), anchor=st.get("anchor"), gran=st["gran"])
    info.update(answers=[kg.name(e) for e in a], answer_ids=a, fact_ids=sup, n_partners=len(ev.partner_set()),
                pseudo=is_pseudo_temporal(ev, a) if a else False)
    return info
