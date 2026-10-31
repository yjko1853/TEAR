"""Event-anchored temporal constraints: the reference event's date is looked up in the KG (no training, no LLM).

MultiTQ multi-constraint questions often fix the time by another event instead of a date:
  'After the Danish Ministry of Defence and Security, who was the first to visit Iraq?'
  'Who last would wish to visit Thailand before the Charge D'affaires of United States did?'
  'Who made optimistic remarks about China in the same month as Islam Karimov?'
The anchor X takes the answer's role in the same kind of event with the same partner, the question's topic entity:
the reference event is a fact (X, r, topic, t) when the answer slot is the subject and (topic, r, X, t) when it is the
object. TimeR4 resolves it by retrieval plus gpt-3.5 rewriting ('After 2006-01-05, ...'); here it is a KG lookup:
  1. anchor phrase: leading 'Before/After X,', trailing 'before/after X [did]?', or 'in/on the same day/month/year
     as/of X'; a phrase that starts with a date or month name is left to the date parser.
  2. anchor entity: KG entities named in the phrase (rules.build_mentioner / build_sector_matcher), restricted to the
     entities that share a fact with the topic entity in the question's slot layout; if none is named, the most similar
     of those entities by name embedding (all-mpnet-base-v2 cosine >= tau), which catches descriptions such as
     'the leader of Pakistan' -> 'Head of Government (Pakistan)'.
  3. reference event: among the facts that link the anchor and the topic in that layout, the one closest to the question
     (question embedding . fact embedding, the retriever's own score) fixes the anchor entity and the relation.
  4. date: when that (anchor, relation, topic) event happened on several days, the widest window is kept: the earliest
     day for 'after' and same-period questions, the latest for 'before'.
The choices (tau = 0.5, relation by fact score, the widest-window date rule, and the embedding fallback even when the
phrase names only the topic entity, guard_topic=False) were fixed on a sample of the MultiTQ dev split
(scripts/14_ground_anchors.py --dev_design); the test split was not used for them.
"""
from __future__ import annotations

import re

import numpy as np

from ..temporal import days_to_date
from .rules import build_mentioner, build_sector_matcher

_MON = (r"(?:january|february|march|april|may|june|july|august|september|october|november|december|"
        r"jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)")
_NOT_DATE = r"(?!\d|the\s+year\s+\d|" + _MON + r"\b)"
SAME = re.compile(r"\b(?:in|on|during|at)\s+the\s+same\s+(?P<g>day|month|year)\s+(?:as|of|that)\s+" + _NOT_DATE +
                  r"(?P<x>[^,?]+?)(?:\s+did)?\s*(?:,|\?|$)", re.I)
LEAD = re.compile(r"^\s*(?P<c>before|after|prior to|following)\s+" + _NOT_DATE + r"(?P<x>[^,?]+?)\s*,", re.I)
TRAIL = re.compile(r"\b(?P<c>before|after|prior to|following)\s+" + _NOT_DATE +
                   r"(?P<x>[^,?]+?)(?:\s+(?:did|does|had|was|were))?\s*\??\s*$", re.I)
DATE_RULES = ("widest", "earliest", "latest", "best")


def anchor_phrase(question: str) -> tuple[str, str] | None:
    """(type, phrase) of an event-anchored constraint; type is before / after / day / month / year."""
    m = SAME.search(question)
    if m:
        return m.group("g").lower(), m.group("x").strip()
    for rx in (LEAD, TRAIL):
        m = rx.search(question)
        if m:
            return ("before" if m.group("c").lower() in ("before", "prior to") else "after"), m.group("x").strip()
    return None


def pick_date(days: np.ndarray, typ: str, best_day: int, rule: str = "widest") -> int:
    """Reference day among the days of one (anchor, relation, topic) event."""
    if rule == "best":
        return int(best_day)
    if rule == "earliest" or (rule == "widest" and typ != "before"):
        return int(days.min())
    return int(days.max())


class AnchorResolver:
    """KG lookup of the reference event. facts: npz arrays s_id, r_id, o_id, t_days; entity_emb: (E, d) unit vectors
    of the entity names; fact_emb: (F, d) unit vectors of the fact sentences (may be a memory map)."""

    def __init__(self, facts, entities, relations, entity_emb=None, fact_emb=None, tau: float = 0.5,
                 date_rule: str = "widest", guard_topic: bool = False):
        self.S, self.R, self.O = facts["s_id"], facts["r_id"], facts["o_id"]
        self.TD = facts["t_days"].astype(np.int64)
        self.ents, self.rels = entities, relations
        self.eid = {e: i for i, e in enumerate(entities)}
        self.entity_emb, self.fact_emb = entity_emb, fact_emb
        self.tau, self.date_rule, self.guard_topic = tau, date_rule, guard_topic
        self.mentions, self.sector = build_mentioner(entities), build_sector_matcher(entities)
        self.order = {"s": np.argsort(self.S, kind="stable"), "o": np.argsort(self.O, kind="stable")}
        self.sorted = {"s": self.S[self.order["s"]], "o": self.O[self.order["o"]]}

    def facts_with(self, side: str, e: int) -> np.ndarray:
        lo, hi = np.searchsorted(self.sorted[side], e, "left"), np.searchsorted(self.sorted[side], e, "right")
        return np.sort(self.order[side][lo:hi])

    def layout(self, slot: str, t: int):
        """Facts with the topic in the non-answer position, and the array holding the answer-side entity."""
        if slot == "subject":
            return self.facts_with("o", t), self.S
        return self.facts_with("s", t), self.O

    def resolve(self, typ: str, phrase: str, slot: str, topic: str, q_vec=None, x_vec=None) -> dict:
        out = {"type": typ, "phrase": phrase, "status": "", "anchor": None, "anchor_source": None,
               "relation": None, "date": None, "n_dates": 0}
        if slot not in ("subject", "object") or topic not in self.eid:
            out["status"] = "no_topic"
            return out
        t = self.eid[topic]
        pool, other = self.layout(slot, t)
        partners = np.unique(other[pool])
        partners = partners[partners != t]
        named = {self.eid[e] for e in set(self.mentions(phrase)) | set(self.sector(phrase)) if e in self.eid}
        xm = named & set(partners.tolist())
        src = "mention"
        if not xm:
            if self.guard_topic and t in named and len(named) == 1:
                out["status"] = "anchor_is_topic"
                return out
            if x_vec is not None and self.entity_emb is not None and len(partners) and self.tau < 1:
                sims = self.entity_emb[partners] @ x_vec
                j = int(np.argmax(sims))
                if sims[j] >= self.tau:
                    xm, src = {int(partners[j])}, "embedding"
        if not xm:
            out["status"] = "no_anchor_entity"
            return out
        cand = pool[np.isin(other[pool], list(xm))]
        if q_vec is not None and self.fact_emb is not None:
            best = int(cand[int(np.argmax(np.asarray(self.fact_emb[cand]) @ q_vec))])
        else:
            best = int(cand[0])
        same = cand[(self.R[cand] == self.R[best]) & (other[cand] == other[best])]
        days = np.unique(self.TD[same])
        out.update(status="resolved", anchor=self.ents[int(other[best])], anchor_source=src,
                   relation=self.rels[int(self.R[best])], n_dates=int(len(days)),
                   date=days_to_date(pick_date(days, typ, self.TD[best], self.date_rule)))
        return out

    def implied_answers(self, slot: str, topic: str, relation: str, typ: str, ref_day: int, cues) -> set:
        """Answers the KG gives under the resolved constraint (design check against gold answers): the first / last
        partner after / before ref_day, or every partner in the same period; same relation and topic layout."""
        from ..temporal import period_bounds
        t, r = self.eid[topic], self.rels.index(relation)
        pool, other = self.layout(slot, t)
        allf = pool[self.R[pool] == r]
        if typ in ("before", "after"):
            sel = allf[self.TD[allf] > ref_day] if typ == "after" else allf[self.TD[allf] < ref_day]
            if not len(sel):
                return set()
            first = "first" in cues or ("last" not in cues and typ == "after")
            tsel = self.TD[sel].min() if first else self.TD[sel].max()
            return {self.ents[int(other[k])] for k in sel[self.TD[sel] == tsel]}
        lo, hi = period_bounds(ref_day, typ)
        sel = allf[(self.TD[allf] >= lo) & (self.TD[allf] <= hi)]
        return {self.ents[int(other[k])] for k in sel}
