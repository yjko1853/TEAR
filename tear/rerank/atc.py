"""Answer-type classification (ATC) penalty masks.

A candidate fact is penalised (score x lambda1) when the value in the predicted answer slot is the question's topic
entity: such a fact cannot contain the answer in that slot (e.g. "Who visited Iraq?" with slot=subject penalises
facts whose subject is Iraq).

Two matching rules:
  exact      the slot value IS the topic entity (default for the paper; entity identity)
  substring  original TEAR rule (retrieval_temporal.basic_result): topic in slot value or slot value in topic, with a
             fallback on the whole fact string; it also penalises e.g. "Citizen (Iraq)" for topic "Iraq".
"""
from __future__ import annotations

import numpy as np

SLOTS = ("subject", "object", "relation", "timestamp")


def resolve_labels(key: str, slots: dict, topics_override: dict | None = None):
    """(answer_slot, topic_entity) for one question; unknown slots -> None."""
    rec = slots.get(key, {})
    slot = rec.get("answer_slot") or None
    topic = rec.get("topic_entity") or None
    if topics_override is not None:
        topic = (topics_override.get(key) or {}).get("topic_entity") or None
    if slot not in SLOTS:
        slot = None
    return slot, topic


def atc_mask_exact(slot, topic, s_c, o_c, r_c, ents, rels):
    """Boolean mask over candidates, or None when the slot has no entity/relation value (timestamp)."""
    t = topic.lower()
    if slot in ("subject", "object"):
        ids_slot = s_c if slot == "subject" else o_c
        return np.fromiter((ents[i].lower() == t for i in ids_slot), bool, len(ids_slot))
    if slot == "relation":
        return np.fromiter((rels[i].lower() == t for i in r_c), bool, len(r_c))
    return None


class ContainMasks:
    """topic (lower-cased) -> per-slot (topic in value, value in topic) masks over the vocabularies."""

    def __init__(self, ents, rels, tss):
        self.el = [e.lower() for e in ents]
        self.rl = [r.lower() for r in rels]
        self.tl = [t.lower() for t in tss]
        self.cache = {}

    def __call__(self, topic):
        t = topic.lower()
        if t not in self.cache:
            res = {}
            for name, L in (("subject", self.el), ("object", self.el), ("relation", self.rl), ("timestamp", self.tl)):
                if name == "object":
                    res[name] = res["subject"]
                    continue
                res[name] = (np.fromiter((t in x for x in L), bool, len(L)), np.fromiter((x in t for x in L), bool, len(L)))
            self.cache[t] = res
        return self.cache[t]


def atc_mask_substring(slot, topic, cand, s_c, o_c, r_c, ts_c, contain: ContainMasks, sent_lower):
    """Original TEAR rule: topic vs slot value, then topic vs the whole fact string."""
    t = topic.lower()
    m_in, m_rev = contain(topic)[slot]
    ids_slot = {"subject": s_c, "object": o_c, "relation": r_c, "timestamp": ts_c}[slot]
    m1 = m_in[ids_slot]
    m2 = m_rev[ids_slot] & ~m1
    mask = m1.copy()
    for j in np.nonzero(m2)[0]:
        fs = sent_lower[cand[j]]
        if t in fs or fs in t:
            mask[j] = True
    return mask
