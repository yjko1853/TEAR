"""KG-derived ("oracle") answer slots and topic entities.

DIAGNOSTIC ONLY - do not use as gold labels for MultiTQ. For entity answers the slot is a majority vote over every
KG fact that links an answer entity to an entity mentioned in the question, regardless of the relation. ICEWS codes
many events in both directions (Make a visit / Host a visit, signing, consultations), so the vote often returns the
reversed slot; the rule classifier (rules.py) reads the slot off the question grammar instead.

MultiTQ entity answers: majority vote over the answer's position (subject | object) among those facts; the most
frequent counterpart among the winning facts is the topic entity.
MultiTQ time answers: slot = timestamp; topic entity = the question entity that most often co-occurs in facts whose
timestamp matches the answer at the answer's granularity (day / month / year).
TimelineKGQA: slot from the dataset's answer_type (subject / object / timestamp_*); topic entity = the counterpart of
the gold fact that contains the answer (the gold fact's subject for timestamp answers); duration / relation_* -> null.
"""
from __future__ import annotations

import collections

import numpy as np
import pandas as pd

from ..text import normalize
from ..utils.io import load_json


def build_ngram_lookup(entities):
    norm2id = {}
    maxlen = 1
    for i, e in enumerate(entities):
        n = normalize(e)
        if not n:
            continue
        norm2id.setdefault(n, i)
        maxlen = max(maxlen, len(n.split()))
    return norm2id, maxlen


def entities_in_text(text, norm2id, maxlen):
    toks = normalize(text).split()
    found = set()
    for i in range(len(toks)):
        for n in range(1, min(maxlen, len(toks) - i) + 1):
            e = norm2id.get(" ".join(toks[i:i + n]))
            if e is not None:
                found.add(e)
    return found


def multitq_oracle(d):
    """Returns (labels dict, derivation table)."""
    f = np.load(d / "facts.npz")
    voc = load_json(d / "vocab.json")
    ents, tss = voc["entities"], voc["timestamps"]
    s_id, o_id, t_id = f["s_id"], f["o_id"], f["t_id"]
    qs = load_json(d / "questions.json")
    norm2id, maxlen = build_ngram_lookup(ents)
    order_s = np.argsort(s_id, kind="stable")
    order_o = np.argsort(o_id, kind="stable")
    bounds_s = np.searchsorted(s_id[order_s], np.arange(len(ents) + 1))
    bounds_o = np.searchsorted(o_id[order_o], np.arange(len(ents) + 1))
    by_day = {t: i for i, t in enumerate(tss)}
    by_month, by_year = collections.defaultdict(list), collections.defaultdict(list)
    for i, t in enumerate(tss):
        by_month[t[:7]].append(i)
        by_year[t[:4]].append(i)
    order_t = np.argsort(t_id, kind="stable")
    bounds_t = np.searchsorted(t_id[order_t], np.arange(len(tss) + 1))

    out, rows = {}, []
    for q in qs:
        qents = entities_in_text(q["question"], norm2id, maxlen) | entities_in_text(q["rewrite"], norm2id, maxlen)
        rec = {"answer_slot": None, "topic_entity": None, "derived": False, "method": "kg_oracle",
               "n_question_entities": len(qents)}
        if q["answer_type"] == "entity":
            pos_votes = collections.Counter()
            cp_votes = {"subject": collections.Counter(), "object": collections.Counter()}
            ans_ids = [norm2id.get(normalize(a)) for a in q["answers"]]
            ans_ids = [a for a in ans_ids if a is not None]
            rec["answer_entities_found"] = len(ans_ids)
            for a in ans_ids:
                fs = order_s[bounds_s[a]:bounds_s[a + 1]]  # answer is subject -> counterpart = object
                for cp in o_id[fs]:
                    if cp in qents and cp != a:
                        pos_votes["subject"] += 1
                        cp_votes["subject"][int(cp)] += 1
                fo = order_o[bounds_o[a]:bounds_o[a + 1]]  # answer is object -> counterpart = subject
                for cp in s_id[fo]:
                    if cp in qents and cp != a:
                        pos_votes["object"] += 1
                        cp_votes["object"][int(cp)] += 1
            if pos_votes:
                slot = pos_votes.most_common(1)[0][0]
                rec["answer_slot"] = slot
                rec["topic_entity"] = ents[cp_votes[slot].most_common(1)[0][0]]
                rec["derived"] = True
                rec["votes"] = dict(pos_votes)
        else:
            rec["answer_slot"] = "timestamp"
            tids = set()
            for a in q["answers"]:
                if len(a) == 10 and a in by_day:
                    tids.add(by_day[a])
                elif len(a) == 7:
                    tids.update(by_month.get(a, []))
                elif len(a) == 4:
                    tids.update(by_year.get(a, []))
            cand = collections.Counter()
            if tids and qents:
                qe = np.fromiter(qents, dtype=np.int32)
                fids = np.concatenate([order_t[bounds_t[t]:bounds_t[t + 1]] for t in tids]) if tids else np.array([], int)
                ss, oo = s_id[fids], o_id[fids]
                for arr in (ss[np.isin(ss, qe)], oo[np.isin(oo, qe)]):
                    for e, c in zip(*np.unique(arr, return_counts=True)):
                        cand[int(e)] += int(c)
            if cand:
                rec["topic_entity"] = ents[cand.most_common(1)[0][0]]
                rec["derived"] = True
            elif qents:
                rec["topic_entity"] = ents[max(qents, key=lambda e: len(ents[e]))]
                rec["derived"] = True
                rec["method"] = "longest_question_entity"
        out[str(q["quid"])] = rec
        rows.append({"qtype": q["qtype"], "answer_type": q["answer_type"], "derived": rec["derived"],
                     "slot": rec["answer_slot"]})
    df = pd.DataFrame(rows)
    tab = df.groupby(["answer_type", "qtype"]).agg(n=("derived", "size"), derived_rate=("derived", "mean"),
                                                   subject=("slot", lambda s: (s == "subject").mean()),
                                                   object=("slot", lambda s: (s == "object").mean()),
                                                   timestamp=("slot", lambda s: (s == "timestamp").mean())).reset_index()
    print(tab.to_string())
    print("overall derived rate:", df.derived.mean())
    return out, tab


def timeline_oracle(d):
    """Returns (labels dict, derivation table)."""
    f = np.load(d / "facts.npz")
    voc = load_json(d / "vocab.json")
    ents = voc["entities"]
    s_id, o_id = f["s_id"], f["o_id"]
    qs = load_json(d / "questions.json")
    out, rows = {}, []
    for q in qs:
        at = q["answer_type"]
        rec = {"answer_slot": None, "topic_entity": None, "derived": False, "method": "dataset_label"}
        gold = [g for g in q["gold_fact_ids"] if g >= 0]
        if at in ("subject", "object"):
            rec["answer_slot"] = at
            a = q["answer"]
            for g in gold:
                if at == "subject" and ents[s_id[g]] == a:
                    rec["topic_entity"] = ents[o_id[g]]; rec["derived"] = True; break
                if at == "object" and ents[o_id[g]] == a:
                    rec["topic_entity"] = ents[s_id[g]]; rec["derived"] = True; break
        elif at.startswith("timestamp"):
            rec["answer_slot"] = "timestamp"
            if gold:
                rec["topic_entity"] = ents[s_id[gold[0]]]; rec["derived"] = True
        out[str(q["qid"])] = rec
        rows.append({"answer_type": at, "level": q["level"], "derived": rec["derived"], "slot": rec["answer_slot"]})
    df = pd.DataFrame(rows)
    tab = df.groupby(["answer_type", "level"]).agg(n=("derived", "size"), derived_rate=("derived", "mean")).reset_index()
    print(tab.to_string())
    return out, tab
