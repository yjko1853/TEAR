"""Fidelity statistics of generated question sets (tear.data.qgen): question form, answers, and the KG properties TEAR's rules rely on.

Text side (any question set): answer type and time level shares, question length, answers per question, temporal
cue words, date expressions (all formats, and the before/after dates that TEAR's date parser v2 reads), event-anchor
phrases that TEAR's anchor grounding detects, same-period phrases, topic concentration.
KG side (generated questions, or MultiTQ questions aligned to the KG): the share of questions whose gold fact belongs
to an (s, r, o) group seen on two or more days (the situation TR addresses), anchor events seen on several days
(where the reference date is ambiguous), and the number of periods of time-answer triples.
"""
from __future__ import annotations

import collections
import re

import numpy as np
import pandas as pd

from ..data.qgen import timeexpr
from ..labeling.anchor import anchor_phrase
from ..temporal import SAME_PERIOD, extract_temporal_cue, extract_target_dates_v2

QTYPES = ("equal", "before_after", "first_last", "equal_multi", "after_first", "before_last")


def text_row(q: dict) -> dict:
    t = q["question"]
    low = t.lower()
    cues = extract_temporal_cue(t)
    dates = timeexpr.find(t)
    ba = extract_target_dates_v2(t)
    return {"qtype": q["qtype"], "answer_type": q["answer_type"], "time_level": q["time_level"],
            "n_words": len(t.split()), "n_answers": len(q["answers"]),
            "cue_first_last": ("first" in cues) or ("last" in cues),
            "cue_before_after": ("before" in cues) or ("after" in cues),
            "has_date": bool(dates), "ba_date_parsed_v2": bool(ba),
            "anchor_phrase": anchor_phrase(t) is not None, "same_period": bool(SAME_PERIOD.search(t)),
            "date_shape": dates[0]["shape"] if dates else None, "low": low}


def text_table(questions) -> pd.DataFrame:
    return pd.DataFrame([text_row(q) for q in questions])


def summarise_text(df: pd.DataFrame, by: str | None = "qtype") -> pd.DataFrame:
    def agg(g):
        return pd.Series({
            "n": len(g),
            "time_answer": (g.answer_type == "time").mean(),
            "level_day": (g.time_level == "day").mean(), "level_month": (g.time_level == "month").mean(),
            "level_year": (g.time_level == "year").mean(),
            "words_mean": g.n_words.mean(), "words_p10": g.n_words.quantile(0.1), "words_p90": g.n_words.quantile(0.9),
            "answers_mean": g.n_answers.mean(), "answers_1": (g.n_answers == 1).mean(),
            "answers_ge5": (g.n_answers >= 5).mean(), "answers_max": g.n_answers.max(),
            "cue_first_last": g.cue_first_last.mean(), "cue_before_after": g.cue_before_after.mean(),
            "has_date": g.has_date.mean(), "ba_date_parsed_v2": g.ba_date_parsed_v2.mean(),
            "anchor_phrase": g.anchor_phrase.mean(), "same_period": g.same_period.mean()})
    if by is None:
        return agg(df).to_frame().T.assign(qtype="all")
    out = df.groupby(by).apply(agg).reset_index()
    return pd.concat([out, agg(df).to_frame().T.assign(**{by: "all"})], ignore_index=True)


def reweight(summary: pd.DataFrame, weights: dict) -> pd.Series:
    """Per-type rows averaged with the given type weights (e.g. MultiTQ test proportions)."""
    s = summary[summary.qtype.isin(list(weights))].set_index("qtype")
    w = pd.Series(weights) / sum(weights.values())
    cols = [c for c in s.columns if c != "n" and pd.api.types.is_numeric_dtype(s[c])]
    return (s[cols].astype(float).mul(w, axis=0)).sum()


def date_shape_share(df: pd.DataFrame, k: int = 8) -> pd.DataFrame:
    rows = []
    for lvl, g in df[df.date_shape.notna()].groupby("time_level"):
        vc = g.date_shape.value_counts(normalize=True)
        for shape, p in vc.head(k).items():
            rows.append({"time_level": lvl, "shape": shape, "share": p})
    return pd.DataFrame(rows)


def topic_concentration(topics: list) -> dict:
    c = collections.Counter(t for t in topics if t)
    n = sum(c.values())
    return {"n": n, "distinct_topics": len(c), "top10_share": sum(v for _, v in c.most_common(10)) / n if n else np.nan,
            "top1": c.most_common(1)[0] if c else None}


def kg_rows_generated(questions, kg) -> pd.DataFrame:
    """KG-side properties of generated questions (from the generator's record)."""
    nd = kg.sro_n_days()
    rows = []
    for q in questions:
        g = q["gen"]
        gold = g.get("gold_fact_ids") or []
        rows.append({"qtype": q["qtype"], "answer_type": q["answer_type"],
                     "gold_sro_repeated": bool(gold) and bool((nd[np.array(gold)] >= 2).any()),
                     "anchor_n_days": g.get("anchor_n_days"), "n_periods": g.get("n_periods"),
                     "topic": g.get("topic_entity") or g.get("subject"), "variant": g["variant"],
                     "template_status": g.get("template_status")})
    return pd.DataFrame(rows)


def kg_rows_aligned(questions, rows, kg) -> pd.DataFrame:
    """KG-side properties of MultiTQ questions from their alignment rows (aligned questions only)."""
    nd = kg.sro_n_days()
    qby = {str(q["quid"]): q for q in questions}
    out = []
    for r in rows:
        if r.get("relation") is None:
            continue
        q = qby[str(r["quid"])]
        rid = kg.rid[r["relation"]]
        if r["side"] == "timestamp":
            fids = kg.sro_fact_ids(kg.eid[r["subject"]], rid, kg.eid[r["object"]])
        else:
            topic = kg.eid[r["topic"]]
            ev = kg.events(r["side"], rid, topic)
            ans = {kg.eid[a] for a in q["answers"] if a in kg.eid}
            fids = ev.fids[np.isin(ev.partners, list(ans))]
        out.append({"qtype": r["qtype"], "answer_type": r["answer_type"],
                    "gold_sro_repeated": bool(len(fids)) and bool((nd[fids] >= 2).any()),
                    "anchor_n_days": r.get("anchor_n_days"), "n_periods": r.get("n_periods"),
                    "topic": r.get("topic") or r.get("subject"), "variant": r.get("variant"),
                    "template_status": r.get("status")})
    return pd.DataFrame(out)


def summarise_kg(df: pd.DataFrame) -> pd.DataFrame:
    def agg(g):
        an = g.anchor_n_days.dropna()
        npd = g.n_periods.dropna()
        return pd.Series({"n": len(g), "gold_sro_repeated": g.gold_sro_repeated.mean(),
                          "anchored": len(an) / len(g), "anchor_multi_day": (an > 1).mean() if len(an) else np.nan,
                          "time_single_period": (npd == 1).mean() if len(npd) else np.nan})
    out = df.groupby("qtype").apply(agg).reset_index()
    return pd.concat([out, agg(df).to_frame().T.assign(qtype="all")], ignore_index=True)


def kg_summary(kg, relations=None) -> dict:
    """Fact-level properties of a KG (optionally restricted to some relations)."""
    nd = kg.sro_n_days()
    m = np.ones(kg.n_facts, bool) if relations is None else np.isin(kg.R, [kg.rid[r] for r in relations if r in kg.rid])
    used = np.union1d(kg.S[m], kg.O[m])
    return {"facts": int(m.sum()), "entities_used": int(len(used)), "relations": int(len(np.unique(kg.R[m]))),
            "days": int(len(np.unique(kg.D[m]))), "facts_in_repeated_sro": float((nd[m] >= 2).mean())}


# ---------------------------------------------------------------------- TEAR's rules against the generator's record
def _anchor_type(q) -> str | None:
    g = q["gen"]
    if g["variant"] == "same":
        return q["time_level"]
    if q["qtype"] == "after_first":
        return "after"
    if q["qtype"] == "before_last":
        return "before"
    if q["qtype"] == "before_after" and g["variant"] == "anchor":
        return "before" if g.get("before") else "after"
    return None


def rule_check(questions, voc: dict, facts, labels: dict | None = None, anchors: dict | None = None) -> dict:
    """Rule v2 answer slot and topic entity, date parser v2, first/last cues and anchor phrases against what the
    generator recorded. labels: a slots file of scripts/04 (default: rule v2 computed here without the NER fallback).
    anchors: anchors.json of scripts/14 (full grounding with the retriever's scores); without it a mention-only lookup
    is used (AnchorResolver without fact scores, so the relation is the first candidate's: a lower bound)."""
    from ..labeling import rules
    from ..labeling.anchor import AnchorResolver, anchor_phrase
    from ..temporal import date_to_days, extract_temporal_cue, target_days
    if labels is None:
        labels = rules.label_multitq(questions, voc["entities"], None, "v2")
    res = AnchorResolver(facts, voc["entities"], voc["relations"], None, None)
    rows = []
    for q in questions:
        g = q["gen"]
        lab = labels[str(q["quid"])]
        gold_topics = {g["subject"], g["object"]} if q["answer_type"] == "time" else {g["topic_entity"]}
        cues = extract_temporal_cue(q["question"])
        row = {"quid": q["quid"], "qtype": q["qtype"], "answer_type": q["answer_type"], "variant": g["variant"],
               "template_status": g.get("template_status"), "gold_slot": g["answer_slot"],
               "rule_slot": lab["answer_slot"], "slot_ok": lab["answer_slot"] == g["answer_slot"],
               "rule_topic": lab["topic_entity"], "topic_source": lab.get("topic_source"),
               "topic_ok": lab["topic_entity"] in gold_topics}
        if g.get("first") is not None:
            want = "first" if g["first"] else "last"
            row["cue_ok"] = want in cues and (("last" if g["first"] else "first") not in cues)
        if q["qtype"] in ("after_first", "before_last"):
            row["cue_ok"] = ("first" if q["qtype"] == "after_first" else "last") in cues
        te = g.get("time_expression")
        if te and g["variant"] == "date":
            td = target_days(q["question"], "any", "v2")
            want = date_to_days(te["from"]) if g.get("before") or q["qtype"] == "before_last" else date_to_days(te["to"])
            row["date_parsed"] = bool(td)
            row["date_ok"] = bool(td) and td[0][1] == want
        at = _anchor_type(q)
        if at:
            ph = anchor_phrase(q["question"])
            row["anchor_detected"] = ph is not None
            row["anchor_type_ok"] = ph is not None and ph[0] == at
            if anchors is not None and str(q["quid"]) in anchors:
                r = anchors[str(q["quid"])]
                ph = ph or (r.get("type"), r.get("phrase"))
            elif ph is not None:
                r = res.resolve(ph[0], ph[1], lab["answer_slot"], lab["topic_entity"])
            if ph is not None:
                row["anchor_resolved"] = r["status"] == "resolved"
                row["anchor_entity_ok"] = r["anchor"] == g["anchor_entity"]
                row["anchor_relation_ok"] = r["relation"] == g["relation"]
                row["anchor_date_eq_gen"] = r["date"] == g["anchor_date"]
                row["anchor_n_days"] = g.get("anchor_n_days")
            else:
                row.update(anchor_resolved=False, anchor_entity_ok=False, anchor_relation_ok=False,
                           anchor_date_eq_gen=False, anchor_n_days=g.get("anchor_n_days"))
        rows.append(row)
    df = pd.DataFrame(rows)

    def acc(group_cols, cols):
        out = []
        for keys, gdf in ([(("all",), df)] + list(df.groupby(group_cols)) if group_cols else [(("all",), df)]):
            keys = keys if isinstance(keys, tuple) else (keys,)
            rec = dict(zip(group_cols or ["group"], keys)) if keys != ("all",) else {c: "all" for c in (group_cols or ["group"])}
            rec["n"] = len(gdf)
            for c in cols:
                if c in gdf:
                    v = gdf[c].dropna()
                    rec[c] = v.astype(float).mean() if len(v) else np.nan
                    rec[c + "_n"] = len(v)
            out.append(rec)
        return pd.DataFrame(out)
    slot = acc(["qtype"], ["slot_ok", "topic_ok", "cue_ok", "date_parsed", "date_ok"])
    by_status = acc(["template_status"], ["slot_ok", "topic_ok"])
    by_slot = acc(["gold_slot"], ["slot_ok"])
    an = df[df.get("anchor_detected").notna()] if "anchor_detected" in df else df.iloc[:0]
    anchor = acc(["qtype", "variant"], ["anchor_detected", "anchor_type_ok", "anchor_resolved", "anchor_entity_ok",
                                       "anchor_relation_ok", "anchor_date_eq_gen"]) if len(an) else pd.DataFrame()
    if len(an):
        anchor = anchor[anchor.qtype.isin(list(an.qtype.unique()) + ["all"])]
        anchor = anchor[anchor["anchor_detected_n"] > 0]
    conf = pd.crosstab(df.gold_slot, df.rule_slot).reset_index()
    return {"per_question": df, "by_qtype": slot, "by_template_status": by_status, "by_gold_slot": by_slot,
            "slot_confusion": conf, "anchor": anchor}


def rule_vs_alignment(questions, rows, entities) -> pd.DataFrame:
    """Reference for rule_check on MultiTQ itself: rule v2 answer slot (no NER fallback) against the answer side of
    the aligned structure (the same alignment that produced the templates), per qtype and alignment status."""
    from ..labeling import rules
    al = {str(r["quid"]): r for r in rows if r.get("relation")}
    qs = [q for q in questions if str(q["quid"]) in al]
    labels = rules.label_multitq([dict(q, rewrite=q["question"]) for q in qs], entities, None, "v2")
    out = []
    for q in qs:
        r = al[str(q["quid"])]
        lab = labels[str(q["quid"])]
        gold_t = {r.get("subject"), r.get("object")} if r["side"] == "timestamp" else {r.get("topic")}
        out.append({"qtype": q["qtype"], "status": r["status"], "slot_ok": lab["answer_slot"] == r["side"],
                    "topic_ok": lab["topic_entity"] in gold_t})
    df = pd.DataFrame(out)
    tab = df.groupby("qtype")[["slot_ok", "topic_ok"]].mean().reset_index()
    tab = pd.concat([tab, pd.DataFrame([{"qtype": "all", "slot_ok": df.slot_ok.mean(), "topic_ok": df.topic_ok.mean()}])])
    by = df.groupby("status")[["slot_ok", "topic_ok"]].mean().reset_index().rename(columns={"status": "qtype"})
    by["qtype"] = "status=" + by["qtype"]
    tab = pd.concat([tab, by], ignore_index=True)
    tab["n"] = [int((df.qtype == x).sum()) if x in set(df.qtype) else (len(df) if x == "all" else
                int((df.status == x.split("=", 1)[1]).sum())) for x in tab.qtype]
    return tab
