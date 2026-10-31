"""Per-question reranking features over the candidate pool (one pass, pure functions from atc / tr / baselines)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..temporal import date_to_days, explicit_same_period, extract_temporal_cue, period_bounds, target_days
from .atc import ContainMasks, atc_mask_exact, atc_mask_substring, resolve_labels
from .baselines import chrono_direction, timer4_phi
from .tr import (before_after_validity, from_to_window, group_keys, group_representatives, non_representative_mask,
                 period_validity)


def build_features(pool, slots: dict, topics_override: dict | None = None, date_granularity: str = "full",
                   atc_match: str = "substring", date_parser: str = "v1", anchors: dict | None = None,
                   equal_window: bool = False) -> dict:
    """anchors (optional, tear.labeling.anchor): {question key: {"type": before|after|day|month|year, "date"}}, the
    reference date of an event-anchored constraint; a before/after anchor is used when the question has no explicit
    before/after date, and the anchor entity ("anchor") joins the topic entity in the ATC penalty. equal_window: a same-period constraint ('in the same month as <date>', or a day/month/year
    anchor) restricts the window and TR's valid candidates to that period.
    Arrays of shape (Q, N) unless noted:
      atc          True = penalised by the ATC rule (lambda1)
      tr           True = not a group representative (lambda2)
      window       True = inside the before/after (or from-to) window; has_window (Q,)
      phi_before / phi_after  TimeR4 phi_t (NaN rows when the question has no such date)
      chrono_dir (Q,), cand_date, meta (DataFrame: which rules fired per question)
    """
    Q, N = pool.ids.shape
    atc = np.zeros((Q, N), bool)
    tr = np.zeros((Q, N), bool)
    window = np.ones((Q, N), bool)
    has_window = np.zeros(Q, bool)
    phi_before = np.full((Q, N), np.nan, np.float32)
    phi_after = np.full((Q, N), np.nan, np.float32)
    chrono_dir = np.zeros(Q, np.int8)
    cand_date = np.zeros((Q, N), np.int64)
    meta = []
    contain = ContainMasks(pool.ents, pool.rels, pool.tss) if atc_match == "substring" else None

    for qi, q in enumerate(pool.qs):
        cand = pool.ids[qi]
        s_c, r_c, o_c = pool.s[cand], pool.r[cand], pool.o[cand]
        cand_date[qi] = pool.date[cand]
        # ---- answer slot / topic entity -> ATC penalty mask
        slot, topic = resolve_labels(pool.key(q), slots, topics_override)
        if slot and topic:
            if atc_match == "exact":
                m = atc_mask_exact(slot, topic, s_c, o_c, r_c, pool.ents, pool.rels)
                if m is not None:
                    atc[qi] = m
            else:
                atc[qi] = atc_mask_substring(slot, topic, cand, s_c, o_c, r_c, pool.ts_start[cand], contain,
                                             pool.sent_lower)
        anc = anchors.get(pool.key(q)) if anchors else None
        anchor_atc = False
        if slot and anc and anc.get("anchor"):
            # the anchor is named by the question like the topic, so it cannot be the answer either
            m = atc_mask_exact(slot, anc["anchor"], s_c, o_c, r_c, pool.ents, pool.rels)
            if m is not None:
                atc[qi] |= m; anchor_atc = bool(m.any())
        # ---- temporal constraints
        text = q[pool.cue_field]
        cues = extract_temporal_cue(text)
        tdays = target_days(text, date_granularity, date_parser)
        anc_days = date_to_days(anc["date"]) if anc and anc.get("date") else None
        anchor_used, period = False, None
        if anc_days is not None and not tdays and anc["type"] in ("before", "after"):
            tdays = [(anc["type"], anc_days)]; anchor_used = True
        if equal_window:
            sp = explicit_same_period(text)
            if sp is None and anc_days is not None and anc["type"] in ("day", "month", "year"):
                sp = (anc["type"], anc_days); anchor_used = True
            if sp is not None:
                period = period_bounds(sp[1], sp[0])
        valid = before_after_validity(tdays, pool.start[cand], pool.end[cand], pool.date[cand], pool.interval)
        if period is not None:
            valid &= period_validity(period, pool.date[cand], pool.start[cand], pool.end[cand], pool.interval)
        if tdays or period is not None:
            window[qi] = valid; has_window[qi] = True
        elif pool.interval:
            w = from_to_window(text, pool.start[cand], pool.end[cand])
            if w is not None:
                window[qi] = w; has_window[qi] = True
        for typ, store in (("before", phi_before), ("after", phi_after)):
            ph = timer4_phi(tdays, typ, cand_date[qi])
            if ph is not None:
                store[qi] = ph
        chrono_dir[qi] = chrono_direction(cues)
        # ---- TR mask (group representatives)
        tr_fired = False
        if cues and slot:
            keys = group_keys(slot, s_c, r_c, o_c, pool.E, pool.R)
            reps = group_representatives(keys, valid, cand_date[qi], cues)
            tr[qi] = non_representative_mask(N, reps)
            tr_fired = len(reps) > 0
        row = {"slot": slot, "topic": topic, "has_cue_first": "first" in cues, "has_cue_last": "last" in cues,
               "has_cue_before": "before" in cues, "has_cue_after": "after" in cues,
               "has_target_date": bool(tdays), "has_window": bool(has_window[qi]), "tr_fired": tr_fired,
               "atc_any": bool(atc[qi].any())}
        if anchors is not None or equal_window:
            row.update({"anchor_used": anchor_used, "has_period": period is not None, "anchor_atc": anchor_atc})
        meta.append(row)
        if qi % 10000 == 0:
            print(f"  features {qi}/{Q}", flush=True)
    return dict(atc=atc, tr=tr, window=window, has_window=has_window, phi_before=phi_before, phi_after=phi_after,
                chrono_dir=chrono_dir, cand_date=cand_date, meta=pd.DataFrame(meta))
