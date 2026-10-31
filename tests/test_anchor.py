import types

import numpy as np
import pytest

from tear.labeling.anchor import AnchorResolver, anchor_phrase, pick_date
from tear.rerank.features import build_features
from tear.rerank.tr import period_validity
from tear.temporal import date_to_days, days_to_date, explicit_same_period, period_bounds

# Original MultiTQ test questions (the anchor is an event, not a date)
PHRASES = [
    ("After the Danish Ministry of Defence and Security, who was the first to visit Iraq?", ("after", "the Danish Ministry of Defence and Security")),
    ("Who was the first to praise Iraq after Borys Tarasyuk?", ("after", "Borys Tarasyuk")),
    ("Who last would wish to visit Thailand before the Charge D'affaires of United States did?", ("before", "the Charge D'affaires of United States")),
    ("Which country receive Special Rapporteurs of the United Nations's visit from Special Rapporteurs of the United Nations first after Cambodia did?", ("after", "Cambodia")),
    ("Who made optimistic remarks about China in the same month of Islam Karimov?", ("month", "Islam Karimov")),
    ("Who negotiated with Thailand on the same day as Lee Myung Bak?", ("day", "Lee Myung Bak")),
    ("In the same month as the African government, who did China wish to engage in diplomatic cooperation with?", ("month", "the African government")),
]
NOT_ANCHORED = [
    "After 2006-01-05, who was the first to visit Iraq?",
    "After 12 May 2011, which country did Cambodia denounce?",
    "Before Sep, 2011, which country did China praise?",
    "In 2012, with which country did Malaysia last wish to establish diplomatic cooperation?",
    "Who negotiated with Thailand on the same day as 2009-04-11?",
    "When did China use conventional military force against Iraq?",
]


@pytest.mark.parametrize("question,expected", PHRASES)
def test_anchor_phrase(question, expected):
    assert anchor_phrase(question) == expected


@pytest.mark.parametrize("question", NOT_ANCHORED)
def test_dates_are_not_anchors(question):
    assert anchor_phrase(question) is None


def test_pick_date_widest_window():
    days = np.array([10, 20, 30])
    assert pick_date(days, "after", 20) == 10 and pick_date(days, "before", 20) == 30
    assert pick_date(days, "month", 20) == 10 and pick_date(days, "after", 20, "best") == 20


def test_explicit_same_period_and_bounds():
    g, d = explicit_same_period("Who praised China in the same month of 2015-09-03?")
    assert g == "month" and [days_to_date(x) for x in period_bounds(d, g)] == ["2015-09-01", "2015-09-30"]
    g, d = explicit_same_period("Who praised China on the same day of 2011-11?")   # coarse reference widens the period
    assert g == "month"
    assert explicit_same_period("Who visited China on the same day as Georgios Papandreou?") is None
    lo, hi = period_bounds(date_to_days("2012-02-10"), "year")
    assert (days_to_date(lo), days_to_date(hi)) == ("2012-01-01", "2012-12-31")
    assert period_validity((5, 7), np.array([4, 5, 7, 8])).tolist() == [False, True, True, False]


# toy KG: topic Iraq; Japan visits Iraq on days 100 and 300, consults on 200; France visits on 150 and 400
ENTS = ["Iraq", "Japan", "France", "Citizen (Japan)"]
RELS = ["Make a visit", "Consult"]
FACTS = {"s_id": np.array([1, 1, 1, 2, 2]), "r_id": np.array([0, 0, 1, 0, 0]), "o_id": np.array([0, 0, 0, 0, 0]),
         "t_days": np.array([100, 300, 200, 150, 400])}


def resolver(**kw):
    fe = np.zeros((5, 2), np.float32); fe[:, 0] = [1, 1, 0, 0.5, 0.5]; fe[:, 1] = [0, 0, 1, 0, 0]
    ee = np.eye(4, 2, dtype=np.float32)
    return AnchorResolver(FACTS, ENTS, RELS, ee, fe, **kw)


def test_resolver_kg_lookup():
    q_visit = np.array([1, 0], np.float32)          # the question is about visits
    r = resolver().resolve("after", "Japan", "subject", "Iraq", q_visit)
    assert (r["status"], r["anchor"], r["relation"], r["n_dates"], r["date"]) == \
        ("resolved", "Japan", "Make a visit", 2, days_to_date(100))
    assert resolver().resolve("before", "Japan", "subject", "Iraq", q_visit)["date"] == days_to_date(300)
    q_consult = np.array([0, 1], np.float32)
    assert resolver().resolve("after", "Japan", "subject", "Iraq", q_consult)["relation"] == "Consult"
    assert resolver().resolve("after", "Germany", "subject", "Iraq", q_visit)["status"] == "no_anchor_entity"
    assert resolver().resolve("after", "Japan", "timestamp", "Iraq", q_visit)["status"] == "no_topic"
    # implied answers: first visitor of Iraq after day 100 is France (150)
    assert resolver().implied_answers("subject", "Iraq", "Make a visit", "after", 100, ["after", "first"]) == {"France"}


def test_build_features_uses_anchor_only_when_enabled():
    ids = np.array([[0, 1, 2, 3, 4]])
    pool = types.SimpleNamespace(
        ids=ids, s=FACTS["s_id"], r=FACTS["r_id"], o=FACTS["o_id"], date=FACTS["t_days"], start=FACTS["t_days"],
        end=FACTS["t_days"], ts_start=np.zeros(5, int), interval=False, ents=ENTS, rels=RELS, tss=["x"], E=4, R=2,
        qs=[{"quid": 1, "rewrite": "After Japan, who was the first to visit Iraq?"}], cue_field="rewrite",
        key=lambda q: str(q["quid"]), sent_lower=[""] * 5)
    slots = {"1": {"answer_slot": "subject", "topic_entity": "Iraq"}}
    base = build_features(pool, slots, None, "any", "exact")
    assert not base["has_window"][0] and "anchor_used" not in base["meta"].columns
    anchors = {"1": {"type": "after", "date": days_to_date(100), "anchor": "Japan"}}
    F = build_features(pool, slots, None, "any", "exact", "v2", anchors)
    assert F["has_window"][0] and F["window"][0].tolist() == [False, True, True, True, True]
    assert F["meta"].anchor_used[0]
    # the anchor (Japan, subject of facts 0-2) joins the topic in the ATC penalty; without anchors only the topic
    assert F["atc"][0].tolist() == [True, True, True, False, False] and not base["atc"][0].any()
    pool.qs = [{"quid": 1, "rewrite": "Who visited Iraq in the same month as Japan?"}]
    F = build_features(pool, slots, None, "any", "exact", "v2", {"1": {"type": "month", "date": "1970-04-10"}}, True)
    lo, hi = period_bounds(date_to_days("1970-04-10"), "month")
    assert F["window"][0].tolist() == [(lo <= t <= hi) for t in FACTS["t_days"]]
