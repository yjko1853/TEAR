"""MultiTQ-style question generator (tear.data.qgen) on a synthetic KG."""
import gzip
import json

import numpy as np
import pytest

from tear.data import is_multitq
from tear.data.qgen import aliases, semantics, timeexpr
from tear.data.qgen.align import Aligner, FrameLM, frame_words
from tear.data.qgen.freeze import load_frozen, save_frozen, text_variant
from tear.data.qgen.generate import Generator, fill
from tear.data.qgen.kg import TemporalKG, prepare_renet
from tear.temporal import date_to_days

D = date_to_days


def make_kg():
    ents = ["Iran", "Japan", "China", "France", "Government (Nigeria)", "Barack Obama"]
    rels = ["Make a visit", "Praise or endorse"]
    rows = [  # s, r, o, day
        ("Japan", "Make a visit", "Iran", "2010-01-05"),
        ("China", "Make a visit", "Iran", "2010-01-20"),
        ("Japan", "Make a visit", "Iran", "2010-03-02"),
        ("France", "Make a visit", "Iran", "2010-03-02"),
        ("Barack Obama", "Make a visit", "Iran", "2011-07-01"),
        ("Government (Nigeria)", "Praise or endorse", "France", "2012-05-05"),
        ("Japan", "Praise or endorse", "China", "2012-05-06"),
    ]
    e = {x: i for i, x in enumerate(ents)}
    r = {x: i for i, x in enumerate(rels)}
    return TemporalKG([e[a] for a, _, _, _ in rows], [r[b] for _, b, _, _ in rows], [e[c] for _, _, c, _ in rows],
                      [D(d) for *_, d in rows], ents, rels)


# ---------------------------------------------------------------------- dates
@pytest.mark.parametrize("text,gran,shape", [
    ("on 28 September 2007", "day", "{D} {Month} {YYYY}"),
    ("in Sep 2014,", "month", "{Mon} {YYYY}"),
    ("after Jun, 2011?", "month", "{Mon}, {YYYY}"),
    ("in 2015-06?", "month", "{YYYY}-{MM}"),
    ("Before Apr 11th, 2014,", "day", "{Mon} {Dth}, {YYYY}"),
    ("on 15.08.2012?", "day", "{D}.{MM}.{YYYY}"),
    ("On the 11th of March 2015,", "day", "the {Dth} of {Month} {YYYY}"),
    ("Before Nov, year 2007,", "month", "{Mon}, year {YYYY}"),
    ("in July, the year of 2007?", "month", "{Month}, the year of {YYYY}"),
    ("after 2008?", "year", "{YYYY}"),
])
def test_time_expression_roundtrip(text, gran, shape):
    f = timeexpr.find(text)
    assert len(f) == 1 and f[0]["gran"] == gran and f[0]["shape"] == shape
    assert timeexpr.render(f[0]["lo"], shape) == f[0]["text"]
    assert timeexpr.shape_granularity(shape) == gran


def test_period_and_labels():
    lo, hi = timeexpr.period_of(D("2012-02-10"), "month")
    assert (lo, hi) == (D("2012-02-01"), D("2012-02-29"))
    assert timeexpr.period_label(D("2012-02-10"), "year") == "2012"
    assert [timeexpr.ordinal(n) for n in (1, 2, 3, 4, 11, 12, 13, 21, 22, 23)] == \
        ["1st", "2nd", "3rd", "4th", "11th", "12th", "13th", "21st", "22nd", "23rd"]


# ---------------------------------------------------------------------- aliases
def test_alias_render_and_mentions():
    n = "Government (Nigeria)"
    assert aliases.render(n, {"kind": "of", "case": "lower", "article": "the "}) == "the government of Nigeria"
    assert aliases.render(n, {"kind": "poss"}) == "Nigeria's Government"
    assert aliases.render(n, {"kind": "adj", "case": "lower", "article": "the "}, {"Nigeria": "Nigerian"}) == \
        "the Nigerian government"
    assert aliases.render(n, {"kind": "adj"}, {}) is None
    assert aliases.render("Citizen (Brazil)", {"kind": "of", "number": "plur", "case": "lower"}) == "citizens of Brazil"
    idx = aliases.AliasIndex(["Government (Nigeria)", "Nigeria", "Iran"], {"Nigeria": "Nigerian"})
    ms = idx.mentions("Who praised the Nigerian government before Iran did?")
    gov = [m for m in ms if m["entity"] == 0]
    assert gov and gov[0]["kind"] == "adj" and gov[0]["article"] == "the " and gov[0]["case"] == "lower"
    assert gov[0]["text"] == "the Nigerian government"
    assert any(m["entity"] == 2 for m in ms)


# ---------------------------------------------------------------------- answer procedures
def test_entity_semantics():
    kg = make_kg()
    vis, iran = kg.rid["Make a visit"], kg.eid["Iran"]
    ev = kg.events("subject", vis, iran)
    name = lambda ids: {kg.name(i) for i in ids}  # noqa: E731
    lo, hi = timeexpr.period_of(D("2010-01-10"), "month")
    a, _, _ = semantics.entity_answers(ev, "equal", "period", lo=lo, hi=hi)
    assert name(a) == {"Japan", "China"}
    a, _, _ = semantics.entity_answers(ev, "before_after", "date", before=True, lo=D("2010-03-02"), hi=D("2010-03-02"))
    assert name(a) == {"Japan", "China"}                                   # strict
    a, _, _ = semantics.entity_answers(ev, "first_last", "extreme", first=False)
    assert name(a) == {"Barack Obama"}
    a, _, _ = semantics.entity_answers(ev, "first_last", "extreme", first=True)
    assert name(a) == {"Japan"}
    # anchor Japan: earliest occurrence 2010-01-05 (not 2010-03-02)
    a, _, info = semantics.entity_answers(ev, "after_first", "anchor", anchor=kg.eid["Japan"])
    assert name(a) == {"China"} and info["t0"] == D("2010-01-05") and info["anchor_n_days"] == 2
    a, _, _ = semantics.entity_answers(ev, "before_last", "anchor", anchor=kg.eid["France"])
    assert name(a) == {"China"}
    a, _, _ = semantics.entity_answers(ev, "before_after", "anchor", anchor=kg.eid["China"], before=False)
    assert name(a) == {"Japan", "France", "Barack Obama"}
    a, _, _ = semantics.entity_answers(ev, "equal_multi", "same", anchor=kg.eid["France"], gran="month")
    assert name(a) == {"Japan"}                                            # the anchor itself is not an answer
    lo, hi = timeexpr.period_of(D("2010-06-01"), "year")
    a, _, _ = semantics.entity_answers(ev, "equal_multi", "period_fl", first=False, lo=lo, hi=hi)
    assert name(a) == {"Japan", "France"}                                 # ties on the last day of 2010


def test_time_semantics_and_pseudo():
    kg = make_kg()
    days = kg.sro_days(kg.eid["Japan"], kg.rid["Make a visit"], kg.eid["Iran"])
    assert semantics.time_answers(days, "equal", "unique", "year") == ["2010"]
    assert semantics.time_answers(days, "equal", "unique", "month") == []   # two months -> no question
    assert semantics.time_answers(days, "first_last", "extreme", "day", first=False) == ["2010-03-02"]
    ev = kg.events("subject", kg.rid["Make a visit"], kg.eid["Iran"])
    assert semantics.is_pseudo_temporal(ev, [kg.eid[x] for x in ("Japan", "China", "France", "Barack Obama")])


# ---------------------------------------------------------------------- alignment
def test_alignment_reproduces_answers_and_delexicalises():
    kg = make_kg()
    al = Aligner(kg, aliases.AliasIndex(kg.entities))
    q = {"quid": 1, "question": "After Japan, who was the first to visit Iran?", "answers": ["China"],
         "answer_type": "entity", "qtype": "after_first", "time_level": "day"}
    h, fr, status = al.resolve(q, al.hypotheses(q))
    assert status == "unique" and kg.relations[h["r"]] == "Make a visit" and h["side"] == "subject"
    assert fr["frame"] == "After {Y}, who was the first to visit {O}?"
    q2 = {"quid": 2, "question": "In which year did Japan visit Iran?", "answers": ["2010"], "answer_type": "time",
          "qtype": "equal", "time_level": "year"}
    h, fr, status = al.resolve(q2, al.hypotheses(q2))
    assert status == "unique" and fr["frame"] == "In which year did {S} visit {O}?"
    q3 = {"quid": 3, "question": "Who was the last to visit Iran?", "answers": ["Japan"], "answer_type": "entity",
          "qtype": "first_last", "time_level": "day"}     # Japan visited first: the 'last' cue contradicts the answer
    assert al.resolve(q3, al.hypotheses(q3))[2] == "none"
    q4 = dict(q3, question="Who was the first to visit Iran?")
    h, fr, status = al.resolve(q4, al.hypotheses(q4))
    assert status == "unique" and h["first"] is True and fr["frame"] == "Who was the first to visit {O}?"


def test_frame_lm_prefers_seen_placeholder_order():
    lm = FrameLM(["Who visited {O} in {T}?"] * 5 + ["Who did {S} visit in {T}?"] * 5)
    assert lm.score("Who negotiated with {O} in {T}?") > lm.score("Who negotiated with {S} in {T}?")
    assert "visited" in frame_words("Who visited {O} in 2005?")


def test_text_variant():
    assert text_variant({"qtype": "before_after", "answer_type": "entity", "question": "Who visited X before 2005?"}) == "date"
    assert text_variant({"qtype": "before_after", "answer_type": "entity", "question": "Who visited X before Y?"}) == "anchor"
    assert text_variant({"qtype": "equal_multi", "answer_type": "entity",
                         "question": "Who visited X in the same month as Y?"}) == "same"


# ---------------------------------------------------------------------- generation
def tiny_frozen():
    t = {"template_id": 0, "qtype": "after_first", "answer_type": "entity", "time_level": "day", "variant": "anchor",
         "relation": "Make a visit", "side": "subject", "frame": "After {Y}, who was the first to visit {O}?",
         "slots": [{"ph": "Y"}, {"ph": "O"}], "count": 3, "split_counts": {"dev": 3}, "usable": True,
         "status_major": "unique"}
    t2 = {"template_id": 1, "qtype": "equal", "answer_type": "time", "time_level": "year", "variant": "unique",
          "relation": "Make a visit", "side": "timestamp", "frame": "In which year did {S} visit {O}?",
          "slots": [{"ph": "S"}, {"ph": "O"}], "count": 1, "split_counts": {"dev": 1}, "usable": True,
          "status_major": "unique"}
    return {"templates": [t, t2], "weight_split": "dev", "demonyms": {},
            "bucket_distribution": {"after_first": {"entity|day|anchor": 1.0}, "equal": {"time|year|unique": 1.0}},
            "alias_forms": {"sector": {"of|sing|lower|the": 1}, "plain": {"-": 1}}}


def test_generator_answers_follow_the_procedure():
    kg = make_kg()
    g = Generator(kg, tiny_frozen(), seed=0, config={"anchor_single_day": False})
    fs = kg.facts_of(kg.rid["Make a visit"])
    t = tiny_frozen()["templates"][0]
    q = g.instantiate(t, fact=int(fs[2]))     # seed fact Japan visits Iran on 2010-03-02: anchor Japan, t0 = 2010-01-05
    assert q["question"] == "After Japan, who was the first to visit Iran?"
    assert q["answers"] == ["China"] and q["gen"]["anchor_date"] == "2010-01-05" and q["gen"]["answer_slot"] == "subject"
    assert q["gen"]["gold_fact_ids"] == [1]
    assert g.instantiate(t, fact=int(fs[4])) is None    # Obama's visit is the last one: no answer after it
    q = g.instantiate(tiny_frozen()["templates"][1], fact=int(fs[1]))
    assert q["question"] == "In which year did China visit Iran?" and q["answers"] == ["2010"]
    assert q["gen"]["answer_slot"] == "timestamp"
    qs = Generator(kg, tiny_frozen(), seed=1, config={"anchor_single_day": False}).generate({"after_first": 2}, 100)
    assert [q["quid"] for q in qs] == [100, 101] and len({q["question"] for q in qs}) == 2


def test_generator_v2_rules():
    kg = make_kg()
    g = Generator(kg, tiny_frozen(), seed=0)                    # v2 defaults
    fs = kg.facts_of(kg.rid["Make a visit"])
    t = tiny_frozen()["templates"][0]
    assert g.instantiate(t, fact=int(fs[2])) is None             # anchor Japan visited Iran on two days
    assert g.rejects["anchor_event_repeated"] == 1
    q = g.instantiate(t, fact=int(fs[1]))                        # anchor China (one day): next day 2010-03-02 has two
    assert q is None and g.rejects["not_unique_answer"] == 1     # visitors (Japan, France) -> not a unique answer
    lone = dict(t, frame="After {Y}, who was the first to visit {O}?")
    assert Generator(kg, tiny_frozen(), seed=0, config={"unique_answer": []}).instantiate(lone, fact=int(fs[1]))[
        "answers"] == ["Japan", "France"]


def test_degenerate_level_weight_moves_to_sibling_buckets():
    kg = make_kg()
    fr = tiny_frozen()
    t3 = dict(fr["templates"][1], template_id=2, time_level="month", frame="In which month did {S} visit {O}?")
    fr["templates"].append(t3)
    fr["bucket_distribution"]["equal"] = {"time|year|unique": 0.5, "time|month|unique": 0.25, "entity|year|period": 0.25}
    days = kg.D.copy()
    one_year = TemporalKG(kg.S, kg.R, kg.O, np.where(days > D("2010-12-31"), D("2010-12-30"), days), kg.entities,
                          kg.relations)
    g = Generator(one_year, fr, seed=0)
    assert g.degenerate_levels == ["year"]
    keys, w = g.bucket_p["equal"]
    assert keys == [("time", "month", "unique")] and w == [0.75]


def test_fill_capitalises_and_cleans():
    assert fill("{O} was visited by {S} ?", {"O": "the government of Nigeria", "S": "Iran"}) == \
        "The government of Nigeria was visited by Iran?"


# ---------------------------------------------------------------------- files
def test_prepare_renet(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "entity2id.txt").write_text("Iran\t0\nCitizen (Iran)\t1\n")
    (raw / "relation2id.txt").write_text("Make statement\t0\n")
    (raw / "train.txt").write_text("0\t0\t1\t0\t0\n1\t0\t0\t24\t0\n")
    (raw / "valid.txt").write_text("0\t0\t1\t48\t0\n")
    (raw / "test.txt").write_text("")
    import datetime as dt
    info = prepare_renet(raw, tmp_path / "out", dt.date(2018, 1, 1))
    assert info["facts"] == 3 and info["last_day"] == "2018-01-03"
    f = np.load(tmp_path / "out/facts.npz")
    assert set(f.files) == {"s_id", "r_id", "o_id", "t_id", "t_days"}
    assert (tmp_path / "out/sentences.txt").read_text().splitlines()[1] == "Citizen (Iran) Make statement Iran in 2018-01-02"
    kg = TemporalKG.from_cache(tmp_path / "out")
    assert kg.n_facts == 3 and list(kg.sro_days(0, 0, 1)) == [D("2018-01-01"), D("2018-01-03")]


def test_frozen_file_hash_is_checked(tmp_path):
    p = tmp_path / "t.json.gz"
    save_frozen({"templates": [], "x": 1}, p)
    assert load_frozen(p)["x"] == 1
    with gzip.open(p, "wb") as f:
        f.write(json.dumps({"templates": [], "x": 2}).encode())
    with pytest.raises(ValueError):
        load_frozen(p)


def test_generated_sets_use_the_multitq_format():
    assert is_multitq("icews18") and is_multitq("icews0515_regen") and is_multitq("multitq_orig")
    assert not is_multitq("icews") and not is_multitq("cron")


def test_role_signature_and_surface_fixes():
    from tear.data.qgen.generate import FormSampler, signature
    kg = make_kg()
    assert signature(kg, {"O": kg.eid["Iran"], "Y": kg.eid["Government (Nigeria)"]}) == "FT"
    assert signature(kg, {"S": kg.eid["Government (Nigeria)"], "O": kg.eid["France"]}) == "TF"
    fr = tiny_frozen()
    fr["role_signatures"] = None
    t = fr["templates"][0]
    g = Generator(kg, fr, seed=0, config={"anchor_single_day": False, "unique_answer": []})
    fs = kg.facts_of(kg.rid["Make a visit"])
    assert g.instantiate(t, fact=int(fs[1]), want_signature="FT") is None and g.rejects["role_signature"] == 1
    assert g.instantiate(t, fact=int(fs[1]), want_signature="FF") is not None
    assert aliases.plural_word("Affairs") == "Affairs" and aliases.plural_word("Business") == "businesses".capitalize()
    import random
    fs_ = FormSampler({"sector": {"of_the|sing|lower|the": 1}, "plain": {"-": 1},
                       "article_by_entity": {"Spain": 0.0, "Philippines": 1.0}}, random.Random(0), {})
    assert fs_.render("Citizen (Spain)")[0] == "the citizen of Spain"
    assert fs_.render("Citizen (Philippines)")[0] == "the citizen of the Philippines"
