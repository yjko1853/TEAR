from tear.labeling.granularity import asked_granularity, asked_unit, project
from tear.reader.prompt import build_prompt


def test_asked_unit():
    assert asked_unit("In which year did Iraq first visit China?") == "year"
    assert asked_unit("What month did X praise Y?") == "month"
    assert asked_unit("When did X last visit Y?") == "day"


def test_asked_granularity_needs_timestamp_slot():
    assert asked_granularity("In which year did X visit Y?", "timestamp") == "year"
    assert asked_granularity("Who visited Y in the same year as X?", "subject") is None


def test_project():
    assert project("2010-07-13", "year") == "2010"
    assert project("['2010-07-13']", "month") == "2010-07"
    assert project("2010", "day") == "2010"          # too short for the unit: unchanged
    assert project("Barack Obama", "year") == "Barack Obama"


def test_v3g_equals_v3_without_granularity_and_adds_one_line():
    facts = ["Iraq Make a visit China 2006-01-05"]
    q = "In which year did Iraq visit China?"
    assert build_prompt(facts, q, "v3g") == build_prompt(facts, q, "v3")
    g = build_prompt(facts, q, "v3g", "year")
    extra = set(g.splitlines()) - set(build_prompt(facts, q, "v3").splitlines())
    assert extra == {"- This question asks for a year: answer with the year only, written as YYYY."}
