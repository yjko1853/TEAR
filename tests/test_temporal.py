from tear.temporal import (date_to_days, days_to_date, extract_from_to_dates, extract_target_dates,
                           extract_target_dates_extended, extract_temporal_cue, target_days)


def test_cues_are_substrings():
    assert extract_temporal_cue("After 2006-01-05, who was the first to visit Iraq?") == ["first", "after"]
    assert extract_temporal_cue("Who visited Iraq?") == []


def test_original_regex_only_iso_dates():
    assert extract_target_dates("before 2014-10-21, who ...") == [{"type": "before", "date": "2014-10-21"}]
    assert extract_target_dates("after 12 May 2011, which country ...") == []


def _dates(q):
    return [(x["type"], x["date"]) for x in extract_target_dates_extended(q)]


def test_extended_parser_forms():
    assert _dates("After 12 May 2011, which country did Cambodia denounce?") == [("after", "2011-05-12")]
    assert _dates("before May 2011") == [("before", "2011-05-01")]
    assert _dates("after May 2011") == [("after", "2011-05-31")]
    assert _dates("after 2011-05 who") == [("after", "2011-05-31")]
    assert _dates("before 2011 who") == [("before", "2011-01-01")]
    assert _dates("after the year 2011") == [("after", "2011-12-31")]
    assert _dates("before China, who") == []


def test_target_days_and_round_trip():
    td = target_days("Before 2007-10-31, which country did China threaten last?", "full")
    assert td == [("before", date_to_days("2007-10-31"))]
    assert days_to_date(date_to_days("2007-10-31")) == "2007-10-31"


def test_from_to():
    assert extract_from_to_dates("Who was affiliated to X from 2005-01-01 to end of time?") == ("2005-01-01", "end of time")
    assert extract_from_to_dates("Who visited Iraq?") is None


# --------------------------------------------------------------------------- date parser v2
import pytest  # noqa: E402

from tear.temporal import extract_target_dates_v2  # noqa: E402

V2_FORMS = [
    ("After Sep, 2011, which country did Thailand negotiate with?", [("after", "2011-09-30")]),
    ("Who wanted to cooperate with Ethiopia before Sep 2011?", [("before", "2011-09-01")]),
    ("after Sep. 2011 who", [("after", "2011-09-30")]),
    ("after Sept 2011 who", [("after", "2011-09-30")]),
    ("Before Aug, 2006, who wanted to cooperate with Ethiopia?", [("before", "2006-08-01")]),
    ("Which country did Cambodia request for after Jun 2, 2014?", [("after", "2014-06-02")]),
    ("before 5 Sep 2011 who", [("before", "2011-09-05")]),
    ("Who requested for China after December 1st, 2015?", [("after", "2015-12-01")]),
    ("After November, 2006, who received Comoros's visit?", [("after", "2006-11-30")]),
    ("after Dec, 2012 who", [("after", "2012-12-31")]),
    # forms the v1 extended parser already read are unchanged
    ("After 12 May 2011, which country did Cambodia denounce?", [("after", "2011-05-12")]),
    ("before May 2011", [("before", "2011-05-01")]),
    ("after 2011-05 who", [("after", "2011-05-31")]),
    ("before the year 2011", [("before", "2011-01-01")]),
    ("Before 2007-10-31, which country did China threaten last?", [("before", "2007-10-31")]),
    ("before China, who", []),
    ("before Feb 30, 2011", []),                    # invalid day: no date rather than a guess
]


@pytest.mark.parametrize("question,expected", V2_FORMS)
def test_parser_v2_forms(question, expected):
    assert [(x["type"], x["date"]) for x in extract_target_dates_v2(question)] == expected


def test_parser_v1_and_full_mode_unchanged():
    q = "After Sep, 2011, which country did Thailand negotiate with?"
    assert extract_target_dates_extended(q) == []                     # v1: not read (kept for reproduction)
    assert target_days(q, "any") == [] and target_days(q, "any", "v1") == []
    assert target_days(q, "any", "v2") == [("after", date_to_days("2011-09-30"))]
    assert target_days("before Sep 2011", "full", "v2") == []          # original TEAR regex mode ignores the parser
