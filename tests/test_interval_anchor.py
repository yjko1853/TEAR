import numpy as np
import pytest

from tear.labeling.interval_anchor import IntervalAnchorResolver, allen_mask, allen_of, parse_constraints, read_answer

# validation-split questions (TimelineKGQA ICEWS-Actor / CronQuestions-KG)
CASES = [
    ("Who Affiliation To Far Left while Takis Kliridis Affiliation To Ministry of Finance?",
     [("during", {"d", "di", "o", "oi"}, None, "Takis Kliridis Affiliation To Ministry of Finance")]),
    ("Before Borjana Kristo Affiliation To Elite (Bosnia and Herzegovina), which organisation is Affiliation To by Nikitas Kaklamanis?",
     [("before", {"<"}, None, "Borjana Kristo Affiliation To Elite (Bosnia and Herzegovina), which organisation is Affiliation To by Nikitas Kaklamanis")]),
    ("Martín Lousteau Affiliation To which organisation, 2043 days after Mladen Bajić Affiliation To Municipal / District Court (Croatia), 4272 days after Khondokar Mahbubuddin Ahmed Affiliation To Government Major Party (In Government) (Bangladesh)?",
     [("days_after", None, ("after", 2043), "Mladen Bajić Affiliation To Municipal / District Court (Croatia)"),
      ("days_after", None, ("after", 4272), "Khondokar Mahbubuddin Ahmed Affiliation To Government Major Party (In Government) (Bangladesh)")]),
    ("Who academic degree Doctor of Science 35429 days {'relation': 'X < Y', 'description': 'X is before Y', 'category': 'tp', 'code': 'tp-24', 'semantic': 'before'} Karl Wilhelm Verhoeff award received Silver Leibniz medal?",
     [("days_dict", None, ("before", 35429), "Karl Wilhelm Verhoeff award received Silver Leibniz medal")]),
    ("Who Affiliation To Labour Party, at the same time Mirza Judith Meza Cardona Arreaga Affiliation To National Unity of Hope?",
     [("same_time", {"=", "s", "si", "f", "fi", "d", "di", "o", "oi"}, None, "Mirza Judith Meza Cardona Arreaga Affiliation To National Unity of Hope")]),
    ("Keriako Tobiko Affiliation To which organisation, finishedby Blaz Kavcic Affiliation To National Council, finishedby Thein Nyunt Affiliation To Provincial (Myanmar)?",
     [("finishedby", {"fi"}, None, "Blaz Kavcic Affiliation To National Council"),
      ("finishedby", {"fi"}, None, "Thein Nyunt Affiliation To Provincial (Myanmar)")]),
]


@pytest.mark.parametrize("question,expected", CASES)
def test_parse_constraints(question, expected):
    got = [(c["kind"], c["relations"], c["offset"], c["clause"]) for c in parse_constraints(question)]
    assert got == expected


def test_no_constraint_in_simple_questions():
    assert parse_constraints("When did Babatunde Osotimehin end Affiliation To United Nations Population Fund?") == []


def test_allen_relations_are_exclusive_and_exhaustive():
    # proper intervals (start < end); a point interval (start = end, CronQuestions-KG years) can satisfy two
    # relations at once, and allen_of then returns the first in ALLEN order
    assert allen_of(5, 5, 2, 5) == "mi"
    rng = np.random.default_rng(0)
    for _ in range(300):
        xs, ys = rng.integers(0, 6, 2); xe, ye = xs + rng.integers(1, 4), ys + rng.integers(1, 4)
        hits = [r for r in ("<", ">", "=", "m", "mi", "s", "si", "f", "fi", "d", "di", "o", "oi")
                if allen_mask(r, np.array([xs]), np.array([xe]), ys, ye)[0]]
        assert len(hits) == 1 and allen_of(xs, xe, ys, ye) == hits[0]


# toy KG: X candidates of Far Left, anchor (Takis, AT, Ministry) with two intervals
ENTS = ["A", "B", "Far Left", "Takis Kliridis", "Ministry of Finance"]
PREDS = ["Affiliation To"]
FACTS = {"s_id": np.array([0, 1, 3, 3]), "p_id": np.zeros(4, int), "o_id": np.array([2, 2, 4, 4]),
         "start_days": np.array([5, 50, 10, 100]), "end_days": np.array([30, 60, 20, 110])}


def test_resolve_and_window_any_interval():
    res = IntervalAnchorResolver(FACTS, ENTS, PREDS)
    cons = res.resolve("Who Affiliation To Far Left while Takis Kliridis Affiliation To Ministry of Finance?")
    assert cons[0]["anchors"] == [2, 3]
    # A [5, 30] contains [10, 20] (di); B [50, 60] relates to neither anchor interval by d/di/o/oi
    assert res.window(cons, FACTS["start_days"][:2], FACTS["end_days"][:2]).tolist() == [True, False]
    assert res.anchor_entities(cons) == {3, 4}
    off = res.resolve("Who Affiliation To Far Left, 30 days after Takis Kliridis Affiliation To Ministry of Finance?")
    assert res.window(off, np.array([50, 51]), np.array([60, 60])).tolist() == [True, False]   # 50 - 20 = 30


def test_read_answer():
    assert read_answer("subject", 1, FACTS, ENTS) == "B" and read_answer("object", 1, FACTS, ENTS) == "Far Left"
    assert read_answer("duration", 1, FACTS, ENTS) is None
