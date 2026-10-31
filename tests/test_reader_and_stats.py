import numpy as np

from tear.evaluation.hits import answer_granularity, score_reader_record
from tear.evaluation.significance import mcnemar, paired_bootstrap
from tear.reader.parse import parse_list
from tear.reader.prompt import build_prompt


def test_parse_list_formats():
    assert parse_list("['Jack Straw', 'Tony Blair']") == ["Jack Straw", "Tony Blair"]
    assert parse_list("1. Japan\n2. China") == ["Japan", "China"]
    assert parse_list("- 2012-05") == ["2012-05"]
    assert parse_list("[Japan, China]") == ["Japan", "China"]


def test_prompt_and_granularity():
    p = build_prompt(["Iraq Make a visit Japan in 2006-01-07"], "Who visited Japan?")
    assert p.endswith("Question:\n Who visited Japan?") and "Historical facts:['Iraq" in p
    assert answer_granularity("In which year did X visit Y?") == 4
    assert answer_granularity("In which month did X visit Y?") == 7
    assert answer_granularity("When did X visit Y?") == 10


def test_score_reader_record():
    r = {"prediction": "['Japan', 'China']", "answers": ["China"]}
    assert score_reader_record(r) == {"hit1_exact": False, "hit1_lenient": False, "hit_any_lenient": True}


def test_mcnemar():
    a = np.array([1] * 10 + [0] * 30, bool)
    b = np.array([0] * 10 + [1] * 30, bool)      # 30 discordant improvements, 10 regressions
    r = mcnemar(a, b)
    assert r["n01"] == 30 and r["n10"] == 10 and r["method"] == "chi2_cc"
    assert abs(r["statistic"] - (19 ** 2) / 40) < 1e-12 and r["p_value"] < 0.01
    small = mcnemar(np.array([1, 0, 0], bool), np.array([0, 1, 1], bool))
    assert small["method"] == "exact" and abs(small["p_value"] - 1.0) < 1e-12
    assert mcnemar(a, a)["p_value"] == 1.0


def test_paired_bootstrap():
    rng = np.random.default_rng(0)
    a = rng.random(2000) < 0.5
    b = a | (rng.random(2000) < 0.2)
    r = paired_bootstrap(a, b, n_boot=2000)
    assert r["diff"] > 0 and r["ci_low"] > 0 and r["p_value"] < 0.01
    same = paired_bootstrap(a, a, n_boot=200)
    assert same["diff"] == 0 and same["p_value"] == 1.0


def test_v2_prompt_numbers_facts():
    p = build_prompt(["A Make a visit B in 2006-01-07", "B Host a visit A in 2006-01-07"], "Who visited B?", "v2")
    assert p.startswith("Historical facts (each line: subject, relation, object, date):\n1. A Make a visit B")
    assert "\n2. B Host a visit A in 2006-01-07\n\nQuestion: Who visited B?\n" in p


def test_select_questions_is_a_fixed_shuffle():
    from tear.reader.mlx_reader import select_questions
    sample = [str(i) for i in range(100)]
    first = select_questions(sample, 0, 0, 10)
    assert first == select_questions(sample, 0, 0, 10) and first != sample[:10]
    assert select_questions(sample, 0, 0, None)[10:20] == select_questions(sample, 0, 10, 20)
    assert select_questions(sample, 10) == sample[::10][:10]
