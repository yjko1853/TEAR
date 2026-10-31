import json

import pytest

from tear.data import is_multitq
from tear.data.multitq import SHARED_FILES, make_original_variant, original_question_variant

QS = [
    {"quid": 1, "question": "After the Danish Ministry of Defence and Security, who was the first to visit Iraq?",
     "rewrite": "After 2006-01-05, who was the first to visit Iraq?", "answers": ["Jack Straw"]},
    {"quid": 2, "question": "When did China praise Iraq?", "rewrite": "When did China praise Iraq?", "answers": ["2007"]},
]


def test_is_multitq():
    assert is_multitq("multitq") and is_multitq("multitq_orig")
    assert not is_multitq("icews") and not is_multitq("cron")


def test_original_question_variant():
    out = original_question_variant(QS)
    assert [q["rewrite"] for q in out] == [q["question"] for q in QS]
    assert [q["rewrite_timer4"] for q in out] == [q["rewrite"] for q in QS]
    assert QS[0]["rewrite"].startswith("After 2006-01-05")          # input not mutated
    assert all(o["answers"] == q["answers"] and o["quid"] == q["quid"] for o, q in zip(out, QS))


def test_make_original_variant(tmp_path):
    src = tmp_path / "multitq"; src.mkdir()
    for name in SHARED_FILES:
        (src / name).write_text(name)
    (src / "questions.json").write_text(json.dumps(QS))
    out = tmp_path / "multitq_orig"
    make_original_variant(src, out)
    for name in SHARED_FILES:
        assert (out / name).is_symlink() and (out / name).read_text() == name
    assert json.loads((out / "questions.json").read_text())[0]["rewrite"] == QS[0]["question"]
    assert json.loads((src / "questions.json").read_text()) == QS               # source untouched
    make_original_variant(src, out)                                               # idempotent
    with pytest.raises(ValueError):
        make_original_variant(src, src)
