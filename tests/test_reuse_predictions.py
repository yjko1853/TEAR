import importlib.util
import json
from pathlib import Path

import numpy as np

SPEC = importlib.util.spec_from_file_location(
    "reuse", Path(__file__).resolve().parents[1] / "scripts" / "12_reuse_reader_predictions.py")
reuse = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reuse)


def _fixture(tmp_path):
    d = tmp_path / "multitq"; (d / "rerank").mkdir(parents=True)
    (d / "sentences.txt").write_text("\n".join(f"fact {i}" for i in range(6)) + "\n")
    qs = [{"quid": 1, "question": "Who visited Iraq?", "rewrite": "Who visited Iraq?", "timer4_top15_raw": ["x"]},
          {"quid": 2, "question": "Who praised China?", "rewrite": "", "timer4_top15_raw": ["y"]},
          {"quid": 3, "question": "Who met Japan?", "rewrite": "Who met Japan?", "timer4_top15_raw": ["z"]}]
    (d / "questions.json").write_text(json.dumps(qs))
    v2 = [dict(q, query=q["rewrite"] or q["question"]) for q in qs]
    (d / "questions_v2.json").write_text(json.dumps(v2))
    (d / "rerank" / "question_order.json").write_text(json.dumps({"question_ids": [1, 2, 3]}))
    old = np.array([[0, 1], [2, 3], [4, 5]], np.int32)
    new = np.array([[0, 1], [2, 3], [5, 4]], np.int32)          # question 3: different fact order
    np.save(d / "rerank" / "none_old_top30_ids.npy", old)
    np.save(d / "rerank" / "none_new_top30_ids.npy", new)
    src = tmp_path / "qwen"; src.mkdir()
    name = "none_old__Model__k2__p-v2.jsonl"
    (src / name).write_text("".join(json.dumps({"quid": str(q), "prediction": f"['a{q}']", "answers": ["a"]}) + "\n"
                                    for q in (1, 2, 3)))
    return d, src, name


def test_reuse_copies_only_identical_inputs(tmp_path):
    d, src, name = _fixture(tmp_path)
    dst = tmp_path / "qwen_v2"
    r = reuse.reuse_file(src / name, d, dst, "old", "new", "questions.json", "questions_v2.json")
    assert r["identical_input"] == 1 and r["copied"] == 1          # 2: query changed, 3: facts changed
    out = dst / "none_new__Model__k2__p-v2.jsonl"
    assert [json.loads(l)["quid"] for l in out.read_text().splitlines()] == ["1"]
    again = reuse.reuse_file(src / name, d, dst, "old", "new", "questions.json", "questions_v2.json")
    assert again["copied"] == 0 and again["already_present"] == 1   # idempotent


def test_dry_run_and_name_parsing(tmp_path):
    d, src, name = _fixture(tmp_path)
    r = reuse.reuse_file(src / name, d, tmp_path / "qwen_v2", "old", "new", "questions.json", "questions_v2.json",
                         dry_run=True)
    assert r["copied"] == 0 and not (tmp_path / "qwen_v2").exists()
    assert reuse.parse_name("tear_window_rule_exact_anydate__Qwen__k15__p-v3__tune.jsonl", "rule_exact_anydate")[:4] == \
        ("tear_window", "Qwen", 15, "v3")
    assert reuse.parse_name("timer4_public__Qwen__k15__p-v2.jsonl", "x")[0] == "timer4_public"
    assert reuse.parse_name("none_other_tag__Qwen__k15__p-v2.jsonl", "rule_exact_anydate") is None


def test_reuse_across_datasets(tmp_path):
    # new run on another dataset (multitq_orig): its own questions and rerank arrays; only identical prompts are copied
    d, src, name = _fixture(tmp_path)
    d2 = tmp_path / "multitq_orig"; (d2 / "rerank").mkdir(parents=True)
    (d2 / "sentences.txt").write_text((d / "sentences.txt").read_text())
    qs = json.loads((d / "questions.json").read_text())
    orig = [dict(q, rewrite_timer4=q["rewrite"], rewrite=q["question"]) for q in qs]
    orig[0]["rewrite"] = "Who visited Iraq first?"                      # question 1: original wording differs
    (d2 / "questions.json").write_text(json.dumps(orig))
    (d2 / "rerank" / "question_order.json").write_text(json.dumps({"question_ids": [1, 2, 3]}))
    np.save(d2 / "rerank" / "none_ground_top30_ids.npy", np.array([[0, 1], [2, 3], [4, 5]], np.int32))
    r = reuse.reuse_file(src / name, d, tmp_path / "qwen_ground", "old", "ground", "questions_v2.json",
                         "questions.json", d_new=d2)
    assert r["identical_input"] == 2 and r["copied"] == 2                # 2 and 3 have the same text and facts
    out = tmp_path / "qwen_ground" / "none_ground__Model__k2__p-v2.jsonl"
    assert [json.loads(l)["quid"] for l in out.read_text().splitlines()] == ["2", "3"]
