"""TimelineKGQA reader inputs in the format scripts/07_read_with_llm.py reads (no change to the reader code).

For ds in icews, cron:
  cache/<ds>_rd/       questions.json (quid, question, rewrite = question, answers = [answer], qtype, qlabel = level,
                       answer_type), sentences.txt and rerank/{question_order.json, <setting>_<tag>_top30_ids.npy} as
                       symlinks to cache/<ds>/ (settings none, tear_window, hardwindow of scripts/16 --tag <tag>),
                       sample_1000.json: 1,000 test questions drawn at random (seed 13) from the grounding targets
                       (medium / complex, answer = subject or object); 07 reads them in its fixed shuffle (seed 2026),
                       so every prefix is a paired subset across settings. sample_1000_noleak.json: the same draw from
                       the targets whose question does not contain the answer string (every complex subject question
                       and 7-26% of the medium ones name their own answer - a generator artefact); each record carries
                       answer_in_question for a post-hoc split.
  cache/<ds>_rd_tune/  prompt-choice set from the validation split: 60 target questions (seed 7), their own 'none'
                       retrieval (all-mpnet-base-v2 on CPU, exact top-30 by cosine), tune_60.json.
The question text given to the reader is the original question.
"""
import argparse
import importlib.util
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from tear.paths import CACHE  # noqa: E402
from tear.utils.io import load_json, save_json  # noqa: E402

TARGET_LEVELS, TARGET_TYPES = ("medium", "complex"), ("subject", "object")
SETTINGS = ("none", "tear_window", "hardwindow")


def reader_record(q: dict) -> dict:
    return {"quid": str(q["qid"]), "question": q["question"], "rewrite": q["question"], "answers": [q["answer"]],
            "qtype": q["qtype"], "qlabel": q["level"], "answer_type": q["answer_type"], "level": q["level"],
            "timer4_top15_raw": [], "answer_in_question": q["answer"].lower() in q["question"].lower()}


def link(target: Path, name: Path) -> None:
    if name.is_symlink() or name.exists():
        name.unlink()
    name.symlink_to(os.path.relpath(target, name.parent))


def is_target(q: dict) -> bool:
    return q["level"] in TARGET_LEVELS and q["answer_type"] in TARGET_TYPES


def build_test(ds: str, tag: str, n: int, seed: int) -> None:
    src, out = CACHE / ds, CACHE / f"{ds}_rd"
    (out / "rerank").mkdir(parents=True, exist_ok=True)
    qs = load_json(src / "questions.json")
    order = [str(x) for x in load_json(src / "rerank" / "question_order.json")["question_ids"]]
    assert order == [str(q["qid"]) for q in qs], "question_order must follow questions.json"
    save_json([reader_record(q) for q in qs], out / "questions.json")
    link(src / "sentences.txt", out / "sentences.txt")
    link(src / "rerank" / "question_order.json", out / "rerank" / "question_order.json")
    for s in SETTINGS:
        arr = src / "rerank" / f"{s}_{tag}_top30_ids.npy"
        assert arr.exists(), f"run scripts/16_timeline_grounding.py --dataset {ds} --tag {tag} first ({arr})"
        link(arr, out / "rerank" / arr.name)
    pool = sorted(str(q["qid"]) for q in qs if is_target(q))
    sample = sorted(random.Random(seed).sample(pool, n), key=int)
    save_json({"seed": seed, "n": n, "population": "medium/complex questions with a subject or object answer",
               "quids": sample}, out / f"sample_{n}.json")
    clean = sorted(str(q["qid"]) for q in qs if is_target(q) and q["answer"].lower() not in q["question"].lower())
    noleak = sorted(random.Random(seed).sample(clean, n), key=int)
    save_json({"seed": seed, "n": n, "population": "the same targets, answer string not in the question",
               "quids": noleak}, out / f"sample_{n}_noleak.json")
    leak = {str(q["qid"]) for q in qs if q["answer"].lower() in q["question"].lower()}
    print(ds, "reader folder", out, "questions", len(qs), "targets", len(pool), "sample", len(sample),
          "of which answer in question", sum(k in leak for k in sample), "| no-leak targets", len(clean))


def build_tune(ds: str, tag: str, n: int, seed: int, threads: int) -> None:
    spec = importlib.util.spec_from_file_location("tl", Path(__file__).with_name("16_timeline_grounding.py"))
    tl = importlib.util.module_from_spec(spec); spec.loader.exec_module(tl)
    src, out = CACHE / ds, CACHE / f"{ds}_rd_tune"
    (out / "rerank").mkdir(parents=True, exist_ok=True)
    val = [q for q in tl.load_split(ds, "validation") if is_target(q) and q["gold_fact_ids"]
           and min(q["gold_fact_ids"]) >= 0]
    qs = sorted(random.Random(seed).sample(val, n), key=lambda q: int(q["qid"]))
    save_json([reader_record(q) for q in qs], out / "questions.json")
    link(src / "sentences.txt", out / "sentences.txt")
    qv = tl.encode([q["question"] for q in qs], threads)
    femb = np.load(src / "fact_emb.npy", mmap_mode="r")
    sims = np.zeros((len(qs), femb.shape[0]), np.float32)
    for lo in range(0, femb.shape[0], 50000):
        sims[:, lo:lo + 50000] = qv @ np.asarray(femb[lo:lo + 50000]).T
    top = np.argsort(-sims, axis=1, kind="stable")[:, :30].astype(np.int32)
    np.save(out / "rerank" / f"none_{tag}_top30_ids.npy", top)
    save_json({"question_ids": [str(q["qid"]) for q in qs]}, out / "rerank" / "question_order.json")
    save_json({"seed": seed, "n": n, "split": "validation", "quids": [str(q["qid"]) for q in qs]},
              out / f"tune_{n}.json")
    gold_in = np.mean([q["gold_fact_ids"][0] in top[i, :15] for i, q in enumerate(qs)])
    print(ds, "tune folder", out, "questions", len(qs), "answer event in top-15 (none):", round(float(gold_in), 3))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", default="icews,cron")
    ap.add_argument("--tag", default="rule_ground")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--n_tune", type=int, default=60)
    ap.add_argument("--tune_seed", type=int, default=7)
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    for ds in args.datasets.split(","):
        build_test(ds, args.tag, args.n, args.seed)
        build_tune(ds, args.tag, args.n_tune, args.tune_seed, args.threads)


if __name__ == "__main__":
    main()
