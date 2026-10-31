"""Stage 0: prepare a dataset (fact table, fact sentences, test questions) under cache/<dataset>/.

--dataset multitq_orig (after multitq): original MultiTQ questions instead of TimeR4's rewrites, sharing the
MultiTQ fact files and embeddings through symlinks.
--dataset multitq_bge (after multitq_orig): the main setting with a second retriever (BAAI/bge-base-en-v1.5); questions
and labels are symlinks, the stages 02 / 03 / 14 / 05 / 06 are rerun with --model.
--dataset multitq --make_query (after multitq): cache/multitq/questions_v2.json with a 'query' field = the TimeR4
rewrite when valid, otherwise the original question (empty or truncated rewrites).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tear.data import multitq, timelinekgqa  # noqa: E402
from tear.paths import CACHE, ensure_dirs  # noqa: E402
from tear.utils.timing import timed  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=["multitq", "multitq_orig", "multitq_bge", "icews", "cron"], required=True)
    ap.add_argument("--out", default=None, help="output directory (default cache/<dataset>)")
    ap.add_argument("--kg", default=str(multitq.DEFAULT_KG), help="MultiTQ: KG full.txt")
    ap.add_argument("--test", default=str(multitq.DEFAULT_TEST), help="MultiTQ: test questions")
    ap.add_argument("--prompt", default=str(multitq.DEFAULT_PROMPT), help="MultiTQ: TimeR4 test_prompt.json")
    ap.add_argument("--datadir", default=str(timelinekgqa.DEFAULT_DATADIR), help="TimelineKGQA: Datasets/ folder")
    ap.add_argument("--split", default="test", help="TimelineKGQA split")
    ap.add_argument("--make_query", action="store_true", help="MultiTQ: write questions_v2.json (query field) only")
    ap.add_argument("--questions_out", default="questions_v2.json", help="--make_query output file name")
    args = ap.parse_args()
    ensure_dirs()
    out = Path(args.out) if args.out else CACHE / args.dataset
    if args.make_query:
        if args.dataset != "multitq":
            raise SystemExit("--make_query applies to --dataset multitq")
        with timed("S0_make_query", args.dataset):
            multitq.make_query_questions(out, "questions.json", args.questions_out)
        return
    with timed("S0_prepare", args.dataset):
        if args.dataset == "multitq":
            multitq.prepare(args.kg, args.test, args.prompt, out)
        elif args.dataset == "multitq_orig":
            multitq.make_original_variant(CACHE / "multitq", out)
        elif args.dataset == "multitq_bge":
            multitq.make_encoder_variant(CACHE / "multitq_orig", out)
        else:
            timelinekgqa.prepare(args.dataset, args.datadir, args.split, out)


if __name__ == "__main__":
    main()
