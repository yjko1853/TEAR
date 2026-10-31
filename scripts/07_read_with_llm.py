"""Stage 5b: local LLM reader (MLX, Apple silicon) over each setting's top-k facts; resumable jsonl output.

  --make_sample                     cache/multitq/sample_<n>.json (n_per questions per qtype, seed 13)
  --make_tune                       cache/multitq/tune_<n>.json, disjoint from the evaluation sample (prompt choice)
  --setting <reranker> --tag <tag>  read cache/<dataset>/rerank/<reranker>_<tag>_top30_ids.npy
  --dataset multitq_orig            original questions (anchor grounding run); questions, rerank arrays and sample
                                    are read from cache/<dataset>/ (default multitq)
  --setting timer4_public           TimeR4's released top-15 (its fine-tuned retriever), same reader
  --prompt timer4 | v2              reader prompt (tear/reader/prompt.py)
  --start / --end                   slice of the fixed shuffled sample (run all settings chunk by chunk)
  --limit 30                        speed check (separate __bench file)
Run with the MLX environment, e.g. .venv-mlx/bin/python -I scripts/07_read_with_llm.py ...
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tear.paths import CACHE, RESULTS, ensure_dirs  # noqa: E402
from tear.reader.mlx_reader import DEFAULT_MODEL, make_sample, make_tune, run_reader  # noqa: E402
from tear.reader.prompt import PROMPTS  # noqa: E402
from tear.utils.io import load_json, save_json  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--make_sample", action="store_true")
    ap.add_argument("--make_tune", action="store_true")
    ap.add_argument("--n_per", type=int, default=None, help="questions per qtype (default 1000 sample / 20 tune)")
    ap.add_argument("--seed", type=int, default=None, help="default 13 sample / 7 tune")
    ap.add_argument("--sample_out", default=None, help="output path for --make_sample / --make_tune")
    ap.add_argument("--setting", default="tear_window")
    ap.add_argument("--tag", default="rule_exact_anydate")
    ap.add_argument("--sample", default="sample_6000.json")
    ap.add_argument("--questions", default="questions.json", help="question file (questions_v2.json: query field)")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--k", type=int, default=15)
    ap.add_argument("--max_tokens", type=int, default=48)
    ap.add_argument("--prompt", choices=list(PROMPTS), default="timer4")
    ap.add_argument("--batch", type=int, default=1, help="1 = sequential generate; >1 = mlx_lm.batch_generate")
    ap.add_argument("--start", type=int, default=0, help="slice of the shuffled sample (inclusive)")
    ap.add_argument("--end", type=int, default=None, help="slice of the shuffled sample (exclusive)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--pred_dir", default=None, help="prediction jsonl directory (default results/qwen)")
    ap.add_argument("--dataset", default="multitq", help="cache/<dataset>/ holds the questions, rerank arrays and sample")
    ap.add_argument("--slots", default=None, help="prompt v3g: rule labels whose timestamp slot gets the granularity line "
                    "(default cache/<dataset>/slots_rule_v2.json)")
    args = ap.parse_args()
    ensure_dirs()
    d = CACHE / args.dataset
    if args.make_sample or args.make_tune:
        qs = load_json(d / "questions.json")
        if args.make_sample:
            s = make_sample(qs, args.n_per or 1000, 13 if args.seed is None else args.seed)
            default = d / f"sample_{len(s['quids'])}.json"
        else:
            used = set(load_json(d / args.sample)["quids"])
            s = make_tune(qs, used, args.n_per or 20, 7 if args.seed is None else args.seed, args.sample)
            default = d / f"tune_{len(s['quids'])}.json"
        out = Path(args.sample_out) if args.sample_out else default
        save_json(s, out)
        print("size", len(s["quids"]), "->", out)
        return
    pred_dir = Path(args.pred_dir) if args.pred_dir else RESULTS / "qwen"
    gran = None
    if args.prompt == "v3g":
        from tear.data.multitq import stage_query
        from tear.labeling.granularity import asked_granularity
        slots = load_json(Path(args.slots) if args.slots else d / "slots_rule_v2.json")
        gran = {str(q["quid"]): asked_granularity(stage_query(q), slots.get(str(q["quid"]), {}).get("answer_slot"))
                for q in load_json(d / args.questions)}
    run_reader(d, pred_dir, args.setting, args.tag, args.sample, args.model, args.k, args.max_tokens, args.prompt,
               args.batch, args.limit, args.start, args.end,
               questions_file=args.questions, granularity=gran)


if __name__ == "__main__":
    main()
