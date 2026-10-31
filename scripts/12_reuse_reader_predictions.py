"""Reuse reader predictions across question/label versions without re-running the LLM.

For every prediction file in --src (default results/qwen/; __bench files skipped unless --include_bench) whose setting
used --old_tag (or is timer4_public), each record is copied to the corresponding file in --dst (default
results/qwen_v2/, setting renamed to --new_tag) only when the reader's full prompt is identical under the new version:
the same top-k fact list (old vs new rerank arrays) and the same question text (old: stage text of --old_questions;
new: stage text of --new_questions, i.e. the 'query' field). Decoding is greedy, so identical input gives identical
output (runs must use the same model, k, prompt version and batch size).

Existing destination files are appended to, skipping questions already present. --dry_run only counts.
  .venv/bin/python -I scripts/12_reuse_reader_predictions.py --dry_run
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tear.data.multitq import stage_query  # noqa: E402
from tear.paths import CACHE, RESULTS  # noqa: E402
from tear.reader.mlx_reader import facts_for  # noqa: E402
from tear.reader.prompt import PROMPTS, build_prompt  # noqa: E402
from tear.utils.io import load_json  # noqa: E402


def parse_name(name: str, old_tag: str):
    """'<setting>[_<tag>]__<model>__k<k>__p-<prompt>[__bench][__tune].jsonl' -> (setting, model, k, prompt, rest) or None."""
    parts = name[:-len(".jsonl")].split("__")
    prefix, model = parts[0], parts[1]
    k = next((int(x[1:]) for x in parts if x.startswith("k") and x[1:].isdigit()), None)
    prompt = next((x[2:] for x in parts if x.startswith("p-")), "timer4")
    if prefix == "timer4_public":
        setting = prefix
    elif prefix.endswith("_" + old_tag):
        setting = prefix[:-len(old_tag) - 1]
    else:
        return None
    return setting, model, k, prompt, parts


def new_name(parts, setting: str, new_tag: str) -> str:
    head = setting if setting == "timer4_public" else f"{setting}_{new_tag}"
    return "__".join([head] + parts[1:]) + ".jsonl"


def reuse_file(path: Path, d: Path, dst: Path, old_tag: str, new_tag: str, old_questions: str, new_questions: str,
               dry_run: bool = False, d_new: Path | None = None) -> dict:
    """d: cache directory of the old run; d_new: of the new run when it is another dataset (e.g. multitq_orig)."""
    d_new = d_new or d
    meta = parse_name(path.name, old_tag)
    if meta is None:
        return {"file": path.name, "status": "skipped (other tag)"}
    setting, model, k, prompt, parts = meta
    if prompt not in PROMPTS or k is None:
        return {"file": path.name, "status": f"skipped (prompt {prompt} / k {k} unknown)"}
    recs = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    keys = [r["quid"] for r in recs]
    old_q = {str(q["quid"]): q for q in load_json(d / old_questions)}
    new_q = {str(q["quid"]): q for q in load_json(d_new / new_questions)}
    f_old = facts_for(d, setting, old_tag, k, keys)
    f_new = facts_for(d_new, setting, new_tag, k, keys)
    out = dst / new_name(parts, setting, new_tag)
    have = set()
    if out.exists():
        have = {json.loads(line)["quid"] for line in out.read_text().splitlines() if line.strip()}
    same = [r for r in recs
            if build_prompt(f_old[r["quid"]], stage_query(old_q[r["quid"]]), prompt)
            == build_prompt(f_new[r["quid"]], stage_query(new_q[r["quid"]]), prompt)]
    todo = [r for r in same if r["quid"] not in have]
    if not dry_run and todo:
        dst.mkdir(parents=True, exist_ok=True)
        with out.open("a") as fo:
            for r in todo:
                fo.write(json.dumps(r, ensure_ascii=False) + "\n")
    return {"file": path.name, "to": out.name, "records": len(recs), "identical_input": len(same),
            "copied": 0 if dry_run else len(todo), "already_present": len(same) - len(todo), "status": "ok"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="multitq")
    ap.add_argument("--new_dataset", default=None, help="dataset of the new run if different (e.g. multitq_orig: original"
                    " questions with anchor grounding); --dataset is then the old run's")
    ap.add_argument("--prompt", default=None, help="only prediction files with this prompt version (e.g. v3)")
    ap.add_argument("--exclude_tune", action="store_true", help="skip __tune files")
    ap.add_argument("--src", default=None, help="default results/qwen")
    ap.add_argument("--dst", default=None, help="default results/qwen_v2")
    ap.add_argument("--old_tag", default="rule_exact_anydate")
    ap.add_argument("--new_tag", default="rule_v2_exact_anydate")
    ap.add_argument("--old_questions", default="questions.json")
    ap.add_argument("--new_questions", default="questions_v2.json")
    ap.add_argument("--include_bench", action="store_true")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()
    d = CACHE / args.dataset
    d_new = CACHE / args.new_dataset if args.new_dataset else None
    src = Path(args.src) if args.src else RESULTS / "qwen"
    dst = Path(args.dst) if args.dst else RESULTS / "qwen_v2"
    if src.resolve() == dst.resolve():
        raise SystemExit("--dst must differ from --src")
    for p in sorted(src.glob("*.jsonl")):
        if "__bench" in p.name and not args.include_bench:
            continue
        if ("__tune" in p.name and args.exclude_tune) or (args.prompt and f"__p-{args.prompt}" not in p.name):
            continue
        if args.prompt and not p.name.split(f"__p-{args.prompt}")[1].startswith((".", "__")):
            continue
        print(reuse_file(p, d, dst, args.old_tag, args.new_tag, args.old_questions, args.new_questions, args.dry_run,
                         d_new))


if __name__ == "__main__":
    main()
