"""Stage 6: score reader predictions in results/qwen/*.jsonl.

Default: evaluation files (no __bench / __tune suffix); --bench or --tune scores those instead. Outputs
results/multitq_qwen_per_question[_bench|_tune].csv and results/multitq_qwen_hit1[_bench|_tune].csv with Hit@1 exact /
lenient on the first listed answer, TimeR4-style any-match over the list, and Hit@1 exact per qtype, per
(setting, prompt, model). --baseline <setting> adds results/multitq_qwen_significance[...].csv: McNemar and paired
bootstrap against the baseline with the same prompt and model, on the questions both answered.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from tear.evaluation.hits import score_reader_record  # noqa: E402
from tear.evaluation.significance import mcnemar, paired_bootstrap  # noqa: E402
from tear.paths import RESULTS, ensure_dirs  # noqa: E402


def load_predictions(pred_dir: Path, bench: bool, tune: bool) -> pd.DataFrame:
    rows = []
    for p in sorted(pred_dir.glob("*.jsonl")):
        if ("__bench" in p.name) != bench or ("__tune" in p.name) != tune:
            continue
        parts = p.stem.split("__")
        setting = parts[0]
        prompt = next((x[2:] for x in parts if x.startswith("p-")), "timer4")
        for line in p.read_text().splitlines():
            r = json.loads(line)
            rows.append({"setting": setting, "prompt": prompt, "model": parts[1], "quid": r["quid"], "qtype": r["qtype"],
                         "qlabel": r["qlabel"], "answer_type": r["answer_type"], **score_reader_record(r)})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred_dir", default=None, help="default results/qwen")
    ap.add_argument("--results_dir", default=None, help="default results/")
    ap.add_argument("--bench", action="store_true", help="score the __bench files instead")
    ap.add_argument("--tune", action="store_true", help="score the __tune files instead")
    ap.add_argument("--baseline", default=None, help="setting name (file prefix before '__'), e.g. none_rule_exact_anydate")
    args = ap.parse_args()
    ensure_dirs()
    pred_dir = Path(args.pred_dir) if args.pred_dir else RESULTS / "qwen"
    results = Path(args.results_dir) if args.results_dir else RESULTS
    results.mkdir(parents=True, exist_ok=True)
    df = load_predictions(pred_dir, args.bench, args.tune)
    if df.empty:
        print("no predictions yet"); return
    tagf = "_bench" if args.bench else ("_tune" if args.tune else "")
    df.to_csv(results / f"multitq_qwen_per_question{tagf}.csv", index=False)
    keys = ["setting", "prompt", "model"]
    agg = df.groupby(keys).agg(n=("quid", "size"), hit1_exact=("hit1_exact", "mean"),
                               hit1_lenient=("hit1_lenient", "mean"),
                               hit_any_lenient=("hit_any_lenient", "mean")).reset_index()
    by_t = df.pivot_table(index=keys, columns="qtype", values="hit1_exact").reset_index()
    res = agg.merge(by_t, on=keys)
    res.to_csv(results / f"multitq_qwen_hit1{tagf}.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(res.round(4).to_string(index=False))
    if args.baseline:
        sig = []
        for (setting, prompt, model), g in df.groupby(keys):
            if setting == args.baseline:
                continue
            base = df[(df.setting == args.baseline) & (df.prompt == prompt) & (df.model == model)].set_index("quid").hit1_exact
            g = g.set_index("quid").hit1_exact
            common = base.index.intersection(g.index)
            if not len(common):
                continue
            a, b = base.loc[common].values, g.loc[common].values
            mc = mcnemar(a, b); bs = paired_bootstrap(a, b)
            sig.append({"setting": setting, "prompt": prompt, "model": model, "baseline": args.baseline,
                        "n": len(common), "diff": bs["diff"], "ci_low": bs["ci_low"], "ci_high": bs["ci_high"],
                        "bootstrap_p": bs["p_value"], "mcnemar_p": mc["p_value"]})
        sig = pd.DataFrame(sig)
        sig.to_csv(results / f"multitq_qwen_significance{tagf}.csv", index=False)
        print(sig.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
