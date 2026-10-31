"""Stage 5a: LLM-free Hit@1 on MultiTQ - the answer is read off the top-1 reranked fact (no LLM, no API).

Output: results/<ds>_llmfree_hit1_<tag>.csv; with --baseline also
results/<ds>_llmfree_significance_<tag>.csv (McNemar + paired bootstrap of each reranker vs the baseline).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tear.evaluation.hits import llmfree_rows, summarize_llmfree  # noqa: E402
from tear.evaluation.significance import mcnemar, paired_bootstrap  # noqa: E402
from tear.paths import CACHE, RESULTS, ensure_dirs  # noqa: E402
from tear.utils.io import load_json  # noqa: E402

RERANKERS = ["none", "timer4", "hardwindow", "atc", "tr", "tear", "tear_window"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="multitq", help="multitq or a variant such as multitq_orig")
    ap.add_argument("--questions", default="questions.json", help="question file in cache/<ds>/ (e.g. questions_v2.json)")
    ap.add_argument("--tag", default="rule_exact_anydate")
    ap.add_argument("--slots", default=None, help="default cache/<ds>/slots_rule.json")
    ap.add_argument("--rerank_dir", default=None, help="input top-k arrays (default cache/<ds>/rerank)")
    ap.add_argument("--results_dir", default=None, help="output directory (default results/)")
    ap.add_argument("--baseline", default=None, help="e.g. none: also write paired significance vs this reranker")
    args = ap.parse_args()
    ensure_dirs()
    ds = args.dataset
    d = CACHE / ds
    rerank_dir = Path(args.rerank_dir) if args.rerank_dir else d / "rerank"
    results = Path(args.results_dir) if args.results_dir else RESULTS
    results.mkdir(parents=True, exist_ok=True)
    qs = load_json(d / args.questions); slots = load_json(args.slots or d / "slots_rule.json")
    order = load_json(rerank_dir / "question_order.json")
    keys = [str(k) for k in (order["question_ids"] if isinstance(order, dict) else order)]
    qby = {str(q["quid"]): q for q in qs}
    facts = np.load(d / "facts.npz"); voc = load_json(d / "vocab.json")
    rows = []
    for rk in RERANKERS:
        p = rerank_dir / f"{rk}_{args.tag}_top30_ids.npy"
        if not p.exists():
            print("missing", p); continue
        rows += llmfree_rows(np.load(p)[:, 0], keys, qby, slots, facts, voc["entities"], voc["timestamps"], rk)
    df = pd.DataFrame(rows, columns=["reranker", "qtype", "qlabel", "answer_type", "exact", "lenient"])
    res = summarize_llmfree(df)
    res.to_csv(results / f"{ds}_llmfree_hit1_{args.tag}.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    print(res.round(4).to_string(index=False))
    if args.baseline:
        n = len(keys)
        base = df[df.reranker == args.baseline].exact.values
        sig = []
        for rk, g in df.groupby("reranker", sort=False):
            if rk == args.baseline:
                continue
            assert len(g) == n
            mc = mcnemar(base, g.exact.values); bs = paired_bootstrap(base, g.exact.values)
            sig.append({"reranker": rk, "baseline": args.baseline, "diff": bs["diff"], "ci_low": bs["ci_low"],
                        "ci_high": bs["ci_high"], "bootstrap_p": bs["p_value"], "mcnemar_p": mc["p_value"],
                        "n01": mc["n01"], "n10": mc["n10"]})
        sig = pd.DataFrame(sig)
        sig.to_csv(results / f"{ds}_llmfree_significance_{args.tag}.csv", index=False)
        print(sig.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
