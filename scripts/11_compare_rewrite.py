"""Ablation: original MultiTQ questions (multitq_orig) vs TimeR4's gpt-3.5 rewrites (multitq), same retriever and rules.

Run after stages 03-06 for both datasets with the same tag. Per-question answer coverage@15 and LLM-free Hit@1 are
recomputed from the saved top-30 arrays (checked against the stage 05/06 CSVs), so the comparison can also be split
into the questions the rewrite changed and those it left verbatim.
Outputs (results/, prefix <variant>_vs_rewrite_):
  summary_<tag>.csv        reranker x question subset (all / changed / unchanged): coverage@15 and Hit@1, both variants
  qtype_<tag>.csv          reranker x qtype: the same metrics
  constraints_<tag>.csv    constraint recognition (cue, target date, window, TR fired) per qtype, both variants
  significance_<tag>.csv   paired McNemar + bootstrap of Hit@1 (variant - rewrite) per reranker
  retrieval_agreement.csv  unchanged questions: identical top-1 / top-15 / top-1000 between the two retrieval runs
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tear.analysis.runs import RERANKERS, check_against_csv, load_run  # noqa: E402
from tear.evaluation.significance import mcnemar, paired_bootstrap  # noqa: E402
from tear.paths import CACHE, RESULTS  # noqa: E402
from tear.utils.io import load_json  # noqa: E402

CR_COLS = ["cue_first", "cue_last", "cue_before", "cue_after", "target_date", "window", "tr_fired", "atc_penalised_any"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="multitq", help="dataset with the TimeR4 rewrites")
    ap.add_argument("--variant", default="multitq_orig", help="dataset with the original questions")
    ap.add_argument("--tag", default="rule_exact_anydate", help="reranker tag of both runs (outputs are named by it)")
    ap.add_argument("--base_tag", default=None, help="base run tag if it differs from --tag")
    ap.add_argument("--base_questions", default="questions.json", help="e.g. questions_v2.json ('query' field)")
    ap.add_argument("--base_slots", default="slots_rule.json")
    ap.add_argument("--variant_slots", default="slots_rule.json")
    ap.add_argument("--base_top", default="top1000_ids.npy", help="base retrieval ids (retrieval agreement)")
    ap.add_argument("--results_dir", default=None, help="output directory (default results/)")
    args = ap.parse_args()
    btag = args.base_tag or args.tag
    out = Path(args.results_dir) if args.results_dir else RESULTS
    out.mkdir(parents=True, exist_ok=True)
    pre = f"{args.variant}_vs_rewrite_"

    order_b, qs_b, per_b = load_run(args.base, btag, args.base_questions, args.base_slots)
    order_v, qs_v, per_v = load_run(args.variant, args.tag, "questions.json", args.variant_slots)
    assert order_b == order_v
    check_against_csv(args.base, btag, per_b); check_against_csv(args.variant, args.tag, per_v)
    changed = np.array([qb["rewrite"] != qv["rewrite"] for qb, qv in zip(qs_b, qs_v)])
    qtype = np.array([q["qtype"] for q in qs_b])
    print(f"{len(order_b)} questions, changed by the rewrite: {changed.sum()}")

    subsets = {"all": np.ones(len(changed), bool), "changed": changed, "unchanged": ~changed}
    rows, qrows, sig = [], [], []
    for rk in RERANKERS:
        b, v = per_b[rk], per_v[rk]
        for name, m in subsets.items():
            rows.append({"reranker": rk, "subset": name, "n": int(m.sum()),
                         "coverage15_rewrite": b["cov15"][m].mean(), "coverage15_orig": v["cov15"][m].mean(),
                         "hit1_rewrite": b["hit1"][m].mean(), "hit1_orig": v["hit1"][m].mean()})
        for qt in sorted(set(qtype)):
            m = qtype == qt
            qrows.append({"reranker": rk, "qtype": qt, "n": int(m.sum()), "n_changed": int((m & changed).sum()),
                          "coverage15_rewrite": b["cov15"][m].mean(), "coverage15_orig": v["cov15"][m].mean(),
                          "hit1_rewrite": b["hit1"][m].mean(), "hit1_orig": v["hit1"][m].mean()})
        mc = mcnemar(b["hit1"], v["hit1"]); bs = paired_bootstrap(b["hit1"], v["hit1"])
        sig.append({"reranker": rk, "metric": "llmfree_hit1_exact", "diff_orig_minus_rewrite": bs["diff"],
                    "ci_low": bs["ci_low"], "ci_high": bs["ci_high"], "bootstrap_p": bs["p_value"],
                    "mcnemar_p": mc["p_value"], "n01": mc["n01"], "n10": mc["n10"]})
    summ = pd.DataFrame(rows); qt = pd.DataFrame(qrows)
    for df in (summ, qt):
        df["coverage15_diff"] = df.coverage15_orig - df.coverage15_rewrite
        df["hit1_diff"] = df.hit1_orig - df.hit1_rewrite
    summ.to_csv(out / f"{pre}summary_{args.tag}.csv", index=False)
    qt.to_csv(out / f"{pre}qtype_{args.tag}.csv", index=False)
    pd.DataFrame(sig).to_csv(out / f"{pre}significance_{args.tag}.csv", index=False)

    crb = pd.read_csv(RESULTS / f"{args.base}_constraint_recognition_{btag}.csv").set_index("qtype")
    crv = pd.read_csv(RESULTS / f"{args.variant}_constraint_recognition_{args.tag}.csv").set_index("qtype")
    cr = pd.concat({"rewrite": crb[["n"] + CR_COLS], "orig": crv[CR_COLS]}, axis=1)
    cr.columns = [f"{c}_{v}" if c != "n" else "n" for v, c in cr.columns]
    cr.reset_index().to_csv(out / f"{pre}constraints_{args.tag}.csv", index=False)

    ib = np.load(CACHE / args.base / args.base_top, mmap_mode="r")
    iv = np.load(CACHE / args.variant / "top1000_ids.npy", mmap_mode="r")
    u = np.flatnonzero(~changed)
    pos = {k: i for i, k in enumerate(str(q["quid"]) for q in load_json(CACHE / args.base / args.base_questions))}
    rows_u = np.array([pos[order_b[i]] for i in u])        # question_order equals questions.json order in practice
    a, c = np.asarray(ib[rows_u]), np.asarray(iv[rows_u])
    agree = {"n_unchanged": len(u), "top1_identical": float((a[:, 0] == c[:, 0]).mean()),
             "top15_identical_order": float((a[:, :15] == c[:, :15]).all(1).mean()),
             "top15_identical_set": float(np.mean([set(x) == set(y) for x, y in zip(a[:, :15], c[:, :15])])),
             "top1000_identical_order": float((a == c).all(1).mean())}
    suffix = "" if args.tag == "rule_exact_anydate" and args.base_top == "top1000_ids.npy" else f"_{args.tag}"
    pd.DataFrame([agree]).to_csv(out / f"{pre}retrieval_agreement{suffix}.csv", index=False)

    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    print(summ.round(4).to_string(index=False)); print(pd.DataFrame(sig).round(4).to_string(index=False))
    print(cr.round(3).to_string()); print(agree)


if __name__ == "__main__":
    main()
