"""Anchor-grounding report: original questions with KG-resolved event anchors vs no grounding vs TimeR4's rewrites.

Run after scripts/14_ground_anchors.py and logs/build_anchor.sh. Outputs (results/anchor_grounding_*.csv):
  accuracy       anchored questions per qtype and anchor type: resolved share, agreement with the date in TimeR4's
                 rewrite (day / month / year and at the granularity the question needs) and, where the two dates
                 differ, the share for which each date yields a gold answer in the KG (same relation and slot layout)
  summary        run x reranker x group: answer coverage@15 and LLM-free Hit@1 (exact); groups all, qtype, anchored
  significance   paired McNemar + bootstrap of LLM-free Hit@1 between runs, per reranker
  reader_reuse   reader sample (cache/multitq/sample_6000.json): per setting, questions whose reader input (question
                 text and top-15 fact sentences) is unchanged between the running reader (rewrite v2) and a new run
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tear.analysis.runs import RERANKERS, check_against_csv, load_run, stage_text  # noqa: E402
from tear.evaluation.significance import mcnemar, paired_bootstrap  # noqa: E402
from tear.labeling.anchor import AnchorResolver  # noqa: E402
from tear.paths import CACHE, RESULTS  # noqa: E402
from tear.temporal import date_to_days, extract_temporal_cue  # noqa: E402
from tear.utils.io import load_json  # noqa: E402

ORIG, RW = "multitq_orig", "multitq"
RWDIR = CACHE / ORIG / "rerank_on_rewrite"
# name: (dataset, question file, labels, tag, rerank dir)
RUNS = {
    "original": (ORIG, "questions.json", "slots_rule_v2.json", "rule_v2_exact_anydate", None),
    "original+anchors": (ORIG, "questions.json", "slots_rule_v2.json", "rule_v2_anchor_exact_anydate", None),
    "original+grounding": (ORIG, "questions.json", "slots_rule_v2.json", "rule_v2_ground_exact_anydate", None),
    "rewrite": (RW, "questions_v2.json", "slots_rule_v2.json", "rule_v2_exact_anydate", None),
    "rewrite+equal_window": (RW, "questions_v2.json", "slots_rule_v2.json", "rule_v2_eqwin_exact_anydate", RWDIR),
    "rewrite+grounding": (RW, "questions_v2.json", "slots_rule_v2.json", "rule_v2_ground_exact_anydate", RWDIR),
}
PAIRS = [("original+grounding", "original"), ("original+grounding", "rewrite+equal_window"),
         ("original+grounding", "rewrite"), ("original+anchors", "original"), ("rewrite+grounding", "rewrite+equal_window")]
NEED = {"before": 10, "after": 10, "day": 10, "month": 7, "year": 4}


def accuracy_table(anchors, qby, slots):
    d = CACHE / ORIG
    voc = load_json(d / "vocab.json")
    res = AnchorResolver(np.load(d / "facts.npz"), voc["entities"], voc["relations"])
    rows = []
    df = pd.DataFrame([dict(a, quid=k) for k, a in anchors.items()])
    for (qt, typ), g in df.groupby(["qtype", "type"]):
        ok = g[g.status == "resolved"]
        both = ok[ok.timer4_date.notna() & (ok.timer4_date.str.len() == 10)]
        need = NEED[typ]
        same = both.date.str[:need] == both.timer4_date.str[:need]
        diff = both[~same]
        cons_ours = cons_t4 = 0
        for _, a in diff.iterrows():
            q, sl = qby[a.quid], slots[a.quid]
            cues = extract_temporal_cue(q["question"])
            for ref, acc in ((a.date, "o"), (a.timer4_date, "t")):
                hit = bool(res.implied_answers(sl["answer_slot"], sl["topic_entity"], a.relation, typ,
                                               date_to_days(ref), cues) & set(q["answers"]))
                if acc == "o":
                    cons_ours += hit
                else:
                    cons_t4 += hit
        rows.append({"qtype": qt, "anchor_type": typ, "n_anchored": len(g), "resolved": len(ok) / len(g),
                     "n_compared": len(both),
                     **{f"agree_{u}": (both.date.str[:n] == both.timer4_date.str[:n]).mean() if len(both) else np.nan
                        for u, n in (("day", 10), ("month", 7), ("year", 4))},
                     "agree_needed_granularity": same.mean() if len(both) else np.nan, "n_differ": len(diff),
                     "differ_gold_consistent_ours": cons_ours / len(diff) if len(diff) else np.nan,
                     "differ_gold_consistent_timer4": cons_t4 / len(diff) if len(diff) else np.nan})
    return pd.DataFrame(rows)


def reader_reuse(per_old, qs_old, order, per_new, qs_new, sample):
    sents = (CACHE / RW / "sentences.txt").read_text().splitlines()
    pos = {k: i for i, k in enumerate(order)}
    idx = [pos[str(k)] for k in sample]
    rows = []
    for rk in ["none", "tear_window", "timer4", "tear", "atc", "tr", "timer4_public"]:
        same = 0
        for i in idx:
            if stage_text(qs_old[i]) != stage_text(qs_new[i]):
                continue
            if rk == "timer4_public":
                same += 1                                    # TimeR4's released top-15 does not depend on the run
                continue
            a = [sents[j] for j in per_old[rk]["top"][i, :15]]
            b = [sents[j] for j in per_new[rk]["top"][i, :15]]
            same += a == b
        rows.append({"setting": rk, "n_sample": len(idx), "reusable": same, "to_rerun": len(idx) - same})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results_dir", default=None, help="output directory (default results/)")
    args = ap.parse_args()
    out = Path(args.results_dir) if args.results_dir else RESULTS
    out.mkdir(parents=True, exist_ok=True)

    anchors = load_json(CACHE / ORIG / "anchors.json")
    qs_orig = load_json(CACHE / ORIG / "questions.json")
    qby = {str(q["quid"]): q for q in qs_orig}
    acc = accuracy_table(anchors, qby, load_json(CACHE / ORIG / "slots_rule_v2.json"))
    acc.to_csv(out / "anchor_grounding_accuracy.csv", index=False)

    runs = {}
    for name, (ds, qf, sf, tag, rdir) in RUNS.items():
        order, qs, per = load_run(ds, tag, qf, sf, rdir)
        check_against_csv(ds, tag, per)
        runs[name] = (order, qs, per)
    order = runs["original"][0]
    assert all(r[0] == order for r in runs.values())
    qtype = np.array([qby[k]["qtype"] for k in order])
    anchored = np.array([k in anchors for k in order])
    resolved = np.array([anchors.get(k, {}).get("status") == "resolved" for k in order])
    groups = {"all": np.ones(len(order), bool), "anchored": anchored, "anchored_resolved": resolved,
              "not_anchored": ~anchored}
    groups.update({qt: qtype == qt for qt in sorted(set(qtype))})
    groups.update({f"anchored:{qt}": anchored & (qtype == qt) for qt in sorted(set(qtype))})
    rows = []
    for name, (_, _, per) in runs.items():
        for rk in RERANKERS:
            for g, m in groups.items():
                if m.any():
                    rows.append({"run": name, "reranker": rk, "group": g, "n": int(m.sum()),
                                 "coverage15": per[rk]["cov15"][m].mean(), "hit1": per[rk]["hit1"][m].mean()})
    summ = pd.DataFrame(rows)
    summ.to_csv(out / "anchor_grounding_summary.csv", index=False)

    sig = []
    for a, b in PAIRS:
        for rk in RERANKERS:
            x, y = runs[b][2][rk]["hit1"], runs[a][2][rk]["hit1"]
            mc = mcnemar(x, y); bs = paired_bootstrap(x, y)
            sig.append({"run": a, "vs": b, "reranker": rk, "diff": bs["diff"], "ci_low": bs["ci_low"],
                        "ci_high": bs["ci_high"], "bootstrap_p": bs["p_value"], "mcnemar_p": mc["p_value"],
                        "vs_wrong_run_right": mc["n01"], "vs_right_run_wrong": mc["n10"]})
    pd.DataFrame(sig).to_csv(out / "anchor_grounding_significance.csv", index=False)

    sample = load_json(CACHE / RW / "sample_6000.json")["quids"]
    reuse = reader_reuse(runs["rewrite"][2], runs["rewrite"][1], order, runs["original+grounding"][2],
                         runs["original+grounding"][1], sample)
    reuse.to_csv(out / "anchor_grounding_reader_reuse.csv", index=False)

    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(acc.round(3).to_string(index=False))
    piv = summ[summ.group == "all"].pivot(index="reranker", columns="run", values="hit1").reindex(RERANKERS)
    print(piv.round(4).to_string())
    print(pd.DataFrame(sig).round(4).to_string(index=False))
    print(reuse.to_string(index=False))


if __name__ == "__main__":
    main()
