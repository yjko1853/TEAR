"""Stage 4: rerankers on the shared top-N pool; answer coverage@k, ablation, lambda sweep, diagnostics.

Rerankers: none / timer4 / timer4_sym / chrono / hardwindow / atc / tr / tear / tear_window (see tear/rerank).
Outputs (results/): <ds>_coverage_main_<tag>.csv, <ds>_ablation_<tag>.csv, <ds>_lambda_sweep_<tag>.csv,
<ds>_constraint_recognition_<tag>.csv, <ds>_slot_occupancy_top15_<tag>.csv, <ds>_timer4_public_top15.csv (MultiTQ);
cache/<ds>/rerank/<reranker>_<tag>_top30_ids.npy (+ question_order.json) for the reader.
'_anydate' is appended to the tag when --date_granularity any (the parser version is not part of the tag).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tear.data import is_multitq  # noqa: E402
from tear.data.candidates import CandidatePool  # noqa: E402
from tear.evaluation.coverage import answer_hit_features, coverage_rows, timer4_public_coverage  # noqa: E402
from tear.paths import RESULTS, ensure_dirs  # noqa: E402
from tear.rerank.core import RERANKERS, rerank  # noqa: E402
from tear.rerank.features import build_features  # noqa: E402
from tear.utils.io import load_json, save_json  # noqa: E402
from tear.utils.timing import timed  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--topn", type=int, default=1000)
    ap.add_argument("--slots", default=None, help="JSON quid->{answer_slot, topic_entity}; default cache/<ds>/slots_oracle.json")
    ap.add_argument("--topics", default=None, help="optional JSON overriding topic_entity (e.g. NER output)")
    ap.add_argument("--tag", default="oracle", help="name suffix for outputs")
    ap.add_argument("--lam1", type=float, default=0.8)
    ap.add_argument("--lam2", type=float, default=0.8)
    ap.add_argument("--mu", type=float, default=0.4)
    ap.add_argument("--chrono_pool", type=int, default=30)
    ap.add_argument("--date_granularity", choices=["full", "any"], default="full")
    ap.add_argument("--date_parser", choices=["v1", "v2"], default="v1",
                    help="with --date_granularity any: v1 = first extended parser, v2 = also 'Sep, 2011', 'Sept 2011', 'Sep 5, 2011'")
    ap.add_argument("--sweep", action="store_true", help="lambda1 x lambda2 grid over {0.5..1.0}")
    ap.add_argument("--save_topk", type=int, default=30)
    ap.add_argument("--atc_match", choices=["substring", "exact"], default="substring",
                    help="substring = original TEAR rule (topic in slot value or vice versa); exact = entity identity")
    ap.add_argument("--results_dir", default=None, help="CSV output directory (default results/)")
    ap.add_argument("--rerank_dir", default=None, help="top-k id arrays output directory (default cache/<ds>/rerank)")
    ap.add_argument("--questions", default="questions.json", help="question file in cache/<ds>/ (e.g. questions_v2.json)")
    ap.add_argument("--suffix", default="", help="candidate arrays top<N><suffix>_{ids,scores}.npy (e.g. _v2)")
    ap.add_argument("--anchors", default=None, help="anchor dates JSON (scripts/14_ground_anchors.py) for event-anchored"
                    " before/after/same-period constraints")
    ap.add_argument("--equal_window", action="store_true", help="window + TR validity for same-period constraints")
    args = ap.parse_args()
    ensure_dirs()
    ds = args.dataset
    pool = CandidatePool(ds, args.topn, questions_file=args.questions, suffix=args.suffix)
    results = Path(args.results_dir) if args.results_dir else RESULTS
    rerank_dir = Path(args.rerank_dir) if args.rerank_dir else pool.d / "rerank"
    results.mkdir(parents=True, exist_ok=True); rerank_dir.mkdir(parents=True, exist_ok=True)
    slots = load_json(args.slots) if args.slots else load_json(pool.d / "slots_oracle.json")
    topics = load_json(args.topics) if args.topics else None
    tag = args.tag + ("" if args.date_granularity == "full" else "_anydate")
    with timed("S5_features", ds, tag):
        F = build_features(pool, slots, topics, args.date_granularity, args.atc_match, args.date_parser,
                           load_json(args.anchors) if args.anchors else None, args.equal_window)
        F.update(answer_hit_features(pool))
    qdf = pd.DataFrame(pool.qs)
    if is_multitq(ds):
        groups = {"overall": pd.Series(["all"] * len(qdf)), "qtype": qdf["qtype"], "answer_type": qdf["answer_type"],
                  "qlabel": qdf["qlabel"], "time_level": qdf["time_level"]}
        gkey = "qtype"
    else:
        groups = {"overall": pd.Series(["all"] * len(qdf)), "answer_type": qdf["answer_type"], "level": qdf["level"],
                  "qtype": qdf["qtype"]}
        gkey = "answer_type"

    # ---- constraint recognition: which rules fired
    meta = F["meta"].copy(); meta[gkey] = qdf[gkey].values
    cr = meta.groupby(gkey).agg(n=("slot", "size"), slot_available=("slot", lambda s: s.notna().mean()),
                                topic_available=("topic", lambda s: s.notna().mean()),
                                cue_first=("has_cue_first", "mean"), cue_last=("has_cue_last", "mean"),
                                cue_before=("has_cue_before", "mean"), cue_after=("has_cue_after", "mean"),
                                target_date=("has_target_date", "mean"), window=("has_window", "mean"),
                                tr_fired=("tr_fired", "mean"), atc_penalised_any=("atc_any", "mean")).reset_index()
    if "anchor_used" in meta.columns:
        extra = meta.groupby(gkey).agg(anchor_used=("anchor_used", "mean"), period_window=("has_period", "mean"),
                                       anchor_atc=("anchor_atc", "mean"))
        cr = cr.merge(extra.reset_index(), on=gkey)
    cr.to_csv(results / f"{ds}_constraint_recognition_{tag}.csv", index=False)
    print(cr.to_string())

    # ---- main comparison + slot occupancy
    rows, occ_rows = [], []
    with timed("S5_rerank_main", ds, tag):
        for name in RERANKERS:
            perm = rerank(name, pool.sims, F, args.lam1, args.lam2, args.mu, args.chrono_pool)
            rows += coverage_rows(F, perm, name, groups)
            np.save(rerank_dir / f"{name}_{tag}_top{args.save_topk}_ids.npy",
                    np.take_along_axis(pool.ids, perm[:, :args.save_topk], axis=1).astype(np.int32))
            # share of top-15 candidates whose answer-slot value is the topic entity
            occ = np.take_along_axis(F["atc"], perm[:, :15], axis=1).mean(1)
            avail = meta["slot"].notna() & meta["topic"].notna()
            for gval in sorted(qdf[gkey].unique()):
                m = (qdf[gkey] == gval).values & avail.values
                occ_rows.append({"reranker": name, gkey: gval, "n": int(m.sum()),
                                 "topic_in_answer_slot_share_top15": float(occ[m].mean()) if m.any() else np.nan,
                                 "questions_with_any_in_top15": float((occ[m] > 0).mean()) if m.any() else np.nan})
            m = avail.values
            occ_rows.append({"reranker": name, gkey: "ALL", "n": int(m.sum()),
                             "topic_in_answer_slot_share_top15": float(occ[m].mean()),
                             "questions_with_any_in_top15": float((occ[m] > 0).mean())})
        for P in (100, args.topn):   # chronological-sort pool variants
            perm = rerank("chrono", pool.sims, F, P=P)
            rows += coverage_rows(F, perm, f"chrono_P{P}", groups)
    df = pd.DataFrame(rows)
    df.to_csv(results / f"{ds}_coverage_main_{tag}.csv", index=False)
    pd.DataFrame(occ_rows).to_csv(results / f"{ds}_slot_occupancy_top15_{tag}.csv", index=False)
    abl = df[df.reranker.isin(["none", "atc", "tr", "tear"])].copy()
    abl["setting"] = abl.reranker.map({"none": "base", "atc": "+ATC", "tr": "+TR", "tear": "+ATC+TR"})
    abl.to_csv(results / f"{ds}_ablation_{tag}.csv", index=False)
    piv = df[(df.k == 15) & (df.group_type.isin(["overall", gkey]))].pivot_table(index="reranker", columns="group",
                                                                                values="coverage")
    print("\ncoverage@15 (pivot):\n", piv.round(4).to_string())
    order = {"question_ids": [q[pool.qkey] for q in pool.qs]}
    qo = rerank_dir / "question_order.json"
    if not qo.exists():
        save_json(order, qo)
    elif load_json(qo) != order:   # never replace an order other runs depend on
        save_json(order, rerank_dir / f"question_order_{tag}.json")
        print(f"question order differs from {qo}; wrote question_order_{tag}.json", flush=True)

    # ---- reference: TimeR4's released top-15 (MultiTQ)
    if is_multitq(ds):
        covs = timer4_public_coverage(pool)
        trows = [{"group_type": "overall", "group": "all", "n": len(covs), "coverage@15": float(covs.mean())}]
        for gtype in ("qtype", "answer_type", "qlabel"):
            for gval in sorted(qdf[gtype].unique()):
                m = (qdf[gtype] == gval).values
                trows.append({"group_type": gtype, "group": gval, "n": int(m.sum()), "coverage@15": float(covs[m].mean())})
        # The TimeR4 reference does not depend on the reranker; keep an existing table unless the content differs.
        text = pd.DataFrame(trows).to_csv(index=False)
        tpath = results / f"{ds}_timer4_public_top15.csv"
        if not tpath.exists():
            tpath.write_text(text)
        elif tpath.read_text() != text:
            (results / f"{ds}_timer4_public_top15_{tag}.csv").write_text(text)
        print("\nTimeR4 public top-15 coverage@15:", float(covs.mean()))

    # ---- lambda sweep
    if args.sweep:
        grid = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        srows = []
        with timed("S5_lambda_sweep", ds, tag):
            for l1 in grid:
                for l2 in grid:
                    perm = rerank("tear", pool.sims, F, l1, l2)
                    sub = {"overall": groups["overall"], gkey: groups[gkey]}
                    if is_multitq(ds):
                        sub["answer_type"] = groups["answer_type"]
                    srows += coverage_rows(F, perm, "tear", sub, ks=[5, 10, 15, 20, 30],
                                           extra={"lambda1": l1, "lambda2": l2})
        sw = pd.DataFrame(srows)
        sw.to_csv(results / f"{ds}_lambda_sweep_{tag}.csv", index=False)
        print("\nlambda sweep (overall coverage@15):\n",
              sw[(sw.k == 15) & (sw.group == "all")].pivot(index="lambda1", columns="lambda2", values="coverage").round(4).to_string())


if __name__ == "__main__":
    main()
