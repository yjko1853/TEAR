"""Compare two label/question versions of the MultiTQ pipeline (default v1 = rule_exact_anydate on questions.json,
v2 = rule_v2_exact_anydate on questions_v2.json).

Writes results/multitq_<new>_vs_<old>_{coverage,llmfree,labels,reuse}.csv and prints markdown tables:
  coverage   answer coverage@15 per reranker (overall, per qtype, Multiple / Single, entity / time)
  llmfree    LLM-free Hit@1 (exact) per reranker
  labels     answer-slot and topic-entity changes between the two label files
  fixes      slot changes attributed to the query fallback and to each rule-v2 fix (a-c, d, e; tear/labeling/rules.py),
             applied cumulatively on the questions whose text is unchanged
  dates      effect of date parser v2 within the new version: constraint recognition and coverage@15 / LLM-free Hit@1
             of the new tag vs its parser-v1 snapshot (results/multitq_*_<new>_dateparser_v1.csv), when present
--addendum PATH also writes the markdown addendum (results/addendum_rule_v2.md).
  reuse      share of questions (all, and the reader sample) whose reader input is unchanged per setting:
             same top-15 facts and same stage text -> an existing prediction can be reused (scripts/12)
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tear.data.multitq import stage_query  # noqa: E402
from tear.labeling.rules import rule_slot_fixes  # noqa: E402
from tear.paths import CACHE, RESULTS  # noqa: E402
from tear.utils.io import load_json  # noqa: E402

RERANKERS = ["none", "timer4", "hardwindow", "atc", "tr", "tear", "tear_window"]
COLS = ["all", "equal", "before_after", "first_last", "equal_multi", "after_first", "before_last", "Multiple", "Single",
        "entity", "time"]


def coverage15(tag):
    df = pd.read_csv(RESULTS / f"multitq_coverage_main_{tag}.csv")
    df = df[(df.k == 15) & df.reranker.isin(RERANKERS)]
    df["col"] = np.where(df.group_type == "overall", "all", df.group)
    t = df[df.col.isin(COLS)].pivot_table(index="reranker", columns="col", values="coverage", aggfunc="first")
    return t.reindex(RERANKERS)[[c for c in COLS if c in t.columns]]


def md(df, fmt="{:.4f}"):
    cols = [df.index.name or ""] + [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for i, r in df.iterrows():
        lines.append("| " + " | ".join([str(i)] + [fmt.format(v) if isinstance(v, float) else str(v) for v in r.values]) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--old_tag", default="rule_exact_anydate")
    ap.add_argument("--new_tag", default="rule_v2_exact_anydate")
    ap.add_argument("--old_questions", default="questions.json")
    ap.add_argument("--new_questions", default="questions_v2.json")
    ap.add_argument("--old_slots", default="slots_rule.json")
    ap.add_argument("--new_slots", default="slots_rule_v2.json")
    ap.add_argument("--sample", default="sample_6000.json")
    ap.add_argument("--addendum", default=None, help="also write the markdown addendum to this path")
    ap.add_argument("--dates_snapshot", default="_dateparser_v1", help="suffix of the parser-v1 snapshot of the new tag")
    args = ap.parse_args()
    d = CACHE / "multitq"
    pre = RESULTS / f"multitq_{args.new_tag}_vs_{args.old_tag}"

    c_old, c_new = coverage15(args.old_tag), coverage15(args.new_tag)
    cov = pd.concat({"v1": c_old, "v2": c_new, "diff": c_new - c_old}, axis=1)
    cov.to_csv(f"{pre}_coverage.csv")

    h_old = pd.read_csv(RESULTS / f"multitq_llmfree_hit1_{args.old_tag}.csv").set_index("reranker")
    h_new = pd.read_csv(RESULTS / f"multitq_llmfree_hit1_{args.new_tag}.csv").set_index("reranker")
    hcols = ["exact_all", "exact_Multiple", "exact_Single", "exact_entity", "exact_time"]
    hit = pd.concat({"v1": h_old[hcols], "v2": h_new[hcols], "diff": h_new[hcols] - h_old[hcols]}, axis=1).reindex(RERANKERS)
    hit.to_csv(f"{pre}_llmfree.csv")

    qo, qn = load_json(d / args.old_questions), load_json(d / args.new_questions)
    so, sn = load_json(d / args.old_slots), load_json(d / args.new_slots)
    rows = []
    for a, b in zip(qo, qn):
        k = str(a["quid"])
        rows.append({"qtype": a["qtype"], "query_changed": stage_query(a) != stage_query(b),
                     "slot_old": so[k]["answer_slot"], "slot_new": sn[k]["answer_slot"],
                     "topic_changed": so[k]["topic_entity"] != sn[k]["topic_entity"]})
    lab = pd.DataFrame(rows)
    lab["slot_changed"] = lab.slot_old != lab.slot_new
    trans = lab[lab.slot_changed].groupby(["slot_old", "slot_new"]).size().rename("n").reset_index()
    summary = pd.DataFrame([{"questions": len(lab), "query_changed": int(lab.query_changed.sum()),
                             "slot_changed": int(lab.slot_changed.sum()), "topic_changed": int(lab.topic_changed.sum()),
                             "slot_changed_with_same_query": int((lab.slot_changed & ~lab.query_changed).sum())}])
    summary.to_csv(f"{pre}_labels.csv", index=False)
    trans.to_csv(f"{pre}_slot_transitions.csv", index=False)

    # attribution: query fallback, then the rule fixes applied cumulatively on unchanged texts
    att = [{"step": "query fallback (380 empty/truncated rewrites)", "questions": int(lab.query_changed.sum()),
            "slot_changes": int((lab.slot_changed & lab.query_changed).sum()), "transitions": ""}]
    same_text = [stage_query(b) for a, b in zip(qo, qn) if stage_query(a) == stage_query(b)]
    prev = [rule_slot_fixes(t, frozenset()) for t in same_text]
    for name, fx in (("rule fixes a-c (do-support position, 'in which', 'with which' clause)", "abc"),
                     ("rule fix d (date with a comma in the leading clause)", "abcd"),
                     ("rule fix e ('suffer from' asks for the victim)", "abcde")):
        cur = [rule_slot_fixes(t, frozenset(fx)) for t in same_text]
        ch = pd.Series([f"{a}->{b}" for a, b in zip(prev, cur) if a != b]).value_counts()
        att.append({"step": name, "questions": len(same_text), "slot_changes": int(ch.sum()),
                    "transitions": ", ".join(f"{k} {v}" for k, v in ch.items())})
        prev = cur
    att = pd.DataFrame(att)
    att.to_csv(f"{pre}_fix_attribution.csv", index=False)

    order = [str(x) for x in load_json(d / "rerank" / "question_order.json")["question_ids"]]
    pos = {k: i for i, k in enumerate(order)}
    sample = [str(x) for x in load_json(d / args.sample)["quids"]]
    srows = np.array([pos[k] for k in sample])
    same_q = np.array([stage_query(a) == stage_query(b) for a, b in zip(qo, qn)])
    qrow = np.array([pos[str(q["quid"])] for q in qo]); same_q_ord = np.empty(len(order), bool); same_q_ord[qrow] = same_q
    reuse = []
    for rk in RERANKERS + ["timer4_public"]:
        if rk == "timer4_public":
            same = same_q_ord
        else:
            A = np.load(d / "rerank" / f"{rk}_{args.old_tag}_top30_ids.npy", mmap_mode="r")[:, :15]
            B = np.load(d / "rerank" / f"{rk}_{args.new_tag}_top30_ids.npy", mmap_mode="r")[:, :15]
            same = (np.asarray(A) == np.asarray(B)).all(1) & same_q_ord
        reuse.append({"setting": rk, "reusable_all": float(same.mean()), "reusable_sample": float(same[srows].mean()),
                      "rerun_sample": int((~same[srows]).sum())})
    reuse = pd.DataFrame(reuse).set_index("setting")
    reuse.to_csv(f"{pre}_reuse.csv")

    pd.set_option("display.width", 250)
    print("## coverage@15\n" + md(cov.xs("v2", axis=1, level=0)) + "\n\n### diff (v2 - v1)\n" + md(cov.xs("diff", axis=1, level=0)))
    print("\n## LLM-free Hit@1 (exact)\n" + md(hit))
    print("\n## labels\n" + summary.to_string(index=False) + "\n" + trans.to_string(index=False))
    print("\n## fix attribution\n" + att.to_string(index=False))
    print("\n## reader-input reuse\n" + md(reuse))
    dates = date_parser_effect(args)
    if dates is not None:
        print("\n## date parser v2 effect\n" + dates["text"])
    if args.addendum:
        write_addendum(Path(args.addendum), args, cov, hit, summary, trans, att, reuse, dates)


def date_parser_effect(args):
    """Constraint recognition and metrics of the new tag before (snapshot, parser v1) and after (parser v2)."""
    snap = args.new_tag + args.dates_snapshot
    cr_old = RESULTS / f"multitq_constraint_recognition_{snap}.csv"
    if not cr_old.exists():
        return None
    a = pd.read_csv(cr_old).set_index("qtype"); b = pd.read_csv(RESULTS / f"multitq_constraint_recognition_{args.new_tag}.csv").set_index("qtype")
    cr = pd.DataFrame({"n": b["n"], "target_date_v1": a["target_date"], "target_date_v2": b["target_date"],
                       "window_v1": a["window"], "window_v2": b["window"], "tr_fired_v1": a["tr_fired"], "tr_fired_v2": b["tr_fired"]})
    c_old, c_new = coverage15(snap), coverage15(args.new_tag)
    h_old = pd.read_csv(RESULTS / f"multitq_llmfree_hit1_{snap}.csv").set_index("reranker")
    h_new = pd.read_csv(RESULTS / f"multitq_llmfree_hit1_{args.new_tag}.csv").set_index("reranker")
    m = pd.DataFrame({"cov15_v1": c_old["all"], "cov15_v2": c_new["all"], "cov15_before_after_v1": c_old["before_after"],
                      "cov15_before_after_v2": c_new["before_after"], "hit1_v1": h_old["exact_all"], "hit1_v2": h_new["exact_all"],
                      "hit1_before_after_v1": h_old["exact_before_after"], "hit1_before_after_v2": h_new["exact_before_after"]}).reindex(RERANKERS)
    cr.to_csv(RESULTS / f"multitq_{args.new_tag}_dateparser_recognition.csv")
    m.to_csv(RESULTS / f"multitq_{args.new_tag}_dateparser_metrics.csv")
    return {"cr": cr, "m": m, "text": md(cr) + "\n\n" + md(m)}


NAMES = {"none": "none", "timer4": "TimeR4-style before rerank", "hardwindow": "hard window", "atc": "ATC only",
         "tr": "TR only", "tear": "TEAR", "tear_window": "TEAR + window", "timer4_public": "TimeR4 public top-15"}


def write_addendum(path: Path, args, cov, hit, summary, trans, att, reuse, dates=None):
    sig = pd.read_csv(RESULTS / f"multitq_llmfree_significance_{args.new_tag}.csv").set_index("reranker")
    row = lambda cells: "| " + " | ".join(cells) + " |"  # noqa: E731
    lab = summary.iloc[0]
    d_none, d_tear = cov.loc["none", "diff"], cov.loc["tear", "diff"]
    atc_eq = (cov.loc["none", "v2"]["equal"], cov.loc["atc", "v2"]["equal"])
    out = [f"# Addendum: rule v2 and query fallback (v2, `{args.new_tag}`) vs v1 (`{args.old_tag}`)", "",
           "Generated by `scripts/13_compare_versions.py --addendum`. Setting: fixed all-mpnet-base-v2 retriever, shared "
           "top-1000 pool, rule labels, exact-entity ATC, extended date parser, lambda1 = lambda2 = 0.8; MultiTQ test, "
           f"{lab.questions:,} questions.", "",
           "**What changed.** (1) Query fallback: TimeR4 rewrites that are empty or truncated are replaced by the original "
           "question (`questions_v2.json`, field `query`); search, labels, reranking cues and LLM-free Hit@1 read it. Only "
           "those rows were re-encoded and re-searched; every other row of `q_emb_v2.npy` / `top1000_v2_*.npy` equals v1. "
           "(2) Answer-slot rule v2 (`tear/labeling/rules.py`, default; v1 kept as `--rule_version v1`) with five fixes from "
           "two human checks of the labels: (a) do-support counts only right after the wh-phrase, so a trailing "
           "\"... before Cambodia did\" no longer makes the answer an object; (b) \"in\" joins the prepositions before the "
           "wh-word; (c) \"the last country with which / to which / that X ...\" -> object; (d) a leading date clause with "
           "a comma inside the date (\"After Sep, 2011, ...\") is removed whole; (e) \"suffer(ed) from\" asks for the victim "
           "-> object, while \"made X suffer\" still asks for the attacker. (3) Date parser v2 (`--date_parser v2`, "
           "`tear/temporal.py`): before/after dates written with a month abbreviation or punctuation ('Sep, 2011', "
           "'Sept. 2011', 'Sep 5, 2011', '5 Sep 2011') are now read; boundaries unchanged (before a month -> its 1st, "
           "after a month -> its last day, a year alone -> Jan 1 / Dec 31). The parser is shared by every reranker that "
           "uses dates (TimeR4-style, hard window, TR, TEAR, TEAR + window); v1 runs keep parser v1.", "",
           "## Label changes", "",
           row(["step", "questions", "slot changes", "transitions"]), row(["---"] * 4)]
    for r in att.itertuples(index=False):
        out.append(row([r.step, f"{r.questions:,}", f"{r.slot_changes:,}", r.transitions or "-"]))
    out += ["", f"Net v1 -> v2: {lab.slot_changed:,} slot changes ("
            + ", ".join(f"{a} -> {b} {n:,}" for a, b, n in trans.itertuples(index=False))
            + f"), {lab.topic_changed:,} topic-entity changes. Each fix is covered by regression cases in "
            "`tests/test_rules.py` (each case also records the v1 error).", "",
            "## Answer coverage@15", "",
            row(["reranker", "v1 all", "v2 all", "diff", "v2 Multiple", "diff", "v2 entity", "diff", "v2 equal_multi",
                 "v2 after_first", "v2 before_last"]), row(["---"] * 11)]
    for rk in RERANKERS:
        v1, v2, df = cov.loc[rk, "v1"], cov.loc[rk, "v2"], cov.loc[rk, "diff"]
        out.append(row([NAMES[rk], f"{v1['all']:.4f}", f"{v2['all']:.4f}", f"{df['all']:+.4f}", f"{v2['Multiple']:.4f}",
                        f"{df['Multiple']:+.4f}", f"{v2['entity']:.4f}", f"{df['entity']:+.4f}", f"{v2['equal_multi']:.4f}",
                        f"{v2['after_first']:.4f}", f"{v2['before_last']:.4f}"]))
    out += ["", f"The query fallback alone moves the label-free rerankers (none: {d_none['all']*100:+.2f} points overall, "
            f"{d_none['Multiple']*100:+.2f} on Multiple); with rule v2, TEAR moves {d_tear['all']*100:+.2f} overall and "
            f"{d_tear['Multiple']*100:+.2f} on Multiple. On `equal` questions ATC still costs coverage "
            f"({atc_eq[0]:.4f} -> {atc_eq[1]:.4f}).", "",
            "## LLM-free Hit@1 (exact; answer read off the top-1 fact)", "",
            row(["reranker", "v1", "v2", "diff", "v2 Multiple", "v2 entity", "v2 vs none, points (95% CI)"]), row(["---"] * 7)]
    for rk in RERANKERS:
        v1, v2, df = hit.loc[rk, "v1"], hit.loc[rk, "v2"], hit.loc[rk, "diff"]
        ci = "" if rk == "none" else (f"{sig.loc[rk, 'diff']*100:+.2f} [{sig.loc[rk, 'ci_low']*100:+.2f}, "
                                      f"{sig.loc[rk, 'ci_high']*100:+.2f}]")
        out.append(row([NAMES[rk], f"{v1['exact_all']:.4f}", f"{v2['exact_all']:.4f}", f"{df['exact_all']:+.4f}",
                        f"{v2['exact_Multiple']:.4f}", f"{v2['exact_entity']:.4f}", ci]))
    pmax = sig[["bootstrap_p", "mcnemar_p"]].max().max()
    out += ["", "v2 differences against `none`: McNemar and paired bootstrap (10,000 replicates) p "
            + ("< 1e-4" if pmax < 1e-4 else f"<= {pmax:.2g}") + f" (`results/multitq_llmfree_significance_{args.new_tag}.csv`).", "",
            "## Reader inputs that stay identical (prediction reuse)", "",
            "A reader prediction can be reused when its top-15 facts and question text are unchanged (greedy decoding). "
            "`scripts/12_reuse_reader_predictions.py` copies exactly those records from `results/qwen/` to `results/qwen_v2/`.", "",
            row(["setting", "unchanged, all questions", f"unchanged, reader sample ({args.sample})", "to re-run in the sample"]),
            row(["---"] * 4)]
    for rk, r in reuse.iterrows():
        out.append(row([NAMES.get(rk, rk), f"{r.reusable_all:.4f}", f"{r.reusable_sample:.4f}", f"{int(r.rerun_sample)}"]))
    if dates is not None:
        cr, mm = dates["cr"], dates["m"]
        out += ["", "## Date parser v2 (within v2)", "",
                "Constraint recognition (share of questions; `target_date` = a before/after date was parsed, `window` = the "
                "time window is active, `tr_fired` = TR selected a group representative):", "",
                row(["qtype", "n", "target_date v1 parser", "v2 parser", "window v1 parser", "v2 parser", "tr_fired v1", "v2"]),
                row(["---"] * 8)]
        for qt, r in cr.iterrows():
            out.append(row([qt, f"{int(r.n):,}", f"{r.target_date_v1:.3f}", f"{r.target_date_v2:.3f}", f"{r.window_v1:.3f}",
                            f"{r.window_v2:.3f}", f"{r.tr_fired_v1:.3f}", f"{r.tr_fired_v2:.3f}"]))
        out += ["", "Coverage@15 and LLM-free Hit@1 (exact) of the v2 configuration with parser v1 vs v2:", "",
                row(["reranker", "cov@15 v1 parser", "v2 parser", "cov@15 before_after v1", "v2", "Hit@1 v1 parser", "v2 parser",
                     "Hit@1 before_after v1", "v2"]), row(["---"] * 9)]
        for rk, r in mm.iterrows():
            out.append(row([NAMES[rk], f"{r.cov15_v1:.4f}", f"{r.cov15_v2:.4f}", f"{r.cov15_before_after_v1:.4f}",
                            f"{r.cov15_before_after_v2:.4f}", f"{r.hit1_v1:.4f}", f"{r.hit1_v2:.4f}",
                            f"{r.hit1_before_after_v1:.4f}", f"{r.hit1_before_after_v2:.4f}"]))
    out += ["", "## Files", "",
            "- `cache/multitq/questions_v2.json` (`query`, `query_source`), `q_emb_v2.npy`, `top1000_v2_{ids,scores}.npy`, "
            "`ner_spans_all_v2.json`, `topic_ner_all_v2.json`, `slots_rule_v2.json`",
            f"- `cache/multitq/rerank/<reranker>_{args.new_tag}_top30_ids.npy` (same `question_order.json`)",
            f"- `results/multitq_{{coverage_main,ablation,lambda_sweep,constraint_recognition,slot_occupancy_top15}}_{args.new_tag}.csv`, "
            f"`results/multitq_llmfree_{{hit1,significance}}_{args.new_tag}.csv`",
            f"- `results/multitq_{args.new_tag}_vs_{args.old_tag}_{{coverage,llmfree,labels,slot_transitions,fix_attribution,reuse}}.csv`",
            "- build: `logs/build_v2.sh` (all steps), `logs/build_v2_labels.sh` (labels onward); CPU, 2 threads",
            "- pending until the current reader run stops: `patches/reader_stage_query.patch` (reader prompt reads the query; "
            "`scripts/07 --questions`) and `patches/hits_stage_query.patch` (LLM-free Hit@1 reads the query; removes the "
            "stopgap in `scripts/06`)."]
    path.write_text("\n".join(out) + "\n")
    print("wrote", path)


if __name__ == "__main__":
    main()
