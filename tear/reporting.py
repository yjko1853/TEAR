"""Assemble results/results.md from the CSV outputs (tables only; interpretation is left to the author)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .paths import CACHE, LOGS, RESULTS
from .utils.io import load_json


def md(df: pd.DataFrame, floatfmt="{:.3f}") -> str:
    df = df.copy()
    for c in df.columns:
        if df[c].dtype.kind == "f":
            df[c] = df[c].map(lambda v: "" if pd.isna(v) else floatfmt.format(v))
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(x) for x in r.values) + " |")
    return "\n".join(lines)


def pivot(df, index, columns, values, k=None, group_type=None, rerankers=None):
    d = df
    if k is not None:
        d = d[d.k == k]
    if group_type is not None:
        d = d[d.group_type.isin(group_type)]
    if rerankers is not None:
        d = d[d.reranker.isin(rerankers)]
        p = d.pivot_table(index=index, columns=columns, values=values, aggfunc="first")
        p = p.reindex([r for r in rerankers if r in p.index])
    else:
        p = d.pivot_table(index=index, columns=columns, values=values, aggfunc="first")
    p.columns = [str(c) for c in p.columns]
    return p.reset_index()


MAIN = ["none", "timer4", "chrono", "hardwindow", "tear"]
EXTRA = ["timer4_sym", "chrono_P100", "chrono_P1000", "atc", "tr", "tear_window"]
ORDER_MULTITQ = ["all", "equal", "before_after", "first_last", "equal_multi", "after_first", "before_last"]


def section_dataset(ds, tag, title, out):
    p = RESULTS / f"{ds}_coverage_main_{tag}.csv"
    if not p.exists():
        out.append(f"\n### {title}\n\n(not available: {p.name} missing)\n")
        return
    df = pd.read_csv(p)
    gkey = "qtype" if ds == "multitq" else "answer_type"
    out.append(f"\n### {title}\n")
    out.append(f"\n**Answer coverage@15 by {gkey}** (five main rerankers; `all` = overall)\n")
    pv = pivot(df, "reranker", "group", "coverage", k=15, group_type=["overall", gkey], rerankers=MAIN)
    if ds == "multitq":
        pv = pv[["reranker"] + [c for c in ORDER_MULTITQ if c in pv.columns]]
    out.append(md(pv, "{:.4f}"))
    out.append(f"\n**Answer coverage@15, extra variants**\n")
    pv = pivot(df, "reranker", "group", "coverage", k=15, group_type=["overall", gkey], rerankers=EXTRA)
    if ds == "multitq":
        pv = pv[["reranker"] + [c for c in ORDER_MULTITQ if c in pv.columns]]
    out.append(md(pv, "{:.4f}"))
    if ds == "multitq":
        out.append("\n**Answer coverage@15 by answer_type / qlabel**\n")
        pv = pivot(df, "reranker", "group", "coverage", k=15, group_type=["answer_type", "qlabel"], rerankers=MAIN + EXTRA)
        out.append(md(pv, "{:.4f}"))
    else:
        out.append("\n**Answer coverage@15 by difficulty level** (subject/object/timestamp_start/timestamp_end questions only; "
                   "coverage is undefined for duration / range / relation_* answers)\n")
        pv = pivot(df, "reranker", "group", "coverage", k=15, group_type=["level"], rerankers=MAIN + EXTRA)
        out.append(md(pv, "{:.4f}"))
        if "gold_recall_all" in df.columns:
            out.append("\n**Gold-event recall@15 (all gold events in top-15 / any gold event in top-15)** by answer_type\n")
            d = df[(df.k == 15) & (df.group_type == "answer_type") & df.reranker.isin(MAIN + ["atc"])]
            pv = d.pivot_table(index="reranker", columns="group", values="gold_recall_all", aggfunc="first")
            pv2 = d.pivot_table(index="reranker", columns="group", values="gold_recall_any", aggfunc="first")
            pv.columns = [f"{c} (all)" for c in pv.columns]; pv2.columns = [f"{c} (any)" for c in pv2.columns]
            out.append(md(pd.concat([pv, pv2], axis=1).reset_index(), "{:.4f}"))
            out.append("\n**Gold-event recall@15 by level**\n")
            d = df[(df.k == 15) & (df.group_type == "level") & df.reranker.isin(MAIN + ["atc"])]
            pv = d.pivot_table(index="reranker", columns="group", values=["gold_recall_all", "gold_recall_any"], aggfunc="first")
            pv.columns = [f"{a}:{b}" for a, b in pv.columns]
            out.append(md(pv.reset_index(), "{:.4f}"))
    out.append("\n**Overall answer coverage@k**\n")
    pv = pivot(df, "reranker", "k", "coverage", group_type=["overall"], rerankers=MAIN + EXTRA)
    out.append(md(pv, "{:.4f}"))
    if ds != "multitq" and "gold_recall_all" in df.columns:
        out.append("\n**Overall gold-event recall@k (all gold events)**\n")
        pv = pivot(df, "reranker", "k", "gold_recall_all", group_type=["overall"], rerankers=MAIN + EXTRA)
        out.append(md(pv, "{:.4f}"))
    # ablation
    p = RESULTS / f"{ds}_ablation_{tag}.csv"
    if p.exists():
        ab = pd.read_csv(p)
        out.append(f"\n**Ablation 2x2 (coverage@15 by {gkey})**\n")
        pv = ab[(ab.k == 15) & ab.group_type.isin(["overall", gkey])].pivot_table(index="setting", columns="group",
                                                                                 values="coverage", aggfunc="first")
        pv = pv.reindex(["base", "+ATC", "+TR", "+ATC+TR"])
        if ds == "multitq":
            pv = pv[[c for c in ORDER_MULTITQ if c in pv.columns]]
        out.append(md(pv.reset_index(), "{:.4f}"))
    # sweep
    p = RESULTS / f"{ds}_lambda_sweep_{tag}.csv"
    if p.exists():
        sw = pd.read_csv(p)
        out.append("\n**lambda1 (rows) x lambda2 (columns) sweep, overall coverage@15**\n")
        pv = sw[(sw.k == 15) & (sw.group == "all")].pivot(index="lambda1", columns="lambda2", values="coverage")
        pv.columns = [str(c) for c in pv.columns]
        out.append(md(pv.reset_index(), "{:.4f}"))
        for g in (["entity", "time"] if ds == "multitq" else ["subject", "object"]):
            sub = sw[(sw.k == 15) & (sw.group == g)]
            if len(sub):
                out.append(f"\n*{g} questions only*\n")
                pv = sub.pivot(index="lambda1", columns="lambda2", values="coverage"); pv.columns = [str(c) for c in pv.columns]
                out.append(md(pv.reset_index(), "{:.4f}"))
    # constraint recognition
    p = RESULTS / f"{ds}_constraint_recognition_{tag}.csv"
    if p.exists():
        out.append("\n**Constraint recognition (share of questions)**: rule-based cue / date extraction and whether TEAR's "
                   "rules actually fired\n")
        out.append(md(pd.read_csv(p), "{:.3f}"))
    p = RESULTS / f"{ds}_slot_occupancy_top15_{tag}.csv"
    if p.exists():
        oc = pd.read_csv(p)
        out.append("\n**Slot occupancy in top-15** (share of top-15 candidates whose answer-slot value is the topic entity, "
                   "i.e. candidates that cannot contain the answer in the predicted slot; questions with slot+topic only)\n")
        pv = oc.pivot_table(index="reranker", columns=gkey, values="topic_in_answer_slot_share_top15", aggfunc="first")
        pv = pv.reindex([r for r in MAIN + EXTRA if r in pv.index])
        if ds == "multitq":
            pv = pv[[c for c in ["ALL"] + ORDER_MULTITQ[1:] if c in pv.columns]]
        out.append(md(pv.reset_index(), "{:.4f}"))


def build_report(out_path: Path | None = None, generator: str = "scripts/10_make_report.py") -> Path:
    out = ["# TEAR revision: retrieval-side results (S0, S1, S2, S4, S5, S8)\n",
           f"Generated by `{generator}` from `results/*.csv` on {pd.Timestamp.now():%Y-%m-%d %H:%M}. "
           "Retriever: sentence-transformers/all-mpnet-base-v2, exact inner-product search (FAISS IndexFlatIP), "
           "candidate pool = top-1000 per question, rerankers applied on the same pool, metrics on the top-k cut.\n",
           "\nMetric definitions\n",
           "- **answer coverage@k**: share of questions for which at least one of the top-k facts has an element "
           "(subject, relation, object, timestamp; TimelineKGQA: subject, predicate, object, start, end) that matches an "
           "answer string under TEAR's `match` rule (lower-case, punctuation/articles removed, answer is a substring of "
           "the element).",
           "- **gold-event recall@k** (TimelineKGQA only): the dataset lists the gold events of every question; "
           "`all` = every gold event is in the top-k (the strict Hits definition of the TimelineKGQA paper), "
           "`any` = at least one gold event is in the top-k.",
           "- MultiTQ: cues and dates are extracted from the TimeR4 rewritten question; TimelineKGQA: from the original "
           "question (its `paraphrased_question` column is misaligned for medium/complex questions, see data notes).\n"]

    # ---- data notes
    out.append("\n## 1. Data and schema notes\n")
    qs = load_json(CACHE / "multitq" / "questions.json")
    n_unmapped = sum(1 for q in qs for x in q["timer4_top15_ids"] if x < 0)
    out.append(f"- MultiTQ (TimeR4 repo): KG `full.txt` 461,329 quadruples, 10,488 entities, 251 relations, 4,017 days; "
               f"test {len(qs):,} questions; `test_prompt.json` top-15 strings mapped to KG facts except {n_unmapped} of "
               f"{15*len(qs):,}.")
    for ds, name in (("icews", "TimelineKGQA ICEWS-Actor"), ("cron", "TimelineKGQA CronQuestions-KG")):
        p = CACHE / ds / "schema_notes.json"
        if not p.exists():
            continue
        s = load_json(p)
        out.append(f"- {name}: KG columns `{', '.join(s['kg_columns'])}`; {s['n_facts']:,} interval facts "
                   f"(open start {s['open_start']:,}, open end {s['open_end']:,}), {s['n_entities']:,} entities, "
                   f"predicates: {', '.join(s['predicates'][:5])}{'...' if len(s['predicates'])>5 else ''}. "
                   f"Question columns `{', '.join(s['question_columns'])}`. Split column counts: {s['split_counts_all']}; "
                   f"test used here: {s['n_test']:,} (by level {s['test_by_level']}). "
                   f"Test answer types: {s['test_by_answer_type']}. Gold-event mapping: {s['gold_event_mapping']}.")
    out.append("- TimelineKGQA schema versus the plan: facts are **intervals** (`start_time`, `end_time`, with the "
               "sentinels `beginning of time` / `end of time`), not point timestamps; ICEWS-Actor has a single predicate "
               "`Affiliation To`; answer types are `subject`, `object`, `timestamp_start`, `timestamp_end`, "
               "`timestamp_range`, `duration`, `relation_ranking`, `relation_duration`, `relation_union_or_intersection`; "
               "levels `simple`/`medium`/`complex`. The repo README reports 53,624/17,874/17,874 for ICEWS-Actor, but the "
               "`split` column of the released file gives 57,453/17,298/14,621 (simple 5,994 and medium 5,330 match the "
               "README, complex 3,297 vs 6,550). The `split` column is used. TimelineKGQA questions contain no "
               "first/last cue words and no `before|after <date>` phrases (before/after refer to other events), so "
               "TEAR's TR rules and the TimeR4/chrono/hard-window baselines have nothing to act on there; only the "
               "slot-conditioned ATC penalty and, for simple `from X to Y` questions, an interval window apply.\n")

    # ---- timing
    p = LOGS / "timing.csv"
    if p.exists():
        t = pd.read_csv(p)
        t = t[~t.step.str.contains("chunk") & (t.note.fillna("") != "smoke")]
        agg = t.groupby(["step", "dataset"]).agg(seconds=("seconds", "sum"), runs=("seconds", "size"),
                                                last_note=("note", "last")).reset_index()
        out.append("\n## 2. Wall-clock time per step (Apple M2, 16 GB, MPS; logs/timing.csv)\n")
        out.append(md(agg, "{:.0f}"))

    # ---- oracle / NER
    out.append("\n## 3. Answer slot and topic entity labels\n")
    for ds in ("multitq", "icews", "cron"):
        p = RESULTS / f"{ds}_oracle_slot_derivation.csv"
        if p.exists():
            out.append(f"\n**{ds}: KG-oracle slot derivation success**\n")
            out.append(md(pd.read_csv(p), "{:.3f}"))
        for lab in ("faithful", "all"):
            p = RESULTS / f"{ds}_topic_entity_agreement_{lab}.csv"
            if p.exists():
                out.append(f"\n**{ds}: NER topic entity vs KG-oracle topic entity (labels={lab})** "
                           f"(`faithful` keeps label ids 1,2 = B-MISC/I-MISC exactly as the original code; `all` keeps "
                           f"PER/ORG/LOC/MISC)\n")
                out.append(md(pd.read_csv(p), "{:.3f}"))
            p = RESULTS / f"{ds}_fuzzy_engine_check_{lab}.txt"
            if p.exists():
                out.append("\n" + p.read_text().strip() + "\n")

    # ---- main results
    out.append("\n## 4. Reranker comparison on the shared top-1000 pool\n")
    out.append("Rerankers: `none` similarity only; `timer4` TimeR4 before-reranking (mu=0.4; `timer4_sym` also handles "
               "after); `chrono` chronological sort of the top-30 by similarity (`_P100`, `_P1000` widen the pool; a pool "
               "<= k cannot change coverage@k); `hardwindow` removes candidates outside the before/after window; "
               "`tear` = ATC penalty (lambda1=0.8) + group-representative TR (lambda2=0.8); `atc`, `tr` are the single "
               "components; `tear_window` (extra, not in the original code) additionally moves candidates outside the "
               "before/after window behind the valid ones. Labels: KG-oracle answer slot and topic entity (to be "
               "replaced by GPT-based ATC output).\n")
    section_dataset("multitq", "oracle_anydate", "MultiTQ (extended date parser: ISO, '12 May 2011', 'May 2011', "
                    "'2011-05', '2011' after before/after)", out)
    section_dataset("multitq", "oracle", "MultiTQ (faithful TEAR date regex: 'before|after YYYY-MM-DD' only)", out)
    p = RESULTS / "multitq_timer4_public_top15.csv"
    if p.exists():
        out.append("\n**Reference: TimeR4 public top-15 (`test_prompt.json`) answer coverage@15**\n")
        out.append(md(pd.read_csv(p), "{:.4f}"))
    for tag, lab in (("oracle_nertopic_anydate", "all"), ("oracle_nertopic_faithful_anydate", "faithful")):
        p = RESULTS / f"multitq_coverage_main_{tag}.csv"
        if p.exists():
            section_dataset("multitq", tag, f"MultiTQ with NER-based topic entities (labels={lab}) instead of the "
                            "oracle topic entity (oracle slot kept; extended date parser)", out)
    section_dataset("icews", "oracle", "TimelineKGQA ICEWS-Actor", out)
    section_dataset("cron", "oracle", "TimelineKGQA CronQuestions-KG", out)

    # ---- motivation
    out.append("\n## 5. Motivation analyses\n")
    for ds in ("multitq", "icews", "cron"):
        p = RESULTS / f"{ds}_same_sro_similarity.csv"
        if p.exists():
            out.append(f"\n**{ds}: cosine similarity of fact pairs sharing (s, r, o) and differing only in time** "
                       f"(20,000 sampled pairs; reference rows: pairs sharing only the subject, random pairs)\n")
            out.append(md(pd.read_csv(p), "{:.4f}"))
    out.append("\nSlot-occupancy tables (share of top-15 candidates in which the topic entity sits in the predicted answer "
               "slot) are given per dataset in section 4.\n")
    out_path = Path(out_path) if out_path else RESULTS / "results.md"
    out_path.write_text("\n".join(out), encoding="utf-8")
    print("wrote", out_path)
    return out_path

