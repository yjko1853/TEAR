"""Final report of the TimelineKGQA reader runs (Qwen2.5-7B-Instruct-4bit, prompt v3i, k = 15, no-leak sample).

Reads the per-question scores written by scripts/08_score_reader.py for each graph and adds what the per-chunk scoring
does not give: the paired comparison of the two windowed settings, results by level and answer slot, and how many
sample questions grounding gave a window (some constraint resolved to an anchor fact and the rule label is an entity
slot; for a timestamp label the window is not applied, see scripts/16_timeline_grounding.py). gained / lost: questions
the system answers and the baseline does not, and the reverse.

Output: results/qwen_timeline_final.md, results/qwen_timeline_final_summary.csv, results/qwen_timeline_final_paired.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tear.evaluation.significance import mcnemar, paired_bootstrap  # noqa: E402
from tear.paths import CACHE, RESULTS  # noqa: E402
from tear.utils.io import load_json  # noqa: E402

SETTINGS = {"none": "none_rule_ground", "tear_window": "tear_window_rule_ground", "hardwindow": "hardwindow_rule_ground"}
PAIRS = [("hardwindow", "none"), ("tear_window", "none"), ("hardwindow", "tear_window")]
TARGET = ("subject", "object")


def load(ds: str, sfx: str) -> pd.DataFrame:
    pq = pd.read_csv(RESULTS / f"qwen_timeline_{ds}{sfx}_scores" / "multitq_qwen_per_question.csv", dtype={"quid": str})
    anchors = load_json(CACHE / ds / "interval_anchors.json")
    slots = load_json(CACHE / ds / "slots_rule_v2.json")
    pq["rule_slot"] = pq.quid.map(lambda q: slots.get(q, {}).get("answer_slot"))
    pq["resolved"] = pq.quid.map(lambda q: any(c.get("anchors") for c in anchors.get(q, [])))
    pq["grounded"] = pq.resolved & pq.rule_slot.isin(TARGET)
    pq["level"] = pq.qlabel
    return pq


def wide(pq: pd.DataFrame, col: str) -> pd.DataFrame:
    w = pq.pivot_table(index="quid", columns="setting", values=col, aggfunc="first").astype(bool)
    return w.rename(columns={v: k for k, v in SETTINGS.items()})


def subsets(pq: pd.DataFrame):
    meta = pq.drop_duplicates("quid").set_index("quid")
    yield "all", meta.index
    for lv in sorted(meta.level.unique()):
        yield f"level={lv}", meta.index[meta.level == lv]
    for at in sorted(meta.answer_type.unique()):
        yield f"answer={at}", meta.index[meta.answer_type == at]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", default="_noleak", help="results/qwen_timeline_<ds><suffix>_scores")
    args = ap.parse_args()
    summ, paired, notes = [], [], []
    for ds in ("icews", "cron"):
        pq = load(ds, args.suffix)
        meta = pq.drop_duplicates("quid")
        notes.append(f"- {ds}: {len(meta)} questions; grounded {int(meta.grounded.sum())}, constraint resolved "
                     f"{int(meta.resolved.sum())}, rule label timestamp (window not applied) "
                     f"{int((meta.rule_slot == 'timestamp').sum())}")
        W = {c: wide(pq, c) for c in ("hit1_exact", "hit1_lenient")}
        for name, idx in subsets(pq):
            row = {"dataset": ds, "subset": name, "n": len(idx)}
            for s in SETTINGS:
                row[f"{s}_exact"] = W["hit1_exact"].loc[idx, s].mean()
                row[f"{s}_lenient"] = W["hit1_lenient"].loc[idx, s].mean()
            summ.append(row)
            if name.startswith("answer="):
                continue
            for a, b in PAIRS:
                x, y = W["hit1_exact"].loc[idx, a].values, W["hit1_exact"].loc[idx, b].values
                bs, mc = paired_bootstrap(y, x, seed=2026), mcnemar(y, x)   # diff = system - baseline
                paired.append({"dataset": ds, "subset": name, "system": a, "baseline": b, "n": len(idx),
                               "diff": bs["diff"], "ci_low": bs["ci_low"], "ci_high": bs["ci_high"],
                               "bootstrap_p": bs["p_value"], "mcnemar_p": mc["p_value"], "gained": mc["n01"], "lost": mc["n10"]})
    S, P = pd.DataFrame(summ), pd.DataFrame(paired)
    S.to_csv(RESULTS / "qwen_timeline_final_summary.csv", index=False)
    P.to_csv(RESULTS / "qwen_timeline_final_paired.csv", index=False)
    def fmt(df):
        df = df.copy()
        for c in df.columns:
            if c in ("bootstrap_p", "mcnemar_p"):
                df[c] = df[c].map(lambda v: f"{v:.1e}")
            elif df[c].dtype.kind == "f":
                df[c] = df[c].map(lambda v: f"{v:.3f}")
        return df.to_markdown(index=False, disable_numparse=True)
    md = ["# TimelineKGQA reader, final (Qwen2.5-7B-Instruct-4bit, prompt v3i, k = 15, no-leak sample)", "",
          "Settings: none = retriever order; tear_window = TEAR + window + interval grounding (rule labels);",
          "hardwindow = window + interval grounding, no ATC / TR (needs only entity vs time from the rule label).",
          "Hit@1 exact = first answer equals a gold answer after normalisation.", "", *notes, "",
          "## Hit@1 by subset", "", fmt(S), "", "## Paired tests (exact Hit@1, bootstrap 10,000, seed 2026)", "", fmt(P), ""]
    (RESULTS / "qwen_timeline_final.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
