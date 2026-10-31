"""Second retriever: the main setting with BAAI/bge-base-en-v1.5 next to all-mpnet-base-v2 (pre-registered:
results/prereg_retriever_and_granularity.md, part A). Reads the stage 05 / 06 CSVs of cache/multitq_orig and
cache/multitq_bge (tag rule_v2_ground_exact_anydate).

Output: results/retriever_comparison.md, results/retriever_comparison.csv
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from tear.paths import RESULTS  # noqa: E402

TAG = "rule_v2_ground_exact_anydate"
RUNS = {"all-mpnet-base-v2": "multitq_orig", "bge-base-en-v1.5": "multitq_bge"}
ORDER = ["none", "timer4", "chrono", "hardwindow", "atc", "tr", "tear", "tear_window"]


def main():
    rows = []
    for enc, ds in RUNS.items():
        cov = pd.read_csv(RESULTS / f"{ds}_coverage_main_{TAG}.csv")
        hit = pd.read_csv(RESULTS / f"{ds}_llmfree_hit1_{TAG}.csv").set_index("reranker")
        sig = pd.read_csv(RESULTS / f"{ds}_llmfree_significance_{TAG}.csv").set_index("reranker")
        c15 = cov[(cov.k == 15) & (cov.group == "all")].set_index("reranker").coverage
        for rk in [r for r in ORDER if r in hit.index]:
            rows.append({"retriever": enc, "reranker": rk, "coverage15": c15.get(rk), "hit1": hit.loc[rk, "exact_all"],
                         "hit1_entity": hit.loc[rk, "exact_entity"], "hit1_time": hit.loc[rk, "exact_time"],
                         "diff_vs_none": sig.loc[rk, "diff"] if rk in sig.index else 0.0,
                         "ci_low": sig.loc[rk, "ci_low"] if rk in sig.index else None,
                         "ci_high": sig.loc[rk, "ci_high"] if rk in sig.index else None,
                         "mcnemar_p": sig.loc[rk, "mcnemar_p"] if rk in sig.index else None})
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "retriever_comparison.csv", index=False)
    t = df.copy()
    for c in t.columns:
        if c == "mcnemar_p":
            t[c] = t[c].map(lambda v: "" if pd.isna(v) else f"{v:.1e}")
        elif t[c].dtype.kind == "f":
            t[c] = t[c].map(lambda v: "" if pd.isna(v) else f"{v:.3f}")
    md = ["# Second retriever (MultiTQ test, original questions + grounding, LLM-free)", "",
          "Pre-registered: results/prereg_retriever_and_granularity.md (A). Same labels, grounding, lambdas and rules; only "
          "the retriever differs.", "", t.to_markdown(index=False, disable_numparse=True), ""]
    (RESULTS / "retriever_comparison.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
