"""Figures of the paper (vector PDF plus PNG preview) -> figures/ (TEAR_FIGURES=<dir> overrides, e.g. the manuscript's
figures folder).

  fig_overview   the TEAR pipeline: frozen retriever, answer-type rules (slot + unit), KG anchor grounding, the three
                 reranking rules, top-k facts, reader
  fig_gains      no reranking vs TEAR on every benchmark, retriever and reader, read from the result CSVs:
                 (a) LLM-free Hit@1 of the top-ranked fact, (b) reader Hit@1 of the local 7B reader
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

from tear.paths import RESULTS  # noqa: E402
from decimal import ROUND_HALF_UP, Decimal  # noqa: E402


def r3(v):
    """Three decimals, half up (the rounding used in the text: 0.4265 -> 0.427)."""
    return str(Decimal(repr(float(v))).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP))


def gain(a, b):
    return str(Decimal(repr(round(100 * (float(b) - float(a)), 6))).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))

OUT = Path(os.environ.get("TEAR_FIGURES") or Path(__file__).resolve().parents[1] / "figures")
plt.rcParams.update({"font.family": "Helvetica", "font.size": 7.5, "pdf.fonttype": 42, "axes.linewidth": 0.6,
                     "mathtext.fontset": "custom", "mathtext.rm": "Helvetica", "mathtext.it": "Helvetica:italic",
                     "mathtext.bf": "Helvetica:bold", "mathtext.fallback": "stixsans"})
INK, MID, LIGHT, ACC, ACC_L = "#222222", "#6b6b6b", "#ececec", "#1f5fa8", "#dbe7f5"


def box(ax, x, y, w, h, title, body="", fc=LIGHT, ec=MID, tc=INK, lw=0.7, bold=True):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.012", fc=fc, ec=ec, lw=lw))
    if body:
        ax.text(x + w / 2, y + h - 0.022, title, ha="center", va="top", color=tc, weight="bold" if bold else "normal")
        ax.text(x + w / 2, y + h - 0.062, body, ha="center", va="top", color=tc, fontsize=6.6, linespacing=1.25)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", color=tc, weight="bold" if bold else "normal")


def arrow(ax, p, q, color=MID, rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=7, lw=0.7, color=color,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=1, shrinkB=1))


def overview():
    fig = plt.figure(figsize=(7.2, 3.4))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    # inputs
    box(ax, 0.01, 0.42, 0.135, 0.46, "Question $q$", "What was the last\ncountry to visit\nKuwait before the\nCouncil of\nRepresentatives?", fc="white")
    box(ax, 0.01, 0.08, 0.135, 0.26, "Temporal KG", "point facts\n$(s, r, o, t)$ or\ninterval facts\n$(s, r, o, [t_1, t_2])$", fc="white")
    # rules, retriever, grounding
    box(ax, 0.19, 0.66, 0.215, 0.24, "Answer-type rules", "slot $\\sigma(q)$: subject, object\nor timestamp; topic $e_q$;\nunit $u(q)$: year, month or day\n(from the question grammar)")
    box(ax, 0.19, 0.42, 0.215, 0.18, "Frozen retriever", "sentence encoder over all\nKG facts, exact top-1000")
    box(ax, 0.19, 0.08, 0.215, 0.27, "Anchor grounding (AG)", "anchor phrase: before $X$\nreference event $(X, r, e_q)$\nlooked up in the KG gives\na date, or a set of Allen\nrelations for intervals")
    # TEAR block
    ax.add_patch(FancyBboxPatch((0.455, 0.08), 0.265, 0.84, boxstyle="round,pad=0.004,rounding_size=0.015",
                                fc=ACC_L, ec=ACC, lw=0.9))
    ax.text(0.5875, 0.895, "TEAR reranking of the pool $C_q$", ha="center", va="top", color=ACC, weight="bold", fontsize=8)
    box(ax, 0.47, 0.63, 0.235, 0.21, "ATC  ($\\times\\lambda_1$)", "a candidate whose predicted\nanswer slot holds the topic\n(or the anchor) is penalised", fc="white", ec=ACC)
    box(ax, 0.47, 0.37, 0.235, 0.22, "TR  ($\\times\\lambda_2$)", "for first / last: in each\n$(s, r, o)$ group keep only the\nearliest or latest valid fact", fc="white", ec=ACC)
    box(ax, 0.47, 0.11, 0.235, 0.22, "Window", "candidates that violate the\n(grounded) time constraint\nmove behind the valid ones", fc="white", ec=ACC)
    # outputs
    box(ax, 0.765, 0.44, 0.105, 0.14, "Top-$k$ facts", "$k = 15$", fc="white")
    box(ax, 0.885, 0.44, 0.105, 0.14, "Answer", "France", fc="white")
    box(ax, 0.765, 0.08, 0.225, 0.29, "Reader", "LLM-free: the answer slot\nof the top-ranked fact, or a\nzero-shot 7B LLM that is told\nthe unit $u(q)$, e.g. answer\nwith the year only", fc="white")
    # arrows
    arrow(ax, (0.145, 0.78), (0.19, 0.78)); arrow(ax, (0.145, 0.51), (0.19, 0.51))
    arrow(ax, (0.12, 0.42), (0.215, 0.35)); arrow(ax, (0.145, 0.21), (0.19, 0.21))
    arrow(ax, (0.405, 0.51), (0.455, 0.51))
    arrow(ax, (0.405, 0.79), (0.47, 0.75)); arrow(ax, (0.405, 0.70), (0.47, 0.55))
    arrow(ax, (0.405, 0.30), (0.47, 0.44)); arrow(ax, (0.405, 0.22), (0.47, 0.22))
    arrow(ax, (0.72, 0.51), (0.765, 0.51)); arrow(ax, (0.8175, 0.44), (0.8175, 0.37)); arrow(ax, (0.9375, 0.37), (0.9375, 0.44))
    ax.text(0.43, 0.535, "$C_q$", ha="center", fontsize=6.5, color=MID)
    ax.text(0.437, 0.805, "$\\sigma, e_q$", ha="center", fontsize=6.5, color=MID)
    ax.text(0.437, 0.305, "date", ha="center", fontsize=6.5, color=MID)
    ax.text(0.877, 0.75, "no training\nno API\nno question rewriting\nthe only LLM call\nis the reader", ha="center", va="center",
            color=MID, fontsize=6.8, linespacing=1.4)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_overview.{ext}", dpi=300)
    plt.close(fig)


def gains():
    rc = pd.read_csv(RESULTS / "retriever_comparison.csv").set_index(["retriever", "reranker"])
    i18 = pd.read_csv(RESULTS / "icews18_llmfree_hit1_rule_v2_ground_exact_anydate.csv").set_index("reranker")
    qg = pd.read_csv(RESULTS / "qwen_ground_final_summary.csv").set_index(["metric", "setting"])
    q18 = pd.read_csv(RESULTS / "qwen_icews18_final_summary.csv").set_index(["setting", "metric"])
    tl = pd.read_csv(RESULTS / "qwen_timeline_final_summary.csv").set_index(["dataset", "subset"])
    left = [("MultiTQ\nall-mpnet", rc.loc[("all-mpnet-base-v2", "none"), "hit1"], rc.loc[("all-mpnet-base-v2", "tear_window"), "hit1"]),
            ("MultiTQ\nbge-base", rc.loc[("bge-base-en-v1.5", "none"), "hit1"], rc.loc[("bge-base-en-v1.5", "tear_window"), "hit1"]),
            ("ICEWS18\n(generated)", i18.loc["none", "exact_all"], i18.loc["tear_window", "exact_all"])]
    right = [("MultiTQ", qg.loc[("reader_date", "none"), "hit1"], qg.loc[("reader_date", "tear_window"), "hit1"]),
             ("ICEWS18\n(generated)", q18.loc[("none", "reader_date"), "hit1"], q18.loc[("tear_window", "reader_date"), "hit1"]),
             ("TimelineKGQA\nICEWS-Actor†", tl.loc[("icews", "all"), "none_exact"], tl.loc[("icews", "all"), "hardwindow_exact"]),
             ("TimelineKGQA\nCronQuestions†", tl.loc[("cron", "all"), "none_exact"], tl.loc[("cron", "all"), "hardwindow_exact"])]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.5), gridspec_kw={"width_ratios": [3, 4]})
    for ax, data, title in ((axes[0], left, "(a) LLM-free Hit@1 (top-ranked fact)"),
                            (axes[1], right, "(b) Reader Hit@1 (Qwen2.5-7B, k = 15)")):
        xs = range(len(data)); w = 0.36
        b0 = [d[1] for d in data]; b1 = [d[2] for d in data]
        ax.bar([x - w / 2 for x in xs], b0, w, color="#c9c9c9", label="no reranking")
        ax.bar([x + w / 2 for x in xs], b1, w, color=ACC, label="TEAR")
        for x, a, b in zip(xs, b0, b1):
            ax.text(x - w / 2, a + 0.008, r3(a), ha="center", va="bottom", fontsize=6)
            ax.text(x + w / 2, b + 0.008, r3(b), ha="center", va="bottom", fontsize=6, color=ACC)
            ax.text(x, max(a, b) + 0.07, "+" + gain(a, b), ha="center", fontsize=6.5, weight="bold", color=INK)
        ax.set_xticks(list(xs)); ax.set_xticklabels([d[0] for d in data], fontsize=6.6)
        ax.set_ylim(0, max(max(b1), max(b0)) + 0.16); ax.set_title(title, fontsize=7.5, loc="left")
        ax.spines[["top", "right"]].set_visible(False); ax.tick_params(width=0.6, length=2.5)
        ax.yaxis.grid(True, lw=0.3, color="#dddddd"); ax.set_axisbelow(True)
    axes[0].legend(frameon=False, fontsize=6.6, loc="upper left")
    fig.tight_layout(w_pad=1.5)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_gains.{ext}", dpi=300)
    plt.close(fig)
    pd.DataFrame([{"panel": "a", "group": g.replace("\n", " "), "none": a, "tear": b} for g, a, b in left] +
                 [{"panel": "b", "group": g.replace("\n", " "), "none": a, "tear": b} for g, a, b in right]
                 ).to_csv(OUT / "fig_gains_values.csv", index=False)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    overview(); gains()
    print("figures ->", OUT)
