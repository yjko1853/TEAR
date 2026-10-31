"""Paraphrase robustness of anchor grounding: original wording vs paraphrase on the same 600 questions
(pre-registered measures, results/prereg_paraphrase_robustness.md).

Original side: the main-setting runs of cache/multitq_orig restricted to the sample. Paraphrase side: the same runs of
cache/multitq_para (logs/run_paraphrase_pipeline.sh). Subsets: valid paraphrases (primary), all, and, once
results/paraphrase_meaning_check.csv is filled, the checked paraphrases judged to keep the meaning.

Output: results/paraphrase_robustness.md, results/paraphrase_robustness_{anchors,llmfree,paired}.csv
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tear.analysis.runs import load_run  # noqa: E402
from tear.evaluation.significance import mcnemar, paired_bootstrap  # noqa: E402
from tear.paths import CACHE, RESULTS  # noqa: E402
from tear.utils.io import load_json  # noqa: E402

GROUND, PLAIN = "rule_v2_ground_exact_anydate", "rule_v2_exact_anydate"
RK = ["none", "tear_window"]


def per_question(ds, quids):
    out = {}
    for tag, name in ((GROUND, "ground"), (PLAIN, "plain")):
        order, _, per = load_run(ds, tag, slots_file="slots_rule_v2.json", rerankers=RK)
        idx = {k: i for i, k in enumerate(order)}
        rows = [idx[k] for k in quids]
        for rk in RK:
            out[f"{rk}_{name}_hit1"] = per[rk]["hit1"][rows]
            out[f"{rk}_{name}_cov15"] = np.asarray(per[rk]["cov15"], bool)[rows]
    return pd.DataFrame(out, index=quids)


def anchor_table(quids, qtype):
    a_o, a_p = load_json(CACHE / "multitq_orig" / "anchors.json"), load_json(CACHE / "multitq_para" / "anchors.json")
    s_o, s_p = load_json(CACHE / "multitq_orig" / "slots_rule_v2.json"), load_json(CACHE / "multitq_para" / "slots_rule_v2.json")
    rows = []
    for k in quids:
        o, p = a_o.get(k), a_p.get(k)
        rows.append({"quid": k, "qtype": qtype[k],
                     "det_orig": o is not None, "det_para": p is not None,
                     "res_orig": bool(o and o["status"] == "resolved"), "res_para": bool(p and p["status"] == "resolved"),
                     "type_orig": o["type"] if o else None, "type_para": p["type"] if p else None,
                     "date_orig": o.get("date") if o else None, "date_para": p.get("date") if p else None,
                     "slot_orig": s_o.get(k, {}).get("answer_slot"), "slot_para": s_p.get(k, {}).get("answer_slot")})
    df = pd.DataFrame(rows).set_index("quid")
    both = df.res_orig & df.res_para
    df["same_date"] = both & (df.date_orig == df.date_para) & (df.type_orig == df.type_para)
    df["both_resolved"] = both
    return df


def summarise(df, subset_name, mask):
    g = df[mask]
    rows = []
    for qt, h in [("all", g)] + list(g.groupby("qtype")):
        nb = int(h.both_resolved.sum())
        rows.append({"subset": subset_name, "qtype": qt, "n": len(h),
                     "detected_orig": h.det_orig.mean(), "detected_para": h.det_para.mean(),
                     "resolved_orig": h.res_orig.mean(), "resolved_para": h.res_para.mean(),
                     "both_resolved": nb, "same_date_of_both": h.same_date.sum() / nb if nb else np.nan,
                     "slot_agree": (h.slot_orig == h.slot_para).mean()})
    return rows


def llmfree(df, subset_name, mask):
    g = df[mask]
    rows = []
    for qt, h in [("all", g)] + list(g.groupby("qtype")):
        r = {"subset": subset_name, "qtype": qt, "n": len(h)}
        for w in ("orig", "para"):
            for c in ("none_ground_hit1", "tear_window_ground_hit1", "tear_window_plain_hit1",
                      "none_ground_cov15", "tear_window_ground_cov15", "tear_window_plain_cov15"):
                r[f"{w}_{c}"] = h[f"{w}_{c}"].mean()
        rows.append(r)
    return rows


def paired(df, subset_name, mask):
    g = df[mask]
    tests = [("orig", "tear_window_ground_hit1", "orig", "none_ground_hit1"),
             ("para", "tear_window_ground_hit1", "para", "none_ground_hit1"),
             ("para", "tear_window_ground_hit1", "para", "tear_window_plain_hit1"),
             ("para", "tear_window_ground_hit1", "orig", "tear_window_ground_hit1"),
             ("para", "none_ground_hit1", "orig", "none_ground_hit1")]
    rows = []
    for ws, cs, wb, cb in tests:
        x, y = g[f"{ws}_{cs}"].values, g[f"{wb}_{cb}"].values
        bs, mc = paired_bootstrap(y, x, seed=2026), mcnemar(y, x)
        rows.append({"subset": subset_name, "system": f"{ws}:{cs}", "baseline": f"{wb}:{cb}", "n": len(g),
                     "diff": bs["diff"], "ci_low": bs["ci_low"], "ci_high": bs["ci_high"], "bootstrap_p": bs["p_value"],
                     "mcnemar_p": mc["p_value"], "gained": mc["n01"], "lost": mc["n10"]})
    return rows


def breakdown(df, subset_name, mask):
    """Where the paraphrase loses hits: anchor phrase missed, found but unresolved, slot label changed, other date."""
    def cat(r):
        if not r.res_orig:
            return "0 unresolved on the original"
        if not r.det_para:
            return "1 anchor phrase not found"
        if not r.res_para:
            return "2 found, not resolved"
        if r.slot_orig != r.slot_para:
            return "3 slot label changed"
        if not r.same_date:
            return "4 same slot, other date"
        return "5 same slot and date"
    g = df[mask].copy()
    g["cause"] = g.apply(cat, axis=1)
    rows = []
    for c, h in g.groupby("cause"):
        rows.append({"subset": subset_name, "cause": c, "n": len(h),
                     "orig_hit1": h.orig_tear_window_ground_hit1.mean(), "para_hit1": h.para_tear_window_ground_hit1.mean(),
                     "lost_hits": int(h.orig_tear_window_ground_hit1.sum() - h.para_tear_window_ground_hit1.sum())})
    return rows


def main():
    src = CACHE / "multitq_para_src"
    quids = load_json(src / "sample.json")["quids"]
    paras = load_json(src / "paraphrases.json")
    qtype = {str(q["quid"]): q["qtype"] for q in load_json(CACHE / "multitq_para" / "questions.json")}
    df = anchor_table(quids, qtype)
    o, p = per_question("multitq_orig", quids), per_question("multitq_para", quids)
    for c in o.columns:
        df[f"orig_{c}"], df[f"para_{c}"] = o[c].values, p[c].values
    df["valid"] = [paras[k]["valid"] for k in quids]
    subsets = [("valid", df.valid), ("all", pd.Series(True, index=df.index))]
    judge_path = RESULTS / "paraphrase_judge.csv"   # Amendment 2: blind LLM judge of all 600 pairs
    judge_note = "LLM judge not run yet"
    if judge_path.exists():
        jd = pd.read_csv(judge_path, dtype={"quid": str})
        same = set(jd[jd.same_meaning.astype(str).str.lower() == "yes"].quid)
        df["judged_same"] = df.index.isin(same)
        judge_note = (f"LLM judge: same meaning {len(same)}/{len(jd)}; valid and same "
                      f"{int((df.valid & df.judged_same).sum())}/{int(df.valid.sum())}")
        subsets += [("judged same meaning", df.judged_same), ("valid and judged same", df.valid & df.judged_same)]
    mc_path = RESULTS / "paraphrase_meaning_check.csv"
    meaning_note = "meaning check not filled yet"
    if mc_path.exists():
        mc = pd.read_csv(mc_path, dtype={"quid": str})
        filled = mc[mc.same_meaning.astype(str).str.lower().isin(["yes", "no"])]
        if len(filled):
            keep = set(filled[filled.same_meaning.astype(str).str.lower() == "yes"].quid)
            meaning_note = f"meaning kept in {len(keep)}/{len(filled)} checked valid paraphrases"
            if "judged_same" in df:
                fj = df.loc[filled.quid, "judged_same"].values
                fm = (filled.same_meaning.astype(str).str.lower() == "yes").values
                meaning_note += f"; agreement with the LLM judge {int((fj == fm).sum())}/{len(filled)}"
            subsets.append(("checked, same meaning", df.index.isin(keep)))
    A = pd.DataFrame([r for n, m in subsets for r in summarise(df, n, m)])
    L = pd.DataFrame([r for n, m in subsets for r in llmfree(df, n, m)])
    T = pd.DataFrame([r for n, m in subsets if n != "checked, same meaning" for r in paired(df, n, m)])
    A.to_csv(RESULTS / "paraphrase_robustness_anchors.csv", index=False)
    L.to_csv(RESULTS / "paraphrase_robustness_llmfree.csv", index=False)
    T.to_csv(RESULTS / "paraphrase_robustness_paired.csv", index=False)
    B = pd.DataFrame([r for n, m in subsets if n in ("valid", "valid and judged same") for r in breakdown(df, n, m)])
    B.to_csv(RESULTS / "paraphrase_robustness_breakdown.csv", index=False)
    df.to_csv(RESULTS / "paraphrase_robustness_per_question.csv")

    def fmt(t):
        t = t.copy()
        for c in t.columns:
            if c in ("bootstrap_p", "mcnemar_p"):
                t[c] = t[c].map(lambda v: f"{v:.1e}")
            elif t[c].dtype.kind == "f":
                t[c] = t[c].map(lambda v: f"{v:.3f}")
        return t.to_markdown(index=False, disable_numparse=True)
    n_valid = int(df.valid.sum())
    md = ["# Paraphrase robustness of anchor grounding (MultiTQ, pre-registered)", "",
          "Pre-registration: results/prereg_paraphrase_robustness.md (with Amendments 1 and 2). Sample: 600 multi-constraint test "
          "questions without a date or month name, 200 per type, seed 2026. Paraphrases: local Qwen2.5-7B-Instruct-4bit, "
          "greedy, one per question.", "",
          f"- valid paraphrases (automatic filter): {n_valid}/600", f"- {judge_note}", f"- {meaning_note}", "",
          "## Anchor detection, resolution, date agreement, slot agreement", "", fmt(A), "",
          "## LLM-free Hit@1 and coverage@15 (main setting: TEAR+window, lambda 0.8; ground = with grounding)", "",
          fmt(L), "", "## Paired tests on exact LLM-free Hit@1 (bootstrap 10,000, seed 2026)", "", fmt(T), "",
          "## Where the paraphrases lose hits (TEAR+window with grounding, exact Hit@1)", "", fmt(B), ""]
    (RESULTS / "paraphrase_robustness.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
