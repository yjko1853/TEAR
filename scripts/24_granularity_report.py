"""Answer granularity (pre-registered: results/prereg_retriever_and_granularity.md, part B).

Variants on the 6,000-question reader sample (exact Hit@1 on the first answer):
  A  v3 as read                     B  v3 + projection onto the predicted unit
  C  v3g as read                    D  v3g + projection
The predicted unit comes from the rule-v2 slot and the question wording (tear/labeling/granularity.py); no gold label.
v3g outputs exist for the questions with a predicted date answer; for the others the v3g prompt equals v3 and the v3
output is used. Also: agreement of the predicted unit with the gold answers on all test time questions.

Output: results/granularity.md, results/granularity_{summary,paired,per_question}.csv
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from tear.data.multitq import stage_query  # noqa: E402
from tear.evaluation.significance import mcnemar, paired_bootstrap  # noqa: E402
from tear.labeling.granularity import UNITS, asked_granularity, asked_unit, project  # noqa: E402
from tear.paths import CACHE, RESULTS  # noqa: E402
from tear.reader.parse import parse_list  # noqa: E402
from tear.text import normalize  # noqa: E402
from tear.utils.io import load_json  # noqa: E402

D, PRED = CACHE / "multitq_orig", RESULTS / "qwen_ground"
TAG, MODEL = "rule_v2_ground_exact_anydate", "Qwen2.5-7B-Instruct-4bit"
SETTINGS = ["none", "tear_window"]


def ok(pred, answers):
    return any(normalize(pred) == normalize(a) for a in answers)


def read(setting, prompt):
    p = PRED / f"{setting}_{TAG}__{MODEL}__k15__p-{prompt}.jsonl"
    return {json.loads(l)["quid"]: json.loads(l) for l in p.read_text().splitlines() if l.strip()}


def first(rec):
    lst = parse_list(rec["prediction"])
    return lst[0] if lst else ""


def main():
    Q = {str(q["quid"]): q for q in load_json(D / "questions.json")}
    S = load_json(D / "slots_rule_v2.json")
    sample = [str(x) for x in load_json(D / "sample_6000.json")["quids"]]
    rows = []
    for s in SETTINGS:
        v3, v3g = read(s, "v3"), read(s, "v3g")
        for k in sample:
            q = Q[k]
            g = asked_granularity(stage_query(q), S[k]["answer_slot"])
            if g and k not in v3g:
                raise SystemExit(f"{s}: v3g output missing for {k}")
            a = first(v3[k]); c = first(v3g[k]) if g else a
            b = project(a, g) if g else a; d = project(c, g) if g else c
            rows.append({"setting": s, "quid": k, "qtype": q["qtype"], "gold_time": q["answer_type"] == "time",
                         "pred_unit": g, "A": ok(a, q["answers"]), "B": ok(b, q["answers"]),
                         "C": ok(c, q["answers"]), "D": ok(d, q["answers"])})
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "granularity_per_question.csv", index=False)
    summ, paired = [], []
    for s, g in df.groupby("setting"):
        for name, h in (("all 6,000", g), ("gold time questions", g[g.gold_time])):
            r = {"setting": s, "subset": name, "n": len(h), **{v: h[v].mean() for v in "ABCD"}}
            r["wrong_unit_only_v3"] = (h.B & ~h.A).mean()
            r["wrong_unit_only_v3g"] = (h.D & ~h.C).mean()
            summ.append(r)
            for x, y in (("C", "A"), ("D", "B"), ("B", "A")):
                bs, mc = paired_bootstrap(h[y].values, h[x].values, seed=2026), mcnemar(h[y].values, h[x].values)
                paired.append({"setting": s, "subset": name, "system": x, "baseline": y, "n": len(h), "diff": bs["diff"],
                               "ci_low": bs["ci_low"], "ci_high": bs["ci_high"], "mcnemar_p": mc["p_value"],
                               "gained": mc["n01"], "lost": mc["n10"]})
    tq = [q for q in Q.values() if q["answer_type"] == "time"]
    unit_ok = sum(len(q["answers"][0]) == UNITS[asked_unit(q["question"])][1] for q in tq)
    slot_ok = sum(S[str(q["quid"])]["answer_slot"] == "timestamp" for q in tq)
    S_, P_ = pd.DataFrame(summ), pd.DataFrame(paired)
    S_.to_csv(RESULTS / "granularity_summary.csv", index=False)
    P_.to_csv(RESULTS / "granularity_paired.csv", index=False)

    def fmt(t):
        t = t.copy()
        for c in t.columns:
            if c == "mcnemar_p":
                t[c] = t[c].map(lambda v: f"{v:.1e}")
            elif t[c].dtype.kind == "f":
                t[c] = t[c].map(lambda v: f"{v:.3f}")
        return t.to_markdown(index=False, disable_numparse=True)
    md = ["# Answer granularity (MultiTQ reader sample, Qwen2.5-7B-Instruct-4bit, k = 15)", "",
          "Pre-registered: results/prereg_retriever_and_granularity.md (B). A = v3, B = v3 + projection, C = v3g, "
          "D = v3g + projection; the unit is predicted by the rules (no gold label).", "",
          f"- predicted unit matches the gold answers on {unit_ok}/{len(tq)} test time questions ({unit_ok / len(tq):.4f})",
          f"- rule-v2 slot is timestamp for {slot_ok}/{len(tq)} test time questions ({slot_ok / len(tq):.4f})", "",
          "## Exact Hit@1", "", fmt(S_), "", "## Paired tests (bootstrap 10,000, seed 2026)", "", fmt(P_), ""]
    (RESULTS / "granularity.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
