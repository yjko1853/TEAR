"""Accuracy of the answer-slot rules against the author's blind check (questions 101-200 of the 300 sampled questions).

Run from the repository root:  .venv/bin/python -I paper/derived/rule_check_accuracy.py
Joins the blind check sheet (CHECK_SHEET, judged-slot column) to the quids of the 300-question file (CHECK_BASE) by 'no',
checks that the question text matches, and compares with slots_rule.json (v1) and slots_rule_v2.json (v2).
Questions without a judgement are excluded. Output: rule_check_accuracy.csv (accuracy, Wilson 95% interval).
The first author's check sheets are private working files and are not released; set TEAR_CHECK_DIR to the
folder that holds them (file, sheet and column names as in the original sheets) to recompute the check.
"""
import json
import math
import os
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parent
CHECK_DIR = Path(os.environ.get("TEAR_CHECK_DIR", "check"))   # first author's check sheets (not released)
CHECK_SHEET = CHECK_DIR / "규칙라벨_검수_2차_101-200.xlsx"     # blind sample, questions 101-200
CHECK_BASE = CHECK_DIR / "규칙라벨_검수_300.csv"               # the 300 sampled questions (no, quid, question)
SHEET, COL_JUDGED, COL_QUESTION = "검수", "판정_slot", "질문"     # sheet, judged-slot column, question column
if not (CHECK_SHEET.exists() and CHECK_BASE.exists()):
    raise SystemExit(f"check sheets not found in {CHECK_DIR} (set TEAR_CHECK_DIR); rule_check_accuracy.csv is kept as released")
chk = pd.read_excel(CHECK_SHEET, sheet_name=SHEET)
base = pd.read_csv(CHECK_BASE, encoding="utf-8-sig")
m = chk.merge(base[["no", "quid", "question"]], on="no", how="left", suffixes=("", "_300"))
same_q = (m[COL_QUESTION].str.strip() == m["question"].str.strip()).mean()
v1 = json.load(open("cache/multitq/slots_rule.json")); v2 = json.load(open("cache/multitq/slots_rule_v2.json"))
m["judged"] = m[COL_JUDGED].astype(str).str.strip().str.lower()
j = m[m["judged"].isin(["subject", "object", "timestamp"])].copy()


def wilson(k, n, z=1.96):
    p = k / n; den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


rows = []
for name, lab in (("v1", v1), ("v2", v2)):
    ok = [lab[str(q)]["answer_slot"] == s for q, s in zip(j["quid"], j["judged"])]
    k, n = sum(ok), len(ok); lo, hi = wilson(k, n)
    rows.append({"rules": name, "judged": n, "correct": k, "accuracy": k / n, "wilson_low": lo, "wilson_high": hi,
                 "unjudged": len(m) - len(j), "question_text_match_share": same_q})
res = pd.DataFrame(rows); res.to_csv(OUT / "rule_check_accuracy.csv", index=False); print(res.round(4).to_string(index=False))
print("disagreements v2:", [(int(r.no), r.judged, v2[str(r.quid)]["answer_slot"]) for r in j.itertuples() if v2[str(r.quid)]["answer_slot"] != r.judged])

# blind estimate as decided on 2026-10-17 (project log): rules with fixes a-c and the query fallback, i.e. before
# fixes d and e, which were derived from this check and therefore cannot be scored on it without bias
import sys  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root (tear/)
from tear.data.multitq import stage_query  # noqa: E402
from tear.labeling.rules import rule_slot_fixes  # noqa: E402
qv2 = {str(q["quid"]): q for q in json.load(open("cache/multitq/questions_v2.json"))}
ok = [rule_slot_fixes(stage_query(qv2[str(q)]), frozenset("abc")) == s for q, s in zip(j["quid"], j["judged"])]
k, n = sum(ok), len(ok); lo, hi = wilson(k, n)
blind = pd.DataFrame([{"rules": "v2_fixes_abc_plus_query_fallback (blind)", "judged": n, "correct": k, "accuracy": k / n,
                       "wilson_low": lo, "wilson_high": hi, "unjudged": len(m) - len(j), "question_text_match_share": same_q}])
pd.concat([res, blind]).to_csv(OUT / "rule_check_accuracy.csv", index=False)
print(blind.round(4).to_string(index=False))
