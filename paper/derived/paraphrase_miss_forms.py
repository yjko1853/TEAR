"""Forms of the anchor phrases that the patterns miss on the paraphrases (paraphrase robustness experiment).

Run from the repository root:  python3 -I paper/derived/paraphrase_miss_forms.py
Inputs : cache/multitq_para_src/paraphrases.json (paraphrase text, validity), results/paraphrase_robustness_per_question.csv
Output : paper/derived/paraphrase_miss_forms.csv  questions resolved on the original wording whose paraphrase has no
         detected anchor phrase, by subset (valid; valid and judged same meaning) and by form of the paraphrase:
         'preceding' (connective not in the pattern list), 'following mid-sentence' (', following X,'),
         'same period without preposition' ('..., the same year as X'), other
"""
import csv
import json
import re
from collections import Counter
from pathlib import Path

OUT = Path(__file__).resolve().parent
P = json.load(open("cache/multitq_para_src/paraphrases.json"))
rows = list(csv.DictReader(open("results/paraphrase_robustness_per_question.csv")))


def form(text):
    t = text.lower()
    if "preceding" in t:
        return "preceding"
    if re.search(r",\s*following", t):
        return "following mid-sentence"
    if "same" in t:
        return "same period without preposition"
    return "other"


out = []
for name, keep in (("valid", lambda r: r["valid"] == "True"),
                   ("valid and judged same", lambda r: r["valid"] == "True" and r["judged_same"] == "True")):
    miss = [r for r in rows if keep(r) and r["res_orig"] == "True" and r["det_para"] != "True"]
    c = Counter(form(P[r["quid"]]["paraphrase"]) for r in miss)
    for f in ("preceding", "following mid-sentence", "same period without preposition", "other"):
        out.append({"subset": name, "form": f, "count": c.get(f, 0), "missed_total": len(miss)})
with open(OUT / "paraphrase_miss_forms.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(out[0]))
    w.writeheader(); w.writerows(out)
for r in out:
    print(r)
