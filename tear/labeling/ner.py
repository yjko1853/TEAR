"""Port of the original TEAR topic-entity extractor (process_test_data.py).

Pipeline (as in the original): dslim/bert-base-NER over the (rewritten) question -> strip a leading
"Before X," / "After X," clause -> hard-coded country list first (fuzzy > 70 against the KG entity vocabulary)
-> otherwise the best NER span by (0.7*fuzzy + 0.3*NER prob) * (1.2 if title-cased).
Documented deviations:
  labels="faithful" keeps label ids 1,2 only (= B-MISC/I-MISC in dslim/bert-base-NER, exactly as the original code);
  labels="all" keeps every non-O label (PER/ORG/LOC/MISC).
  engine="rapidfuzz" uses the same WRatio + default_process scorer as fuzzywuzzy's extractOne, ~100x faster.
"""
from __future__ import annotations

import re
import time

import numpy as np
import pandas as pd

from ..text import normalize

COMMON_COUNTRIES = [
    "Iraq", "China", "Iran", "North Korea", "South Korea", "Japan", "Russia",
    "United States", "India", "Pakistan", "Afghanistan", "Syria", "Israel",
    "Saudi Arabia", "Yemen", "Libya", "Egypt", "Sudan", "Ethiopia", "Somalia",
    "Kenya", "Uganda", "Rwanda", "Congo", "Nigeria", "South Africa", "Ukraine",
    "Turkey", "Germany", "France", "United Kingdom", "Italy", "Spain", "Thailand",
    "Cambodia", "Vietnam", "Malaysia", "Indonesia", "Philippines", "Myanmar",
    "Bangladesh", "Nepal", "Bhutan", "Sri Lanka", "Maldives", "Kazakhstan",
    "Uzbekistan", "Turkmenistan", "Kyrgyzstan", "Tajikistan", "Azerbaijan",
    "Armenia", "Georgia", "Mongolia", "Taiwan", "Laos", "Brunei", "Singapore",
    "East Timor", "Australia", "New Zealand", "Papua New Guinea", "Fiji",
    "Solomon Islands", "Vanuatu", "New Caledonia", "Canada", "Mexico", "Brazil",
    "Argentina", "Chile", "Peru", "Colombia", "Venezuela", "Ecuador", "Bolivia",
    "Paraguay", "Uruguay", "Guyana", "Suriname", "French Guiana", "Burundi",
    "Tanzania", "Mozambique", "Zimbabwe", "Zambia", "Angola", "Namibia",
    "Botswana", "Lesotho", "Swaziland", "Madagascar", "Mauritius", "Seychelles",
    "Comoros", "Djibouti", "Eritrea",
]
LABEL_IDS = {"faithful": {1, 2}, "all": {1, 2, 3, 4, 5, 6, 7, 8}}
NER_MODEL = "dslim/bert-base-NER"


def clean_entity(entity: str) -> str:
    entity = re.sub(r"[^\w\s]", " ", entity)
    return " ".join(entity.split())


def get_main_clause(text: str) -> str:
    return re.sub(r"^(Before|After)[^,]+,\s*", "", text)


def extract_countries(text: str):
    tl = text.lower()
    return [c for c in COMMON_COUNTRIES if c.lower() in tl]


class Matcher:
    """Cached best fuzzy match of a string against the KG entity vocabulary."""

    def __init__(self, choices, engine: str = "rapidfuzz"):
        self.choices = list(choices)
        self.engine = engine
        self.cache = {}
        if engine == "rapidfuzz":
            from rapidfuzz import fuzz, process, utils
            self._f = lambda q: process.extractOne(q, self.choices, scorer=fuzz.WRatio, processor=utils.default_process)
        else:
            from fuzzywuzzy import process
            self._f = lambda q: process.extractOne(q, self.choices)

    def __call__(self, q):
        if q not in self.cache:
            r = self._f(q)
            self.cache[q] = (r[0], float(r[1])) if r else None
        return self.cache[q]


def best_entity(spans, text, matcher):
    main = get_main_clause(text)
    for c in extract_countries(main):
        m = matcher(c)
        if m and m[1] > 70:
            return m[0], "country"
    if not spans:
        return "", "none"
    best = []
    for ent, prob in spans:
        is_person = any(w.istitle() for w in ent.split())
        m = matcher(ent)
        if m and m[1] > 70:
            best.append((m[0], (m[1] * 0.7 + prob * 100 * 0.3) * (1.2 if is_person else 1.0)))
    if best:
        return max(best, key=lambda x: x[1])[0], "ner"
    return "", "none"


def run_ner(texts, device, keep_ids, batch: int = 64):
    """Token-classification spans [(text, mean prob)] per question, decoded as in the original code."""
    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(NER_MODEL)
    model = AutoModelForTokenClassification.from_pretrained(NER_MODEL).to(device).eval()
    print("id2label:", model.config.id2label, flush=True)
    order = np.argsort([len(t) for t in texts])
    res = [None] * len(texts)
    with torch.no_grad():
        for lo in range(0, len(texts), batch):
            idx = order[lo:lo + batch]
            enc = tok([texts[i] for i in idx], return_tensors="pt", truncation=True, max_length=128, padding=True)
            logits = model(**{k: v.to(device) for k, v in enc.items()}).logits.float().cpu()
            probs = torch.softmax(logits, -1)
            preds = probs.argmax(-1)
            for row, i in enumerate(idx):
                toks = tok.convert_ids_to_tokens(enc["input_ids"][row])
                spans, cur, curp = [], [], []
                for t, p, pr in zip(toks, preds[row].tolist(), probs[row]):
                    if t in ("[CLS]", "[SEP]", "[PAD]"):
                        continue
                    if p in keep_ids:
                        if t.startswith("##"):
                            if cur:
                                cur.append(t[2:]); curp.append(float(pr[p]))
                        else:
                            if cur:
                                spans.append(("".join(cur), sum(curp) / len(curp))); cur, curp = [], []
                            cur.append(t); curp.append(float(pr[p]))
                    else:
                        if cur:
                            spans.append(("".join(cur), sum(curp) / len(curp))); cur, curp = [], []
                if cur:
                    spans.append(("".join(cur), sum(curp) / len(curp)))
                res[i] = [(clean_entity(e), pr) for e, pr in spans if clean_entity(e)]
            if (lo // batch) % 100 == 0:
                print(f"  NER {lo}/{len(texts)}", flush=True)
    return res


def topic_entities(qs, key, texts, spans, matcher) -> dict:
    out = {}
    for q, t, sp in zip(qs, texts, spans):
        te, how = best_entity(sp, t, matcher)
        out[str(q[key])] = {"topic_entity": te, "source": how, "ner_spans": [s for s, _ in sp]}
    return out


def agreement_table(qs, key, ner_out: dict, oracle: dict) -> pd.DataFrame:
    """Exact (normalised) agreement of NER topic entities with the KG-oracle topic entities, by question group."""
    rows = []
    for q in qs:
        k = str(q[key]); o = oracle[k]; n = ner_out[k]
        rows.append({"group": q.get("qtype", q.get("answer_type")), "answer_type": q["answer_type"],
                     "oracle_has": bool(o.get("topic_entity")), "ner_has": bool(n["topic_entity"]),
                     "exact": bool(o.get("topic_entity")) and normalize(o["topic_entity"]) == normalize(n["topic_entity"]),
                     "source": n["source"]})
    df = pd.DataFrame(rows)
    tab = df.groupby("group").agg(n=("exact", "size"), ner_nonempty=("ner_has", "mean"), oracle_nonempty=("oracle_has", "mean"),
                                  agree_exact=("exact", "mean"),
                                  from_country=("source", lambda s: (s == "country").mean()),
                                  from_ner=("source", lambda s: (s == "ner").mean())).reset_index()
    both = df[df.oracle_has]
    tab.loc[len(tab)] = ["ALL", len(df), df.ner_has.mean(), df.oracle_has.mean(), df.exact.mean(),
                         (df.source == "country").mean(), (df.source == "ner").mean()]
    tab["agree_exact_given_oracle"] = tab["group"].map(
        lambda g: both[both.group == g].exact.mean() if g != "ALL" else both.exact.mean())
    return tab


def fuzzy_engine_check(matcher: Matcher, entities, n_sample: int, seed: int = 0) -> str:
    """Agreement of rapidfuzz best matches with fuzzywuzzy extractOne on a sample of cached queries."""
    from fuzzywuzzy import process as fwp
    rng = np.random.default_rng(seed)
    qlist = list(matcher.cache)
    sample = [qlist[i] for i in rng.choice(len(qlist), min(n_sample, len(qlist)), replace=False)]
    t0 = time.time(); same = 0
    for s in sample:
        r = fwp.extractOne(s, entities)
        same += int(r is not None and matcher.cache[s] is not None and r[0] == matcher.cache[s][0])
    dt = time.time() - t0
    return (f"rapidfuzz vs fuzzywuzzy best-match agreement: {same}/{len(sample)} "
            f"(fuzzywuzzy {dt/len(sample):.2f}s per query)")
