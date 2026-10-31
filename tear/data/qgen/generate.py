"""The generator: frozen MultiTQ templates -> questions over any point-in-time event KG.

For a question type, a bucket (answer type, time level, variant) is drawn with MultiTQ's dev proportions within the
type (buckets of a time level that the KG does not span, e.g. 'year' for ICEWS18, are left out and the rest
renormalised), a template of the bucket with probability proportional to how many MultiTQ dev questions it reproduced (dev and
test share their wording statistics; see freeze.py), and a seed fact uniformly among the KG facts of the template's relation. The seed fact fixes the
question's entities and period:
  topic   = the seed fact's entity on the non-answer side (time questions: both entities)
  anchor  = the seed fact's entity on the answer side (anchored variants); t0 = the anchor event's earliest day
  period  = the day / month / year of the seed fact (dated variants), written in the template's date format
The answers come from semantics.py (the same procedures that reproduce MultiTQ's gold answers). A draw is rejected
when it has no answer, more than 20 answers (MultiTQ's filter), an 'equal' entity question whose time constraint does
not change the answer (MultiTQ's pseudo-temporal filter), an 'equal' time question whose triple spans several
periods, or a question text that was already generated; generator v2 (DEFAULT_CONFIG) adds MultiTQ dev's regularities:
one answer for first/last-style questions, and anchor events seen on one day only. Entity mentions are written in surface forms drawn from the
frozen form distribution (aliases.py).
"""
from __future__ import annotations

import collections
import random
import re

import numpy as np

from . import semantics, timeexpr
from .aliases import render as render_alias, split_sector
from .kg import TemporalKG, day_str

MAX_ANSWERS = 20
# Generator v2 (resources/qgen/generator_v2.json records the values and the MultiTQ dev statistics behind them):
#   unique_answer       buckets whose MultiTQ dev questions all have exactly one answer
#   anchor_single_day   MultiTQ dev anchors are events seen on one day (97.9% of the aligned anchored questions)
#   time_seed_alpha     a time question's (S, r, O) triple is drawn with probability ~ (number of facts)^alpha
#                       (alpha = 1: uniform over facts), calibrated so that regenerated ICEWS05-15 first/last time
#                       questions match MultiTQ dev's share of triples seen in a single period
#   role_signatures     shares of sector ('Government (Nigeria)') vs other entities in the named roles of MultiTQ dev
#                       questions: entity questions without an anchor never have a sector topic, a question never
#                       names two sector entities; each question draws a signature and keeps only matching seed facts
DEFAULT_CONFIG = {"version": "v2",
                  "unique_answer": [["after_first", "anchor", "entity"], ["before_last", "anchor", "entity"],
                                    ["first_last", "extreme", "entity"], ["equal_multi", "period_fl", "entity"],
                                    ["after_first", "date", "entity"], ["before_last", "date", "entity"]],
                  "anchor_single_day": True, "time_seed_alpha": 1.0, "role_signatures": None}
SIG_KEYS = ("entity|plain", "entity|anchor", "time")


def sig_key(qtype: str, variant: str, answer_type: str) -> str:
    if answer_type == "time":
        return "time"
    return "entity|anchor" if (qtype, variant) in semantics.NEEDS_ANCHOR else "entity|plain"


def signature(kg, names: dict) -> str:
    """Sector ('T') or other ('F') entity in each named role: topic[+anchor] (S/O or O; Y) or S+O for time questions."""
    order = [ph for ph in ("S", "O", "Y") if ph in names]
    return "".join("T" if split_sector(kg.name(names[ph])) else "F" for ph in order)


def bucket_of(t: dict) -> tuple:
    return (t["answer_type"], t["time_level"], t["variant"])


class FormSampler:
    """Surface forms of entity mentions drawn from the frozen distribution: sector names by (kind, number, case,
    article); other names with 'the' at the rate measured for the entity itself (seen in MultiTQ), else for names with
    the same last word, else the overall rate."""

    def __init__(self, forms: dict, rng: random.Random, demonyms: dict):
        self.rng, self.dem = rng, demonyms
        sec = sorted(forms["sector"].items())
        self.sec_keys = [k.split("|") for k, _ in sec]
        self.sec_w = [v for _, v in sec]
        self.by_entity = forms.get("article_by_entity", {})
        self.by_last = forms.get("article_by_last_word", {})
        p = forms["plain"]
        self.p_the = p.get("the", 0) / max(1, sum(p.values()))
        self.fallback = collections.Counter()

    def article_rate(self, name: str) -> float:
        if name in self.by_entity:
            return self.by_entity[name]
        last = name.split()[-1].lower() if name.split() else ""
        return self.by_last.get(last, self.p_the)

    def render(self, name: str) -> tuple[str, dict]:
        sec = split_sector(name)
        if sec:
            kind, number, case, art = self.rng.choices(self.sec_keys, self.sec_w)[0]
            if kind in ("of", "of_the"):          # 'of the Philippines' / 'of Spain': as the country itself is written
                kind = "of_the" if self.rng.random() < self.article_rate(sec[1]) else "of"
            form = {"kind": kind, "number": number, "case": case, "article": "" if art == "-" else "the "}
            txt = render_alias(name, form, self.dem)
            if txt is None:                       # adjective form without a known demonym
                self.fallback[kind] += 1
                form = dict(form, kind="of")
                txt = render_alias(name, form, self.dem)
            return txt, form
        form = {"kind": "plain", "article": "the " if self.rng.random() < self.article_rate(name) else ""}
        return render_alias(name, form), form


def fill(frame: str, values: dict) -> str:
    out = frame
    for ph, txt in values.items():
        out = out.replace("{" + ph + "}", txt)
    out = re.sub(r"\s+", " ", out).strip()
    out = re.sub(r"\s+([?,])", r"\1", out)
    return out[0].upper() + out[1:] if out else out


class Generator:
    def __init__(self, kg: TemporalKG, frozen: dict, seed: int = 2026, config: dict | None = None):
        self.kg, self.frozen = kg, frozen
        self.config = dict(DEFAULT_CONFIG, **(config or {}))
        self.unique = {tuple(x) for x in self.config["unique_answer"]}
        self.rng = random.Random(seed)
        self.forms = FormSampler(frozen["alias_forms"], self.rng, frozen["demonyms"])
        wk = frozen.get("weight_split")
        self.weight = (lambda t: t["split_counts"].get(wk, 0)) if wk else (lambda t: t["count"])
        usable = [t for t in frozen["templates"] if t.get("usable", True) and t["relation"] in kg.rid and self.weight(t)]
        self.by_bucket = collections.defaultdict(list)
        for t in usable:
            self.by_bucket[(t["qtype"],) + bucket_of(t)].append(t)
        # a time level is degenerate when the whole KG falls in one period of it (ICEWS18: Jan-Oct 2018, one year):
        # time answers would always be '2018', before/after constraints would have no answers and equal / same-period
        # constraints would not change the answers, so those buckets are left out and the type's mix renormalised
        self.degenerate_levels = [g for g in ("year", "month", "day")
                                  if len({timeexpr.period_label(int(d), g) for d in np.unique(kg.D)}) < 2]
        # their weight moves to the buckets with the same answer type and variant at the other levels (so the
        # shares of time answers and of dated before/after questions stay MultiTQ's); without such a bucket it is
        # dropped and the type renormalised
        self.bucket_p, self.excluded_buckets = {}, {}
        for qt, dist in frozen["bucket_distribution"].items():
            keep = {tuple(k.split("|")): w for k, w in dist.items()
                    if k.split("|")[1] not in self.degenerate_levels and (qt,) + tuple(k.split("|")) in self.by_bucket}
            excl = {k: w for k, w in dist.items() if tuple(k.split("|")) not in keep}
            for k, w in excl.items():
                at, _, var = k.split("|")
                sib = [b for b in keep if b[0] == at and b[2] == var]
                tot = sum(dist["|".join(b)] for b in sib)
                for b in sib:
                    keep[b] += w * dist["|".join(b)] / tot
            keys = sorted(keep)
            self.bucket_p[qt] = (keys, [keep[k] for k in keys])
            self.excluded_buckets[qt] = excl
        self._facts = {}
        self.seen_text = set()
        self.rejects = collections.Counter()

    def facts_of(self, r: int) -> np.ndarray:
        if r not in self._facts:
            self._facts[r] = self.kg.facts_of(r)
        return self._facts[r]

    def time_seed(self, r: int) -> int:
        """A fact whose (S, r, O) triple is drawn with probability ~ size^alpha (alpha = 1: uniform over facts)."""
        key = ("sro", r)
        if key not in self._facts:
            fs = self.facts_of(r)
            k = (self.kg.S[fs] * self.kg.NR + r) * self.kg.E + self.kg.O[fs]
            uniq, inv, cnt = np.unique(k, return_inverse=True, return_counts=True)
            first = np.zeros(len(uniq), np.int64)
            first[inv[::-1]] = fs[::-1]            # one fact per triple (the first in id order)
            w = cnt.astype(float) ** self.config["time_seed_alpha"]
            self._facts[key] = (first, np.cumsum(w) / w.sum())
        first, cum = self._facts[key]
        return int(first[min(int(np.searchsorted(cum, self.rng.random(), "right")), len(first) - 1)])

    # ------------------------------------------------------------------ one draw
    def draw(self, qtype: str, max_tries: int = 2000) -> dict | None:
        """The bucket is drawn once, then templates and seed facts are redrawn within it, so the mix of answer type,
        time level and variant follows the target distribution whatever the rejection rate of each bucket."""
        keys, w = self.bucket_p[qtype]
        b = self.rng.choices(keys, w)[0]
        ts = self.by_bucket[(qtype,) + b]
        tw = [self.weight(x) for x in ts]
        want = None
        sigs = (self.config.get("role_signatures") or {}).get(sig_key(qtype, b[2], b[0]))
        if sigs:
            ks = sorted(sigs)
            want = self.rng.choices(ks, [sigs[k] for k in ks])[0]
        for _ in range(max_tries):
            t = self.rng.choices(ts, tw)[0]
            q = self.instantiate(t, want_signature=want)
            if q is not None:
                return q
        self.rejects["bucket_exhausted"] += 1
        return None

    def instantiate(self, t: dict, fact: int | None = None, want_signature: str | None = None) -> dict | None:
        kg = self.kg
        r = kg.rid[t["relation"]]
        fs = self.facts_of(r)
        if not len(fs):
            self.rejects["no_fact"] += 1
            return None
        if fact is not None:
            f = int(fact)
        elif t["answer_type"] == "time" and self.config["time_seed_alpha"] != 1.0:
            f = self.time_seed(r)
        else:
            f = int(fs[self.rng.randrange(len(fs))])
        s, o, day = int(kg.S[f]), int(kg.O[f]), int(kg.D[f])
        qt, var, g = t["qtype"], t["variant"], t["time_level"]
        tslot = next((x for x in t["slots"] if x["ph"] == "T"), None)
        st = {"qtype": qt, "variant": var, "answer_type": t["answer_type"], "relation": r, "gran": g,
              "first": t.get("first"), "before": t.get("before")}
        if t["answer_type"] == "time":
            st.update(subject=s, object=o)
        else:
            st.update(side=t["side"], topic=o if t["side"] == "subject" else s)
            if (qt, var) in semantics.NEEDS_ANCHOR:
                st["anchor"] = s if t["side"] == "subject" else o
        if tslot is not None:
            st["lo"], st["hi"] = timeexpr.period_of(day, tslot["gran"])
        if t["answer_type"] == "time":
            names = {"S": s, "O": o}
        else:
            names = {{"subject": "O", "object": "S"}[t["side"]]: st["topic"]}
            if "anchor" in st:
                names["Y"] = st["anchor"]
        sig = signature(kg, names)
        if want_signature is not None and sig != want_signature:
            self.rejects["role_signature"] += 1
            return None
        res = semantics.evaluate(kg, st)
        ans = res["answers"]
        if not ans:
            self.rejects["no_answer"] += 1
            return None
        if len(ans) > MAX_ANSWERS:
            self.rejects["over_20_answers"] += 1
            return None
        if qt == "equal" and t["answer_type"] == "entity" and res.get("pseudo"):
            self.rejects["pseudo_temporal"] += 1
            return None
        if (qt, var, t["answer_type"]) in self.unique and len(ans) != 1:
            self.rejects["not_unique_answer"] += 1
            return None
        if "anchor" in st and self.config["anchor_single_day"] and res.get("anchor_n_days", 1) > 1:
            self.rejects["anchor_event_repeated"] += 1
            return None
        # ---- text
        values, forms, mentions = {}, {}, {}
        for ph, e in names.items():
            values[ph], forms[ph] = self.forms.render(kg.name(e))
            mentions[ph] = kg.name(e)
        if tslot is not None:
            values["T"] = timeexpr.render(day, tslot["shape"])
        text = fill(t["frame"], values)
        if text in self.seen_text:
            self.rejects["duplicate_text"] += 1
            return None
        self.seen_text.add(text)
        gen = {"template_id": t["template_id"], "template_status": t["status_major"], "frame": t["frame"],
               "relation": t["relation"], "variant": var,
               "answer_slot": "timestamp" if t["answer_type"] == "time" else t["side"],
               "seed_fact_id": f, "seed_fact": f"{kg.name(s)} | {t['relation']} | {kg.name(o)} | {day_str(day)}",
               "mentions": mentions, "forms": forms, "role_signature": sig}
        if t["answer_type"] == "time":
            gen.update(subject=kg.name(s), object=kg.name(o), n_periods=res["n_periods"], n_days=res["n_days"],
                       gold_fact_ids=[int(x) for x in kg.sro_fact_ids(s, r, o)])
        else:
            gen.update(topic_entity=kg.name(st["topic"]), gold_fact_ids=[int(x) for x in res["fact_ids"]],
                       n_partners=res["n_partners"], pseudo_temporal=bool(res["pseudo"]))
            if "anchor" in st:
                gen.update(anchor_entity=kg.name(st["anchor"]), anchor_date=day_str(res["t0"]),
                           anchor_fact_ids=[int(x) for x in res["anchor_fids"]],
                           anchor_n_days=res["anchor_n_days"])
        if tslot is not None:
            gen["time_expression"] = {"text": values["T"], "shape": tslot["shape"], "gran": tslot["gran"],
                                      "from": day_str(st["lo"]), "to": day_str(st["hi"])}
        for k in ("first", "before"):
            if t.get(k) is not None:
                gen[k] = t[k]
        return {"question": text, "answers": ans, "answer_type": t["answer_type"], "qtype": qt,
                "qlabel": semantics.QLABEL[qt], "time_level": g, "gen": gen}

    # ------------------------------------------------------------------ sets
    def generate(self, n_per_type: dict, quid_start: int, exclude_texts=()) -> list[dict]:
        self.seen_text |= set(exclude_texts)
        out = []
        for qt in sorted(n_per_type):
            got = 0
            while got < n_per_type[qt]:
                q = self.draw(qt)
                if q is None:
                    raise RuntimeError(f"cannot draw a {qt} question")
                out.append(q); got += 1
        for i, q in enumerate(out):
            q["quid"] = quid_start + i
        return out

