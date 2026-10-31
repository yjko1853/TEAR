"""Alignment of MultiTQ questions to the KG and template extraction (train/dev questions only).

For a question with gold answers, every structure (answer side, relation, topic mention, [anchor mention],
[date expression], cue direction) whose answer procedure (semantics.py) reproduces the gold answer set *exactly* is a
hypothesis. Hypotheses are generated from the facts that link the answers to entities mentioned in the question
(aliases.AliasIndex), so only a handful are evaluated per question. When several relations reproduce the answers,
a naive-Bayes model of the question words given the relation, trained on the unambiguous questions, picks one if it is
clearly ahead (log-odds margin >= MARGIN); otherwise the question is left out. The chosen structure is delexicalised:
the topic / anchor / date spans become {S} / {O} / {Y} / {T}; the frame text, the relation, the type, the time level,
the date shape and the surface forms of the replaced mentions are kept. Identical frames are merged with a count.

Templates are extracted from MultiTQ *train and dev* questions; the test split is aligned only afterwards, for the
fidelity comparison (stats.py), with the same code.
"""
from __future__ import annotations

import collections
import math
import re

import numpy as np

from . import semantics, timeexpr
from .aliases import AliasIndex, split_sector
from .kg import TemporalKG

MARGIN = math.log(20.0)
_STOP = set("""a an the of to with in on at by for from about against who whom whose which what when where why how
did do does was were is are be been being would will could might may can should shall that this these those it its
his her their there here as and or but if than then so such any some one first last before after same day month year
time date country person exact tell me you could please let know""".split())
_FIRST = re.compile(r"\b(first|earliest|initially|initial|1st)\b", re.I)
_LAST = re.compile(r"\b(last|latest|final|finally|most recent|most recently|recently)\b", re.I)
_BEFORE = re.compile(r"\b(before|prior to|earlier than|until|ahead of|preceding)\b", re.I)
_AFTER = re.compile(r"\b(after|following|later than|since|subsequent to)\b", re.I)


def cue_flags(text: str) -> dict:
    return {"first": bool(_FIRST.search(text)), "last": bool(_LAST.search(text)),
            "before": bool(_BEFORE.search(text)), "after": bool(_AFTER.search(text))}


class FactIndex:
    """Fact ids by subject and by object (for hypothesis generation)."""

    def __init__(self, kg: TemporalKG):
        self.kg = kg
        self.by = {}
        for side, arr in (("s", kg.S), ("o", kg.O)):
            order = np.argsort(arr, kind="stable")
            self.by[side] = (order, arr[order])

    def facts(self, side: str, e: int) -> np.ndarray:
        order, srt = self.by[side]
        return order[np.searchsorted(srt, e, "left"):np.searchsorted(srt, e, "right")]


def _overlap(a, b) -> bool:
    return a["start"] < b["end"] and b["start"] < a["end"]


class Aligner:
    def __init__(self, kg: TemporalKG, alias: AliasIndex, relations=None):
        self.kg, self.alias, self.fx = kg, alias, FactIndex(kg)
        self.allowed = None if relations is None else {kg.rid[r] for r in relations if r in kg.rid}
        self.nb = None      # RelationNB, set after the unambiguous pass
        self.lm = None      # FrameLM, set after the unambiguous pass

    # ------------------------------------------------------------------ hypotheses
    def _candidates_entity(self, A, ments):
        """(side, r, topic) such that every answer has an r-fact with the topic in that layout."""
        me = {m["entity"] for m in ments}
        cand = None
        for a in A:
            cur = set()
            fs = self.fx.facts("s", a)          # a is the subject -> answer side subject, topic = object
            for f in fs[np.isin(self.kg.O[fs], list(me))]:
                cur.add(("subject", int(self.kg.R[f]), int(self.kg.O[f])))
            fo = self.fx.facts("o", a)
            for f in fo[np.isin(self.kg.S[fo], list(me))]:
                cur.add(("object", int(self.kg.R[f]), int(self.kg.S[f])))
            cand = cur if cand is None else cand & cur
            if not cand:
                return set()
        if self.allowed is not None:
            cand = {c for c in cand if c[1] in self.allowed}
        return cand

    def hypotheses(self, q: dict) -> list[dict]:
        text, qt, at, g = q["question"], q["qtype"], q["answer_type"], q["time_level"]
        ments = self.alias.mentions(text)
        times = [t for t in timeexpr.find(text) if not any(_overlap(t, m) for m in ments)]
        cues = cue_flags(text)
        out = []
        if at == "entity":
            if any(a not in self.kg.eid for a in q["answers"]):
                return []
            A = {self.kg.eid[a] for a in q["answers"]}
            for side, r, topic in self._candidates_entity(A, ments):
                ev = self.kg.events(side, r, topic)
                partners = None
                for variant in semantics.ENTITY_VARIANTS[qt]:
                    tlist = times if (qt, variant) in semantics.NEEDS_TIME else [None]
                    if (qt, variant) in semantics.NEEDS_ANCHOR:
                        partners = partners or ev.partner_set()
                        ylist = sorted({m["entity"] for m in ments if m["entity"] != topic and m["entity"] in partners})
                    else:
                        ylist = [None]
                    dirs = self._directions(qt, variant)
                    for t in tlist:
                        for y in ylist:
                            for d in dirs:
                                kw = dict(d)
                                if t is not None:
                                    kw.update(lo=t["lo"], hi=t["hi"])
                                a, sup, info = semantics.entity_answers(ev, qt, variant, anchor=y, gran=g, **kw)
                                if a and set(a) == A:
                                    out.append({"side": side, "r": r, "topic": topic, "variant": variant,
                                                "anchor": y, "time": t, **d, "t0": info.get("t0"),
                                                "anchor_n_days": info.get("anchor_n_days"), "fact_ids": sup})
        else:
            me = sorted({m["entity"] for m in ments})
            for s in me:
                fs = self.fx.facts("s", s)
                for o in me:
                    if o == s:
                        continue
                    rs = set(self.kg.R[fs[self.kg.O[fs] == o]].tolist())
                    if self.allowed is not None:
                        rs &= self.allowed
                    for r in rs:
                        days = self.kg.sro_days(s, r, o)
                        for variant in semantics.TIME_VARIANTS.get(qt, ()):
                            for d in self._directions(qt, variant):
                                ans = semantics.time_answers(days, qt, variant, g, d.get("first"))
                                if ans and set(ans) == set(q["answers"]):
                                    out.append({"side": "timestamp", "r": r, "subject": s, "object": o,
                                                "variant": variant, **d, "n_periods": len(
                                                    {timeexpr.period_label(int(x), g) for x in np.unique(days)})})
        for h in out:
            h["cues"] = cues
        return out

    @staticmethod
    def _directions(qt, variant):
        if qt in ("first_last",) or (qt == "equal_multi" and variant == "period_fl"):
            return [{"first": True}, {"first": False}]
        if qt == "before_after":
            return [{"before": True}, {"before": False}]
        return [{}]

    # ------------------------------------------------------------------ choice among hypotheses
    def resolve(self, q, hyps):
        """(hypothesis, frame, status). status: unique (one structure reproduces the answers), resolved (several
        structures; the frame model + relation model picked one with margin >= MARGIN), ambiguous, none (no
        structure reproduces the answers), delex_fail (a slot mention could not be placed)."""
        hyps = [h for h in hyps if self._cue_consistent(h)]
        if not hyps:
            return None, None, "none"
        by_key = {}
        for h in hyps:
            by_key.setdefault(self._structure_key(h), h)
        structs = list(by_key.values())
        if len(structs) == 1:
            fr = self.delexicalise(q, structs[0])
            return (structs[0], fr, "unique") if fr else (None, None, "delex_fail")
        if self.lm is None or self.nb is None:
            return None, None, "ambiguous"
        words = frame_words(q["question"])
        scored = []
        for h in structs:
            fr = self.delexicalise(q, h)
            if fr is None:
                continue
            scored.append((self.lm.score(fr["frame"]) + self.nb.score(words, h["r"]), id(h), h, fr))
        if not scored:
            return None, None, "delex_fail"
        scored.sort(key=lambda x: -x[0])
        if len(scored) > 1 and scored[0][0] - scored[1][0] < MARGIN:
            return None, None, "ambiguous"
        return scored[0][2], scored[0][3], "resolved"

    @staticmethod
    def _cue_consistent(h) -> bool:
        c = h["cues"]
        if "first" in h:
            want, other = ("first", "last") if h["first"] else ("last", "first")
            if not c[want] and c[other]:
                return False
            if not c[want] and not c[other]:
                return False
        if "before" in h:
            want, other = ("before", "after") if h["before"] else ("after", "before")
            if not c[want]:
                return False
        return True

    @staticmethod
    def _structure_key(h):
        t = h.get("time")
        return (h["side"], h["r"], h.get("topic"), h.get("subject"), h.get("object"), h.get("anchor"),
                (t["start"], t["end"]) if t else None, h["variant"], h.get("first"), h.get("before"))

    # ------------------------------------------------------------------ delexicalisation
    def delexicalise(self, q, h) -> dict | None:
        text = q["question"]
        ments = self.alias.mentions(text)
        slots = []          # (start, end, placeholder, form)
        if h["side"] == "timestamp":
            ents = [("S", h["subject"]), ("O", h["object"])]
        else:
            ents = [("O" if h["side"] == "subject" else "S", h["topic"])]
            if h.get("anchor") is not None:
                ents.append(("Y", h["anchor"]))
        taken = []
        if h.get("time"):
            t = h["time"]
            taken.append((t["start"], t["end"]))
            slots.append((t["start"], t["end"], "T", {"shape": t["shape"], "gran": t["gran"]}))
        for ph, e in ents:
            cands = sorted([m for m in ments if m["entity"] == e], key=lambda m: -(m["end"] - m["start"]))
            used = 0
            for m in cands:
                if any(m["start"] < b and a < m["end"] for a, b in taken):
                    continue
                taken.append((m["start"], m["end"]))
                slots.append((m["start"], m["end"], ph, {"kind": m["kind"], "number": m["number"], "case": m["case"],
                                                         "article": m["article"], "name": self.kg.name(e),
                                                         "sector": split_sector(self.kg.name(e)) is not None}))
                used += 1
            if not used:
                return None
        slots.sort()
        parts, pos, forms = [], 0, []
        for a, b, ph, form in slots:
            parts.append(text[pos:a]); parts.append("{" + ph + "}"); pos = b
            forms.append(dict(form, ph=ph))
        parts.append(text[pos:])
        frame = "".join(parts)
        if frame[:1].islower():
            frame = frame[0].upper() + frame[1:]
        return {"frame": frame, "slots": forms}


# ---------------------------------------------------------------------- relation word model
def frame_words(text: str) -> list[str]:
    t = re.sub(r"\{[A-Z]\}", " ", text)
    t = re.sub(r"\d+", " ", t)
    return [w for w in re.findall(r"[a-z]+", t.lower()) if w not in _STOP and len(w) > 2]


class RelationNB:
    """Multinomial naive Bayes over question words (add-one smoothing), trained on unambiguous alignments."""

    def __init__(self, docs: list[tuple[int, list[str]]], margin: float = MARGIN):
        self.wc = collections.defaultdict(collections.Counter)
        self.dc = collections.Counter()
        vocab = set()
        for r, ws in docs:
            self.dc[r] += 1
            self.wc[r].update(ws)
            vocab.update(ws)
        self.V = len(vocab) + 1
        self.tot = {r: sum(c.values()) for r, c in self.wc.items()}
        self.n = sum(self.dc.values())
        self.margin = margin

    def score(self, ws, r) -> float:
        if r not in self.dc:
            return -1e9
        c, tot = self.wc[r], self.tot[r]
        return math.log(self.dc[r] / self.n) + sum(math.log((c[w] + 1) / (tot + self.V)) for w in ws)

    def pick(self, ws, rels):
        sc = sorted(((self.score(ws, r), r) for r in rels), reverse=True)
        if len(sc) == 1 or sc[0][0] - sc[1][0] >= self.margin:
            return sc[0][1]
        return None

    def top_words(self, r, k=8, min_count=20):
        """Words most specific to relation r (log-ratio against the other relations)."""
        c = self.wc[r]
        allc = collections.Counter()
        for x in self.wc.values():
            allc.update(x)
        T = sum(allc.values())
        sc = []
        for w, n in c.items():
            if n < min_count:
                continue
            p_r = (n + 1) / (self.tot[r] + self.V)
            p_o = (allc[w] - n + 1) / (T - self.tot[r] + self.V)
            sc.append((math.log(p_r / p_o), w))
        return [w for _, w in sorted(sc, reverse=True)[:k]]


# ---------------------------------------------------------------------- demonyms
def mine_demonyms(questions, kg: TemporalKG, max_q: int = 200000, min_occ: int = 5, min_cover: float = 0.2,
                  max_ratio: float = 0.6):
    """Country -> adjective ('Kenya' -> 'Kenyan') from MultiTQ questions. For an entity question with an occurrence
    'the W h...' (W capitalised, not an entity name), the candidate countries are those C of sector entities 'h (C)'
    linked to every answer (any relation) whose head h follows W while C itself is not written in the question.
    W is C's adjective when W occurred >= min_occ times, C was a candidate in >= min_cover of them, and the runner-up
    country in at most max_ratio times as many (when W and C do not share their first two letters, e.g. 'British' /
    'United Kingdom': 0.2 times as many, >= 4 * min_occ occurrences, and W is not the first word of a multi-word
    entity name, which rules out 'Grand' (Grand National Party) for South Korea or 'Mahdi' (Mahdi Army))."""
    fx = FactIndex(kg)
    sector = {}
    for i, e in enumerate(kg.entities):
        sp = split_sector(e)
        if sp:
            sector[i] = sp
    names_lower = {e.lower() for e in kg.entities}
    first_words = {e.split()[0] for e in kg.entities if len(e.split()) > 1}   # 'Grand' (Grand National Party)
    occ = collections.Counter()
    cover = collections.defaultdict(collections.Counter)
    pat = re.compile(r"\bthe\s+([A-Z][a-z]+(?:-[A-Z][a-z]+)?)\s+([A-Za-z]+)")
    for q in questions[:max_q]:
        if q["answer_type"] != "entity" or any(a not in kg.eid for a in q["answers"]):
            continue
        text = q["question"]
        found = [(m.group(1), m.group(2).lower()) for m in pat.finditer(text)
                 if m.group(1).lower() not in names_lower and m.group(1).lower() not in _STOP]
        if not found:
            continue
        nb = None
        for a in q["answers"][:3]:
            ai = kg.eid[a]
            cur = set(kg.O[fx.facts("s", ai)].tolist()) | set(kg.S[fx.facts("o", ai)].tolist())
            nb = cur if nb is None else nb & cur
        low = text.lower()
        for w, nxt in found:
            cs = {sector[e][1] for e in nb or () if e in sector and sector[e][0].split()[0].lower() == nxt
                  and not re.search(r"\b" + re.escape(sector[e][1].lower()) + r"\b", low)}
            occ[w] += 1
            for c in cs:
                cover[w][c] += 1
    dem, evidence = {}, {}
    for w, n in occ.items():
        if n < min_occ or not cover[w]:
            continue
        (c1, n1), *rest = cover[w].most_common(2) + [(None, 0)]
        n2 = rest[0][1]
        similar = w[:2].lower() == c1[:2].lower()           # 'Kenyan' / 'Kenya'; irregular forms need strong evidence
        strong = n2 <= 0.2 * n1 and n >= 4 * min_occ and w not in first_words
        if n1 / n >= min_cover and (n2 <= max_ratio * n1 if similar else strong):
            evidence[w] = {"country": c1, "occurrences": n, "cover": round(n1 / n, 3), "second": round(n2 / n, 3)}
            if c1 not in dem or n > evidence[dem[c1]]["occurrences"]:
                dem[c1] = w
    return dict(sorted(dem.items())), evidence


# ---------------------------------------------------------------------- frame language model
def frame_tokens(frame: str) -> list[str]:
    return ["<s>"] + re.findall(r"\{[A-Z]\}|[a-z]+|[?,]", frame.lower().replace("{s}", "{S}").replace(
        "{o}", "{O}").replace("{y}", "{Y}").replace("{t}", "{T}")) + ["</s>"]


class FrameLM:
    """Interpolated bigram model of delexicalised frames (placeholders {S} {O} {Y} {T} are tokens), trained on the
    frames of unambiguous alignments. It decides between structures the KG cannot tell apart (reciprocal relations
    such as Make a visit / Host a visit, symmetric ones such as Engage in negotiation, swapped topic and anchor):
    'Who negotiated with {O} in {T}?' (answer = subject) is far more likely than 'Who negotiated with {S} in {T}?'."""

    def __init__(self, frames, k: float = 0.5):
        self.uni = collections.Counter()
        self.bi = collections.Counter()
        self.ctx = collections.Counter()
        for fr in frames:
            t = frame_tokens(fr)
            self.uni.update(t[1:])
            for a, b in zip(t, t[1:]):
                self.bi[(a, b)] += 1
                self.ctx[a] += 1
        self.N = sum(self.uni.values())
        self.V = len(self.uni) + 1
        self.k = k

    def score(self, frame: str) -> float:
        t = frame_tokens(frame)
        lp = 0.0
        for a, b in zip(t, t[1:]):
            pu = (self.uni[b] + 1) / (self.N + self.V)
            lp += math.log((self.bi[(a, b)] + self.k * self.V * pu) / (self.ctx[a] + self.k * self.V))
        return lp


# ---------------------------------------------------------------------- extraction
def template_record(q: dict, h: dict, fr: dict, kg: TemporalKG, status: str, split: str) -> dict:
    rec = {"qtype": q["qtype"], "answer_type": q["answer_type"], "time_level": q["time_level"],
           "variant": h["variant"], "relation": kg.relations[h["r"]], "side": h["side"], "frame": fr["frame"],
           "slots": fr["slots"], "source": f"{split}:{q['quid']}", "status": status}
    for k in ("first", "before"):
        if k in h:
            rec[k] = h[k]
    return rec


def template_key(rec: dict) -> tuple:
    shapes = tuple(s.get("shape", "") for s in rec["slots"] if s["ph"] == "T")
    return (rec["qtype"], rec["answer_type"], rec["time_level"], rec["variant"], rec["relation"], rec["side"],
            rec.get("first"), rec.get("before"), rec["frame"], shapes)


def align_all(questions, aligner: Aligner, split_of, log=print, every: int = 50000):
    """Two passes: unambiguous questions first (they train the frame and relation models), then the rest.
    Returns (records, per-question rows)."""
    kg = aligner.kg
    qtext = {}
    pending, recs, rows = [], [], []
    for i, q in enumerate(questions):
        qtext[f"{split_of(q)}:{q['quid']}"] = q["question"]
        hyps = aligner.hypotheses(q)
        for hh in hyps:
            hh.pop("fact_ids", None)
        h, fr, st = aligner.resolve(q, hyps)
        if st == "ambiguous":
            pending.append((q, hyps))
        else:
            rows.append(_row(q, h, st, split_of(q), kg))
            if h is not None:
                recs.append(template_record(q, h, fr, kg, st, split_of(q)))
        if every and (i + 1) % every == 0:
            log(f"  pass 1: {i + 1}/{len(questions)} aligned, {len(pending)} ambiguous")
    aligner.lm = FrameLM([r["frame"] for r in recs])
    aligner.nb = RelationNB([(kg.rid[r["relation"]], frame_words(qtext[r["source"]])) for r in recs])
    for q, hyps in pending:
        h, fr, st = aligner.resolve(q, hyps)
        rows.append(_row(q, h, st, split_of(q), kg))
        if h is not None:
            recs.append(template_record(q, h, fr, kg, st, split_of(q)))
    log(f"  pass 2: {len(pending)} ambiguous questions re-scored")
    return recs, rows


def _row(q, h, status, split, kg):
    row = {"quid": q["quid"], "split": split, "qtype": q["qtype"], "answer_type": q["answer_type"],
           "time_level": q["time_level"], "status": status, "n_answers": len(q["answers"])}
    if h is not None:
        row.update(relation=kg.relations[h["r"]], side=h["side"], variant=h["variant"],
                   topic=kg.name(h["topic"]) if h.get("topic") is not None else None,
                   subject=kg.name(h["subject"]) if h.get("subject") is not None else None,
                   object=kg.name(h["object"]) if h.get("object") is not None else None,
                   first=h.get("first"), before=h.get("before"), t0=h.get("t0"),
                   anchor=kg.name(h["anchor"]) if h.get("anchor") is not None else None,
                   anchor_n_days=h.get("anchor_n_days"), n_periods=h.get("n_periods"),
                   time_text=(h.get("time") or {}).get("text"))
    return row


def merge_templates(recs):
    """Identical frames (same type, relation, side, direction, time level, date shape) merged; count = all aligned
    questions, count_<split> per split; up to three source questions are kept as examples."""
    out = {}
    for r in recs:
        k = template_key(r)
        if k not in out:
            out[k] = dict(r, count=0, sources=[])
            out[k].pop("source")
            out[k].pop("status")
            out[k]["status_counts"] = collections.Counter()
            out[k]["split_counts"] = collections.Counter()
        t = out[k]
        t["count"] += 1
        t["status_counts"][r["status"]] += 1
        t["split_counts"][r["source"].split(":", 1)[0]] += 1
        if len(t["sources"]) < 3:
            t["sources"].append(r["source"])
    res = []
    for i, t in enumerate(sorted(out.values(), key=lambda t: (t["qtype"], t["answer_type"], t["variant"],
                                                               -t["count"], t["frame"]))):
        t["status_counts"] = dict(t["status_counts"])
        t["split_counts"] = dict(t["split_counts"])
        t["template_id"] = i
        res.append(t)
    return res


def form_counts(recs):
    """Surface-form frequencies of the replaced entity mentions: sector names by (kind, number, case, article),
    other names by article, and per entity name the article it was written with."""
    sector, plain = collections.Counter(), collections.Counter()
    for r in recs:
        for s in r["slots"]:
            if s["ph"] == "T":
                continue
            if s["sector"]:
                sector["|".join((s["kind"], s["number"], s["case"], s["article"].strip() or "-"))] += 1
            else:
                plain[s["article"].strip() or "-"] += 1
    return {"sector": dict(sector), "plain": dict(plain)}
