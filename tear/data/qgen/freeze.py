"""Building, saving and loading the frozen template file (MultiTQ train + dev questions only).

build_frozen:
  1. demonyms mined from the train questions (align.mine_demonyms)
  2. relation selection: dev questions aligned without a relation restriction; MultiTQ's paper says its questions use
     the 22 most frequent relations, and the relations with >= REL_SHARE of the aligned dev questions are kept
  3. train + dev questions aligned with the selected relations (two passes, align.align_all)
  4. templates merged; a template is usable when its frame keeps every slot the variant needs and no other KG entity
     name is left in the frame text. Generation weights a template by the number of *dev* questions it reproduced
     (WEIGHT_SPLIT): MultiTQ's dev and test questions share their wording statistics (mean length 10.0 / 10.2 words
     for equal, 12.96 / 13.02 for after_first; the paper's Table 6: 11.28 / 11.41 tokens) while train questions are
     longer and differently mixed (12.6 words for equal; first_last entity questions 3.8% of train, 1.2% of dev,
     1.1% of test). Train questions add templates for the record, and statistics below.
  5. bucket distribution: per type, the shares of (answer type, time level, variant) among *all* dev questions
     (variant read from the text: a date expression or not, a same-period phrase or not), so that alignment success
     rates do not shift the mix
  6. surface-form distribution of the replaced mentions (dev), article rates by entity and by last word (train+dev)
  7. the frame model and relation model counts (to align the test split later with exactly these models)
The file is written with sorted keys and its SHA-256 recorded next to it.
"""
from __future__ import annotations

import collections
import datetime as _dt
import gzip
import hashlib
import json
import re
from pathlib import Path

from . import semantics, timeexpr
from .aliases import AliasIndex
from .align import Aligner, FrameLM, RelationNB, align_all, form_counts, merge_templates, mine_demonyms
from .kg import TemporalKG

REL_SHARE = 0.002
VERSION = "v1"
WEIGHT_SPLIT = "dev"
_SAME = re.compile(r"\bsame\s+(day|month|year)\b", re.I)


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def text_variant(q: dict) -> str:
    """Variant of a MultiTQ question read from its text alone (used for the bucket distribution)."""
    qt, at = q["qtype"], q["answer_type"]
    if at == "time":
        return semantics.TIME_VARIANTS[qt][0]
    if qt == "equal":
        return "period"
    if qt == "first_last":
        return "extreme"
    if qt == "equal_multi":
        return "same" if _SAME.search(q["question"]) else "period_fl"
    return "date" if timeexpr.find(q["question"]) else "anchor"


def bucket_distribution(questions) -> dict:
    c = collections.defaultdict(collections.Counter)
    for q in questions:
        c[q["qtype"]]["|".join((q["answer_type"], q["time_level"], text_variant(q)))] += 1
    return {qt: {k: v / sum(cc.values()) for k, v in sorted(cc.items())} for qt, cc in sorted(c.items())}


def required_slots(t: dict) -> set:
    if t["answer_type"] == "time":
        req = {"S", "O"}
    else:
        req = {"O" if t["side"] == "subject" else "S"}
        if (t["qtype"], t["variant"]) in semantics.NEEDS_ANCHOR:
            req.add("Y")
    if (t["qtype"], t["variant"]) in semantics.NEEDS_TIME:
        req.add("T")
    return req


def leftover_entities(frame: str, alias: AliasIndex) -> list[str]:
    """KG entity names still written in a frame (capitalised mentions of >= 4 characters)."""
    text = re.sub(r"\{[A-Z]\}", " ", frame)
    out = []
    for m in alias.mentions(text):
        span = m["text"][len(m["article"]):]
        if len(span) >= 4 and span[:1].isupper() and m["start"] > 0:
            out.append(alias.entities[m["entity"]])
    return sorted(set(out))


def article_rates(recs, min_entity: int = 5, min_word: int = 20) -> tuple[dict, dict]:
    by_e = collections.defaultdict(lambda: [0, 0])
    by_w = collections.defaultdict(lambda: [0, 0])
    for r in recs:
        for s in r["slots"]:
            if s["ph"] == "T" or s["sector"]:
                continue
            the = int(bool(s["article"].strip()))
            by_e[s["name"]][0] += the; by_e[s["name"]][1] += 1
            last = s["name"].split()[-1].lower() if s["name"].split() else ""
            by_w[last][0] += the; by_w[last][1] += 1
    return ({e: round(a / n, 4) for e, (a, n) in sorted(by_e.items()) if n >= min_entity},
            {w: round(a / n, 4) for w, (a, n) in sorted(by_w.items()) if n >= min_word})


def relation_phrases(templates, k: int = 3) -> dict:
    """Most frequent wordings of each relation (dev-weighted): the 'equal' frames with the answer as subject,
    'Who <phrase> in/on {T}?', where <phrase> keeps the {O} slot ('made {O} suffer from conventional military force')."""
    c = collections.defaultdict(collections.Counter)
    for t in templates:
        if t["qtype"] != "equal" or t["variant"] != "period" or t["side"] != "subject" or not t.get("usable", True):
            continue
        m = re.match(r"^(?:Who|Which country)\s+(.*\{O\}.*?)\s+(?:in|on|at|during)\s+\{T\}\s*\?$", t["frame"])
        if m and len(m.group(1).split()) <= 10:
            c[t["relation"]][m.group(1)] += t["split_counts"].get(WEIGHT_SPLIT, 0)
    return {r: [p for p, n in cc.most_common(k) if n] for r, cc in sorted(c.items())}


def build_frozen(train, dev, kg: TemporalKG, sources: dict, log=print) -> dict:
    split_of = {}
    for q in train:
        split_of[q["quid"]] = "train"
    for q in dev:
        split_of[q["quid"]] = "dev"
    log("demonyms from train questions")
    demonyms, dem_evidence = mine_demonyms(train, kg, max_q=len(train))
    log(f"  {len(demonyms)} demonyms")
    alias = AliasIndex(kg.entities, demonyms)

    log("relation selection: dev questions, no relation restriction")
    _, rows_a = align_all(dev, Aligner(kg, alias), lambda q: "dev", log)
    rc = collections.Counter(r["relation"] for r in rows_a if r.get("relation"))
    tot = sum(rc.values())
    relations = sorted(r for r, n in rc.items() if n / tot >= REL_SHARE)
    rel_evidence = {r: {"aligned_dev": n, "share": round(n / tot, 5), "kept": r in relations}
                    for r, n in rc.most_common(40)}
    log(f"  kept {len(relations)} relations: {relations}")

    log("alignment of train + dev with the selected relations")
    al = Aligner(kg, alias, relations)
    recs, rows = align_all(train + dev, al, lambda q: split_of[q["quid"]], log)
    templates = merge_templates(recs)
    n_unusable = collections.Counter()
    for t in templates:
        miss = required_slots(t) - {s["ph"] for s in t["slots"]}
        left = leftover_entities(t["frame"], alias)
        t["usable"] = not miss and not left
        if miss:
            t["missing_slots"] = sorted(miss); n_unusable["missing_slot"] += t["count"]
        if left:
            t["leftover_entities"] = left; n_unusable["leftover_entity"] += t["count"]
        sc = t["status_counts"]
        t["status_major"] = max(sc, key=sc.get)
    forms = form_counts([r for r in recs if r["source"].startswith(WEIGHT_SPLIT + ":")])
    forms["article_by_entity"], forms["article_by_last_word"] = article_rates(recs)
    status = collections.defaultdict(collections.Counter)
    for r in rows:
        status[f"{r['split']}|{r['qtype']}|{r['answer_type']}"][r["status"]] += 1
    lm, nb = al.lm, al.nb
    frozen = {
        "version": VERSION,
        "created": _dt.datetime.now().isoformat(timespec="seconds"),
        "note": "Extracted from MultiTQ train and dev questions only; the test split was not read.",
        "sources": sources,
        "relations": relations, "relation_selection": {"min_share": REL_SHARE, "dev_alignment": rel_evidence},
        "demonyms": demonyms, "demonym_evidence": dem_evidence,
        "alias_forms": forms,
        "weight_split": WEIGHT_SPLIT,
        "bucket_distribution": bucket_distribution(dev),
        "bucket_distribution_train": bucket_distribution(train),
        "templates": templates,
        "unusable_question_counts": dict(n_unusable),
        "alignment_status": {k: dict(v) for k, v in sorted(status.items())},
        "relation_phrases": relation_phrases(templates),
        "models": {"lm": {"bi": [[a, b, n] for (a, b), n in lm.bi.items()], "uni": dict(lm.uni), "k": lm.k},
                   "nb": {"wc": {str(r): dict(c) for r, c in nb.wc.items()}, "dc": {str(r): n for r, n in nb.dc.items()}}},
    }
    return frozen, rows


def save_frozen(frozen: dict, path) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(frozen, ensure_ascii=False, sort_keys=True, default=_jsonable).encode("utf-8")
    with gzip.GzipFile(path, "wb", mtime=0) as f:
        f.write(data)
    digest = sha256(path)
    Path(str(path) + ".sha256").write_text(f"{digest}  {path.name}\n")
    return digest


def _jsonable(x):
    if isinstance(x, (set, frozenset)):
        return sorted(x)
    if hasattr(x, "item"):
        return x.item()
    raise TypeError(type(x))


def load_frozen(path, verify: bool = True) -> dict:
    path = Path(path)
    if verify:
        ref = Path(str(path) + ".sha256")
        if ref.exists():
            want = ref.read_text().split()[0]
            got = sha256(path)
            if got != want:
                raise ValueError(f"{path}: SHA-256 {got} differs from the frozen record {want}")
    with gzip.open(path, "rb") as f:
        return json.loads(f.read().decode("utf-8"))


def aligner_from_frozen(kg: TemporalKG, frozen: dict) -> Aligner:
    """An Aligner with the frozen relations, demonyms, frame model and relation model (for the MultiTQ test split)."""
    al = Aligner(kg, AliasIndex(kg.entities, frozen["demonyms"]), frozen["relations"])
    m = frozen["models"]
    lm = FrameLM([], k=m["lm"]["k"])
    for a, b, n in m["lm"]["bi"]:
        lm.bi[(a, b)] = n
        lm.ctx[a] += n
    lm.uni.update(m["lm"]["uni"])
    lm.N = sum(lm.uni.values()); lm.V = len(lm.uni) + 1
    nb = RelationNB([])
    for r, c in m["nb"]["wc"].items():
        nb.wc[int(r)].update(c)
    for r, n in m["nb"]["dc"].items():
        nb.dc[int(r)] = n
    vocab = set()
    for c in nb.wc.values():
        vocab.update(c)
    nb.V = len(vocab) + 1
    nb.tot = {r: sum(c.values()) for r, c in nb.wc.items()}
    nb.n = sum(nb.dc.values())
    al.lm, al.nb = lm, nb
    return al


def align_with_frozen(questions, kg, frozen, split: str, log=print):
    """Align questions (e.g. the MultiTQ test split) with the frozen models; one pass (the models are fixed)."""
    from .align import _row
    al = aligner_from_frozen(kg, frozen)
    rows = []
    for i, q in enumerate(questions):
        hyps = al.hypotheses(q)
        h, fr, st = al.resolve(q, hyps)
        rows.append(_row(q, h, st, split, kg))
        if (i + 1) % 20000 == 0:
            log(f"  {i + 1}/{len(questions)}")
    return rows
