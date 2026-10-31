"""Surface forms of ICEWS entity names in questions.

ICEWS names sector actors 'Head (Country)' ('Government (Nigeria)', 'Citizen (Brazil)', 'Head of Government
(Greece)'); MultiTQ questions write them as 'the government of Nigeria', 'the citizens of Brazil', "Greece's head of
government", 'the Nigerian government', or verbatim. A form is (kind, number, case, article):
  kind     plain  'Government (Nigeria)' / 'Barack Obama'        of      '{h} of {C}'
           of_the '{h} of the {C}'                               poss    "{C}'s {h}"
           c_h    '{C} {h}'                                      in      '{h} in {C}'
           from   '{h} from {C}'                                 adj     '{Adj} {h}' (demonym learned from MultiTQ)
  number   sing / plur (plural of the head's last word; sector kinds only)
  case     orig / lower (head words lower-cased; sector kinds only)
  article  '' / 'the '
The forms and their frequencies are measured on aligned MultiTQ train/dev questions (align.py) and frozen; the
demonym table is mined from the same questions (align.mine_demonyms). `AliasIndex` finds entity mentions in a text
(normalised n-gram lookup) and reports the form of each.
"""
from __future__ import annotations

import re

SECTOR = re.compile(r"^(.+?)\s*\((.+)\)$")
KINDS = ("plain", "of", "of_the", "poss", "c_h", "in", "from", "adj")
SECTOR_KINDS = KINDS[1:]
_IRREG = {"man": "men", "woman": "women", "child": "children", "person": "people", "personnel": "personnel",
          "police": "police", "military": "military", "media": "media", "staff": "staff", "people": "people",
          "fishermen": "fishermen", "clergy": "clergy"}


def split_sector(name: str):
    """'Head (Country)' -> ('Head', 'Country'); None for other names."""
    m = SECTOR.match(name)
    return (m.group(1).strip(), m.group(2).strip()) if m else None


def plural_word(w: str) -> str:
    lw = w.lower()
    if lw in _IRREG:
        p = _IRREG[lw]
    elif lw.endswith("y") and len(lw) > 1 and lw[-2] not in "aeiou":
        p = lw[:-1] + "ies"
    elif lw.endswith("s") and not lw.endswith("ss"):     # already plural: 'Affairs', 'Forces'
        p = lw
    elif lw.endswith(("ss", "x", "z", "ch", "sh")):
        p = lw + "es"
    else:
        p = lw + "s"
    return p.capitalize() if w[:1].isupper() else p


def head_variant(head: str, number: str, case: str) -> str:
    words = head.split()
    if number == "plur" and words:
        words[-1] = plural_word(words[-1])
    h = " ".join(words)
    return h.lower() if case == "lower" else h


def norm(s: str) -> str:
    """Lower case, possessive 's split off, punctuation removed, whitespace collapsed."""
    s = s.replace("’", "'")
    s = re.sub(r"'s\b", " s", s)
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", s.lower())).strip()


def render(name: str, form: dict, demonyms: dict | None = None) -> str | None:
    """Surface text of entity `name` in `form` (dict kind/number/case/article); None if the form does not apply
    (sector kind for a plain name, adj without a known demonym)."""
    kind = form.get("kind", "plain")
    art = form.get("article", "")
    sec = split_sector(name)
    if kind == "plain":
        return art + name
    if not sec:
        return None
    h, c = sec
    hv = head_variant(h, form.get("number", "sing"), form.get("case", "orig"))
    if kind == "of":
        txt = f"{hv} of {c}"
    elif kind == "of_the":
        txt = f"{hv} of the {c}"
    elif kind == "poss":
        txt = f"{c}'s {hv}"
    elif kind == "c_h":
        txt = f"{c} {hv}"
    elif kind == "in":
        txt = f"{hv} in {c}"
    elif kind == "from":
        txt = f"{hv} from {c}"
    elif kind == "adj":
        adj = (demonyms or {}).get(c)
        if not adj:
            return None
        txt = f"{adj} {hv}"
    else:
        raise ValueError(kind)
    return art + txt


def all_forms(name: str, demonyms: dict | None = None):
    """(form dict without article, normalised text) for every applicable form of a name."""
    out = [({"kind": "plain"}, norm(name))]
    if split_sector(name):
        for kind in SECTOR_KINDS:
            for number in ("sing", "plur"):
                f = {"kind": kind, "number": number, "case": "orig"}
                t = render(name, f, demonyms)
                if t is not None:
                    out.append((f, norm(t)))
    return out


class AliasIndex:
    """Normalised alias n-gram -> {(entity id, kind, number)}; mentions(text) -> spans with entity and form."""

    def __init__(self, entities, demonyms: dict | None = None, min_len: int = 3):
        self.entities = entities
        self.idx = {}
        for i, e in enumerate(entities):
            for f, t in all_forms(e, demonyms):
                if len(t) >= min_len:
                    self.idx.setdefault(t, set()).add((i, f["kind"], f.get("number", "sing")))
        self.maxw = max(len(k.split()) for k in self.idx)

    @staticmethod
    def tokens(text: str):
        """Word tokens with character offsets, aligned with norm() (possessive 's is its own token)."""
        t = text.replace("’", "'")
        return [(m.group(0).lower(), m.start(), m.end()) for m in re.finditer(r"[^\W_]+|_", re.sub(r"'s\b", " s", t))
                if m.group(0) != "_"]

    def mentions(self, text: str) -> list[dict]:
        """Every alias occurrence: entity, kind, number, start/end (characters; a preceding 'the' is included and
        reported as the article), case of the head words."""
        toks = self.tokens(text)
        out = []
        n = len(toks)
        for i in range(n):
            for j in range(min(n, i + self.maxw), i, -1):
                key = " ".join(w for w, _, _ in toks[i:j])
                hits = self.idx.get(key)
                if not hits:
                    continue
                a, b = toks[i][1], toks[j - 1][2]
                art = ""
                if i > 0 and toks[i - 1][0] == "the":
                    art = "the "
                    a = toks[i - 1][1]
                span = text[a:b]
                for e, kind, number in hits:
                    out.append({"entity": e, "kind": kind, "number": number, "start": a, "end": b, "text": span,
                                "article": art, "case": _case(span, self.entities[e], kind)})
        return out


def _case(span: str, name: str, kind: str) -> str:
    sec = split_sector(name)
    if kind == "plain" or not sec:
        return "orig"
    head = sec[0]
    words = [w for w in re.findall(r"[A-Za-z]+", head)]
    span_words = re.findall(r"[A-Za-z]+", span)
    low = {w.lower() for w in words}
    hs = [w for w in span_words if w.lower() in low or w.lower().rstrip("s") in low]
    if hs and all(w[0].islower() for w in hs) and any(w[0].isupper() for w in words):
        return "lower"
    return "orig"
