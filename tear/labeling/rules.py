"""Rule-based answer-slot classifier and topic-entity extractor for MultiTQ (training-free, API-free).

Answer slot = the position of the answer relative to the question's own verb phrase:
  timestamp : wh-time questions ("When ...", "In which month ...", "Could you tell me the exact month when ...")
  object    : do-support after the wh-phrase ("Which country did Cambodia denounce?"), passive with "by"
              ("Which country was first praised by Iraq?"), preposition + wh ("With whom did X ..."),
              relative clause with an overt subject ("Which was the last country Kuwait made negative comments about?")
  subject   : everything else ("Who was the first to visit Iraq?", "Who would want to negotiate with X?")
Versions: v1 = the first release; v2 (default) fixes three errors found in a human check of the labels:
  (a) do-support counts only right after the wh-phrase (wh-word + at most 3 words), so a trailing comparison clause
      such as "... before Cambodia did" no longer makes the answer an object;
  (b) "in" joins the prepositions that can precede the wh-word ("In which country did X ...?" -> object);
  (c) "Which was the last country with which / to which / that X ..." -> object;
  (d) a leading date clause with a comma inside the date ("After Sep, 2011, which country ...") is removed whole
      (v1 stopped at the first comma, so the question started with "2011," and defaulted to subject);
  (e) "suffer(ed) from" as the question's verb asks for the victim, the object of (attacker, Use ... force, victim);
      the causative "made X suffer from" still asks for the attacker (subject).
Topic entity = the longest KG entity mentioned in the main clause (leading constraint clause and trailing
before/after/same-period constraint removed), including ICEWS sector names such as "Men (Argentina)" for
"the men of Argentina"; falls back to the NER topic entity, then to the longest mention in the full question.
"""
from __future__ import annotations

import collections
import re

LEAD = re.compile(r'^\s*(before|after|in|on|during|since|until|from)\b[^,?]*,\s*', re.I)
TIME_START = re.compile(r'^(when|in which (year|month|day)|in what (year|month|day)|which (year|month|day|date)|'
                        r'what (year|month|day|date|time)|on which (date|day)|at what time)\b', re.I)
TIME_TELL = re.compile(r'^(could|can|would) you (please )?(tell me|let me know)\b.*\b(year|month|day|date|time)\b\s*'
                       r'(when|in which|that|did)\b', re.I)
MODAL_SUBJ = re.compile(r'^(who|whom|which \w+|what \w+)\s+(would|will|could|might|may|should|can)\s+'
                        r'(be|want|like|wish|make|have|go|visit|express|sign|seek|try)\b', re.I)
PASSIVE = re.compile(r'^(who|which|what)\b[^?]*\b(was|were|is|are|been)\b[^?]*\bby\b', re.I)
OBJ_AUX = re.compile(r'^((to|with|for|by|on|of|against|from|toward|towards|at|about) )?(whom|who|which|what)\b'
                     r'[^?]*?\b(did|does|do|would|will|could|might)\b', re.I)
PREP_WH = re.compile(r'^(to|with|for|by|on|of|against|from|toward|towards|at|about)\s+(whom|who|which|what)\b', re.I)
REL_CL = re.compile(r'^(which|who|what)\s+(was|is|were)\s+the\s+(first|last|next|final|only)\s+(\w+)\s+(.*)$', re.I)

# v2 corrections (see module docstring)
OBJ_AUX_V2 = re.compile(r"^((to|with|for|by|on|of|against|from|toward|towards|at|about|in) )?(whom|who|which|what)"
                        r"((\s+[\w/\-'()]+){0,3}?)\s+(did|does|do|would|will|could|might)\b", re.I)
PREP_WH_V2 = re.compile(r'^(to|with|for|by|on|of|against|from|toward|towards|at|about|in)\s+(whom|who|which|what)\b', re.I)
REL_CL_PREP_V2 = re.compile(r'^(which|who|what)\s+(was|is|were)\s+the\s+(first|last|next|final|only)\s+(\w+)\s+'
                            r'((to|with|for|by|on|of|against|from|about|at|in)\s+)?(that|which|whom)\s+(?!was\b|were\b|is\b)',
                            re.I)
LEAD_V2 = re.compile(r'^\s*(before|after|in|on|during|since|until|from)\b(?:\s+[A-Za-z]{3,9}\.?,\s*\d{4}\b)?[^,?]*,\s*', re.I)
SUFFER_V2 = re.compile(r'\bsuffer(?:s|ed|ing)?\s+from\b', re.I)
CAUSATIVE_SUFFER_V2 = re.compile(r'\b(?:make|makes|made|making)\b[^?]*\bsuffer', re.I)
RULE_VERSIONS = ("v1", "v2")

TRAIL = re.compile(r'\s*,?\s*\b(before|after|prior to|following|in the same (day|month|year)|on the same (day|month|year)|'
                   r'during the same|in|on|at|during|by the end of)\b\s+(?=(the same|\d|\w+ \d|january|february|march|april|'
                   r'may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)\b)'
                   r'.*$', re.I)
TRAIL_ENT = re.compile(r'\s*,?\s*\b(before|after|prior to|following)\b\s+(?!\d).*$', re.I)


# --------------------------------------------------------------------------- answer slot
def main_clause(q: str, version: str = "v2") -> str:
    """Strip up to two leading constraint clauses ('After 2006-01-05, ...', 'In 2014, ...', v2: 'After Sep, 2011, ...')."""
    lead = LEAD_V2 if version == "v2" else LEAD
    m = lead.sub('', q.strip())
    return lead.sub('', m).strip()


V2_FIXES = frozenset("abcde")     # letters as in the module docstring
FIXES = {"v1": frozenset(), "v2": V2_FIXES}


def rule_slot(q: str, version: str = "v2") -> str:
    if version not in RULE_VERSIONS:
        raise ValueError(version)
    return rule_slot_fixes(q, FIXES[version])


def rule_slot_fixes(q: str, fixes=V2_FIXES) -> str:
    """The classifier with an explicit subset of the v2 fixes (a-e); used to attribute label changes to each fix."""
    m = main_clause(q, "v2" if "d" in fixes else "v1")
    if TIME_START.match(m) or TIME_TELL.match(m):
        return 'timestamp'
    if "e" in fixes and SUFFER_V2.search(m) and not CAUSATIVE_SUFFER_V2.search(m):
        return 'object'
    if MODAL_SUBJ.match(m):
        return 'subject'
    if PASSIVE.match(m):
        return 'object'
    if "c" in fixes and REL_CL_PREP_V2.match(m):
        return 'object'
    rc = REL_CL.match(m)
    if rc:
        rest = rc.group(5)
        if re.match(r'(to|that was|that were|which was|who was)\b', rest, re.I):
            return 'subject'
        if re.match(r'((that|which|whom)\s+)?([A-Z]|the\b|his\b|her\b|its\b|their\b)', rest):
            return 'object'
    prep = PREP_WH_V2 if "b" in fixes else PREP_WH
    aux = OBJ_AUX_V2 if "a" in fixes else OBJ_AUX
    if prep.match(m) or aux.match(m):
        return 'object'
    return 'subject'


# --------------------------------------------------------------------------- topic entity
def _norm(s: str) -> str:
    return re.sub(r'\s+', ' ', re.sub(r'[^\w\s]', ' ', s.lower())).strip()


def _sing(w: str) -> str:
    return {'men': 'man', 'women': 'woman', 'people': 'people'}.get(w, w[:-1] if len(w) > 3 and w.endswith('s') else w)


def build_sector_matcher(entities):
    """'Men (Argentina)', 'Citizen (Nigeria)', 'Military Personnel (Philippines)': head words + parenthetical country."""
    idx = []
    for e in entities:
        m = re.match(r'^(.+?)\s*\((.+)\)$', e)
        if m:
            head = [_sing(w) for w in _norm(m.group(1)).split()]
            idx.append((e, set(head), _norm(m.group(2))))

    def match(text):
        t = _norm(text); words = {_sing(w) for w in t.split()}
        hits = [e for e, head, ctry in idx if head and head <= words and (' ' + ctry + ' ') in (' ' + t + ' ')]
        return hits
    return match


def build_mentioner(entities):
    """Literal n-gram lookup of KG entity names (normalised, length >= 3)."""
    names = {}
    for e in entities:
        n = _norm(e)
        if len(n) >= 3:
            names.setdefault(n, e)
    maxw = max(len(n.split()) for n in names)

    def mentions(text):
        w = _norm(text).split(); out = []
        for i in range(len(w)):
            for j in range(i + 1, min(len(w), i + maxw) + 1):
                span = ' '.join(w[i:j])
                if span in names:
                    out.append(names[span])
        return out
    return mentions


def rule_topic(q: str, mentions, ner_topic: str | None, sector=None, version: str = "v2"):
    m = main_clause(q, version)
    m = TRAIL.sub('', m)
    m = TRAIL_ENT.sub('', m)
    cands = mentions(m) + (sector(m) if sector else [])
    if cands:
        return max(cands, key=len), 'rule_main_clause'
    if ner_topic:
        return ner_topic, 'ner_fallback'
    cands = mentions(q)
    if cands:
        return max(cands, key=len), 'rule_full_question'
    return '', 'none'


# --------------------------------------------------------------------------- dataset-level
def label_multitq(questions, entities, ner_topics: dict | None = None, version: str = "v2") -> dict:
    """{quid: {"answer_slot", "topic_entity", "topic_source"}} from each question's stage text
    (the 'query' field when present, otherwise the TimeR4 rewrite; see tear.data.multitq.stage_query)."""
    from ..data.multitq import stage_query
    mentions = build_mentioner(entities)
    sector = build_sector_matcher(entities)
    out = {}
    for it in questions:
        q = stage_query(it)
        nt = ner_topics.get(str(it['quid']), {}).get('topic_entity') if isinstance(ner_topics, dict) else None
        topic, src = rule_topic(q, mentions, nt, sector, version)
        out[str(it['quid'])] = {'answer_slot': rule_slot(q, version), 'topic_entity': topic, 'topic_source': src}
    return out


def summarize(labels: dict) -> dict:
    return {"slots": dict(collections.Counter(v['answer_slot'] for v in labels.values())),
            "topic_source": dict(collections.Counter(v['topic_source'] for v in labels.values()))}
