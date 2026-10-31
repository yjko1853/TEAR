"""Answer-string matching, ported verbatim from the original TEAR code (predict_answer.py)."""
from __future__ import annotations

import re
import string

_PUNCT = set(string.punctuation)


def normalize(s) -> str:
    """Lower-case, drop punctuation, articles and <pad>, collapse whitespace."""
    s = str(s).lower()
    s = "".join(ch for ch in s if ch not in _PUNCT)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"\b(<pad>)\b", " ", s)
    return " ".join(s.split())


def match(s1, s2) -> bool:
    """True if normalize(s2) is a substring of normalize(s1) (s1 = prediction / fact element, s2 = gold answer)."""
    return normalize(s2) in normalize(s1)


def exact_match(pred, answers) -> bool:
    """Normalised string equality with any gold answer."""
    p = normalize(pred)
    return any(p == normalize(a) for a in answers)


def lenient_match(pred, answers) -> bool:
    """TEAR / TimeR4 rule: some gold answer is a substring of the prediction."""
    return any(match(pred, a) for a in answers)
