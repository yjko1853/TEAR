"""Answer granularity: the second part of the answer type (answer slot + date granularity), read from the question.

For a question whose predicted answer slot is the timestamp (rules.label_multitq, rule v2), the granularity is the
unit the question asks for: 'year' when the question says year ("In which year ...", "What year ..."), 'month' when it
says month, otherwise 'day' ("When ...", "On which day ..."). MultiTQ's gold time answers follow this unit for 99.95% of
the test questions (scripts/24_granularity_report.py checks it). No LLM, no training, no gold label.
"""
from __future__ import annotations

import re

UNITS = {"year": ("YYYY", 4), "month": ("YYYY-MM", 7), "day": ("YYYY-MM-DD", 10)}
_YEAR, _MONTH = re.compile(r"\byear\b", re.I), re.compile(r"\bmonth\b", re.I)


def asked_unit(question: str) -> str:
    if _YEAR.search(question):
        return "year"
    if _MONTH.search(question):
        return "month"
    return "day"


def asked_granularity(question: str, answer_slot: str | None) -> str | None:
    """'year' / 'month' / 'day' for a predicted timestamp answer, None for entity answers."""
    return asked_unit(question) if answer_slot == "timestamp" else None


def project(answer: str, unit: str) -> str:
    """Deterministic projection of a date answer onto the asked unit (first date in the answer); other text unchanged."""
    m = re.search(r"(\d{4})(?:-(\d{2}))?(?:-(\d{2}))?", answer or "")
    if not m:
        return answer
    s = m.group(1) + ("-" + m.group(2) if m.group(2) else "") + ("-" + m.group(3) if m.group(3) else "")
    n = UNITS[unit][1]
    return s[:n] if len(s) >= n else answer
