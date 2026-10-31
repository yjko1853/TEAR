"""Dates, temporal cue words and before/after target dates.

`extract_temporal_cue` and `extract_target_dates` are verbatim ports of the original TEAR code
(retrieval_temporal.Retrieval.extract_temporal_cue / extract_target_date). `extract_target_dates_extended` is an
extension (not in the original code) for the date forms that the TimeR4 rewritten questions actually use; parser v2
(`version="v2"`) also reads month abbreviations and punctuation ('Sep, 2011', 'Sept. 2011', 'Sep 5, 2011', '5 Sep 2011').
"""
from __future__ import annotations

import calendar
import re
from datetime import date

# Sentinel day numbers for TimelineKGQA open intervals ('beginning of time' / 'end of time').
NEG_INF_DAYS = -10**8
POS_INF_DAYS = 10**8
_EPOCH = date(1970, 1, 1)


def date_to_days(s: str) -> int | None:
    """'YYYY-MM-DD' -> days since 1970-01-01; None if not parseable."""
    try:
        y, m, d = s.split("-")
        return (date(int(y), int(m), int(d)) - _EPOCH).days
    except Exception:
        return None


def days_to_date(n: int) -> str:
    if n <= NEG_INF_DAYS:
        return "beginning of time"
    if n >= POS_INF_DAYS:
        return "end of time"
    return (date.fromordinal(_EPOCH.toordinal() + int(n))).isoformat()


# --------------------------------------------------------------------------- cue words (ported)
CUES = ["first", "last", "before", "after"]


def extract_temporal_cue(question: str) -> list[str]:
    """Substring test of each cue on the lower-cased question (original TEAR rule)."""
    q = question.lower()
    return [c for c in CUES if c in q]


# --------------------------------------------------------------------------- target dates
def extract_target_dates(question: str) -> list[dict]:
    """Original TEAR rule: 'before|after YYYY-MM-DD' pairs, before-matches first."""
    q = question.lower()
    out = []
    for prefix in ("before", "after"):
        for m in re.finditer(prefix + r"\s+(\d{4}-\d{2}-\d{2})", q):
            out.append({"type": prefix, "date": m.group(1)})
    return out


_MONTHS = {m: i + 1 for i, m in enumerate(["january", "february", "march", "april", "may", "june", "july", "august",
                                               "september", "october", "november", "december"])}


def _month_bounds(y: int, m: int, prefix: str) -> str:
    """'before <month>' -> first day of that month; 'after <month>' -> first day of the next month."""
    if prefix == "before":
        return f"{y:04d}-{m:02d}-01"
    ny, nm = (y + 1, 1) if m == 12 else (y, m + 1)
    return f"{ny:04d}-{nm:02d}-01"


def _last_day(y: int, m: int) -> str:
    return f"{y:04d}-{m:02d}-{calendar.monthrange(y, m)[1]:02d}"


def extract_target_dates_extended(question: str) -> list[dict]:
    """Also parses 'before|after 12 May 2011', 'May 12, 2011', 'May 2011', '2011-05', '2011' / 'the year 2011'.

    A partial date becomes the boundary day so that the strict comparison still holds
    (before May 2011 -> t < 2011-05-01; after May 2011 -> t > 2011-05-31).
    """
    q = question.lower()
    out = []
    seen = set()
    mon = "|".join(_MONTHS)
    pats = [
        (r"(before|after)\s+(\d{4}-\d{2}-\d{2})", "iso"),
        (r"(before|after)\s+(\d{1,2})\s+(" + mon + r")\s+(\d{4})", "dmy"),
        (r"(before|after)\s+(" + mon + r")\s+(\d{1,2}),?\s+(\d{4})", "mdy"),
        (r"(before|after)\s+(" + mon + r")\s+(\d{4})", "my"),
        (r"(before|after)\s+(\d{4})-(\d{2})(?!-)", "ym"),
        (r"(before|after)\s+(?:the\s+year\s+)?(\d{4})(?![-\d])", "y"),
    ]
    for pat, kind in pats:
        for m in re.finditer(pat, q):
            if m.start() in seen:
                continue
            prefix = m.group(1)
            try:
                if kind == "iso":
                    d = m.group(2)
                elif kind == "dmy":
                    d = f"{int(m.group(4)):04d}-{_MONTHS[m.group(3)]:02d}-{int(m.group(2)):02d}"
                elif kind == "mdy":
                    d = f"{int(m.group(4)):04d}-{_MONTHS[m.group(2)]:02d}-{int(m.group(3)):02d}"
                elif kind == "my":
                    y, mo = int(m.group(3)), _MONTHS[m.group(2)]
                    d = _month_bounds(y, mo, prefix) if prefix == "before" else _last_day(y, mo)
                elif kind == "ym":
                    y, mo = int(m.group(2)), int(m.group(3))
                    d = _month_bounds(y, mo, prefix) if prefix == "before" else _last_day(y, mo)
                else:
                    y = int(m.group(2))
                    d = f"{y:04d}-01-01" if prefix == "before" else f"{y:04d}-12-31"
                if date_to_days(d) is None:
                    continue
            except (KeyError, ValueError):
                continue
            seen.add(m.start())
            out.append({"type": prefix, "date": d, "form": kind})
    return out


# --------------------------------------------------------------------------- parser v2: abbreviations and punctuation
_MONTHS_V2 = dict(_MONTHS)
_MONTHS_V2.update({m[:3]: i for m, i in _MONTHS.items()})
_MONTHS_V2["sept"] = 9
_MON_V2 = "|".join(sorted(_MONTHS_V2, key=len, reverse=True))     # longest first: 'september' before 'sept' / 'sep'
_M = r"(" + _MON_V2 + r")\.?"                                        # month name or abbreviation, optional period


def extract_target_dates_v2(question: str) -> list[dict]:
    """Parser v2: everything extract_target_dates_extended reads, plus month abbreviations (3 letters, 'sept'), a period
    after the month and a comma between month and year:
      'before|after 2011-05-12' | '12 May 2011' / '5 Sep 2011' | 'May 12, 2011' / 'Sep 5, 2011' / 'Sep. 5th 2011'
      | 'May 2011' / 'Sep, 2011' / 'Sep. 2011' / 'Sept 2011' | '2011-05' | '2011' / 'the year 2011'.
    Boundaries as in the extended parser: before <month> -> the 1st of that month; after <month> -> its last day;
    a year alone -> January 1st / December 31st."""
    q = question.lower()
    out, seen = [], set()
    pats = [
        (r"(before|after)\s+(\d{4}-\d{2}-\d{2})", "iso"),
        (r"(before|after)\s+(\d{1,2})(?:st|nd|rd|th)?\s+" + _M + r",?\s+(\d{4})", "dmy"),
        (r"(before|after)\s+" + _M + r"\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})", "mdy"),
        (r"(before|after)\s+" + _M + r",?\s+(\d{4})", "my"),
        (r"(before|after)\s+(\d{4})-(\d{2})(?!-)", "ym"),
        (r"(before|after)\s+(?:the\s+year\s+)?(\d{4})(?![-\d])", "y"),
    ]
    for pat, kind in pats:
        for m in re.finditer(pat, q):
            if m.start() in seen:
                continue
            prefix = m.group(1)
            try:
                if kind == "iso":
                    d = m.group(2)
                elif kind == "dmy":
                    d = f"{int(m.group(4)):04d}-{_MONTHS_V2[m.group(3)]:02d}-{int(m.group(2)):02d}"
                elif kind == "mdy":
                    d = f"{int(m.group(4)):04d}-{_MONTHS_V2[m.group(2)]:02d}-{int(m.group(3)):02d}"
                elif kind == "my":
                    y, mo = int(m.group(3)), _MONTHS_V2[m.group(2)]
                    d = _month_bounds(y, mo, prefix) if prefix == "before" else _last_day(y, mo)
                elif kind == "ym":
                    y, mo = int(m.group(2)), int(m.group(3))
                    d = _month_bounds(y, mo, prefix) if prefix == "before" else _last_day(y, mo)
                else:
                    y = int(m.group(2))
                    d = f"{y:04d}-01-01" if prefix == "before" else f"{y:04d}-12-31"
                if date_to_days(d) is None:
                    continue
            except (KeyError, ValueError):
                continue
            seen.add(m.start())
            out.append({"type": prefix, "date": d, "form": kind})
    return out


DATE_PARSERS = ("v1", "v2")


def target_days(question: str, granularity: str = "full", parser: str = "v1") -> list[tuple[str, int]]:
    """[(before|after, day number)] from the question. granularity 'full' = original TEAR regex (parser ignored);
    'any' = extended parser, version v1 (first release) or v2 (abbreviations and punctuation)."""
    if granularity == "any":
        tds = extract_target_dates_v2(question) if parser == "v2" else extract_target_dates_extended(question)
    else:
        tds = extract_target_dates(question)
    out = [(x["type"], date_to_days(x["date"])) for x in tds]
    return [(a, b) for a, b in out if b is not None]


def extract_from_to_dates(question: str) -> tuple[str, str] | None:
    """TimelineKGQA simple questions: 'from X to Y' where X, Y are dates or open-interval words."""
    q = question.lower()
    tok = r"(\d{4}-\d{2}-\d{2}|beginning of time|end of time)"
    m = re.search(r"from\s+" + tok + r"\s+to\s+" + tok, q)
    if not m:
        return None
    return m.group(1), m.group(2)


# --------------------------------------------------------------------------- same-period constraints (equal window)
SAME_PERIOD = re.compile(r"\b(?:in|on|during|at)\s+the\s+same\s+(day|month|year)\s+(?:as|of|that)\s+", re.I)
_GRAN_ORDER = {"day": 0, "month": 1, "year": 2}


def same_period(question: str) -> tuple[str, str] | None:
    """'... in the same month as X ...' -> ('month', the text after 'as'/'of'); None without a same-period phrase."""
    m = SAME_PERIOD.search(question)
    return (m.group(1).lower(), question[m.end():]) if m else None


def explicit_same_period(question: str) -> tuple[str, int] | None:
    """A same-period constraint whose reference is written as a date, as in the TimeR4 rewrites ('in the same month of
    2015-09-03', 'on the same day as 2009-04-11', 'in the same month of 2011-11'): (granularity, day number). A
    reference coarser than the stated granularity widens the period ('same day of 2011-11' -> that month)."""
    sp = same_period(question)
    if not sp:
        return None
    m = re.match(r"\s*(\d{4})(?:-(\d{2}))?(?:-(\d{2}))?(?![\d-])", sp[1])
    if not m:
        return None
    y, mo, d = m.group(1), m.group(2) or "01", m.group(3) or "01"
    precision = "day" if m.group(3) else "month" if m.group(2) else "year"
    days = date_to_days(f"{y}-{mo}-{d}")
    if days is None:
        return None
    gran = max(sp[0], precision, key=_GRAN_ORDER.get)
    return gran, days


def period_bounds(days: int, granularity: str) -> tuple[int, int]:
    """Inclusive [first, last] day numbers of the day / month / year that contains `days`."""
    dt = date.fromordinal(_EPOCH.toordinal() + int(days))
    if granularity == "day":
        lo = hi = dt
    elif granularity == "month":
        lo, hi = dt.replace(day=1), dt.replace(day=calendar.monthrange(dt.year, dt.month)[1])
    else:
        lo, hi = date(dt.year, 1, 1), date(dt.year, 12, 31)
    return (lo - _EPOCH).days, (hi - _EPOCH).days
