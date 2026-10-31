"""Date expressions in the formats MultiTQ questions use, with their *format* kept as a shape.

MultiTQ's time-generation module writes a day, month or year in many forms ('28 September 2007', 'Jun 5th, 2005',
'Jun, 2011', '2011-05', 'the 11th of March 2015', '15.08.2012', 'Nov, year 2007', ...). `find` returns each
expression with its span, granularity, day range and shape, e.g. '{Mon} {Dth}, {YYYY}'; `render` writes any day in
a shape. Shapes are extracted from MultiTQ train/dev questions and frozen with the templates, so generated questions
reproduce MultiTQ's format distribution (and therefore how often a date parser can read them).

Placeholders: {YYYY} year; {Month}/{month}/{MONTH} full month name (case kept), {Mon}/{mon} three-letter
abbreviation, {Sept} 'Sept'-style four-letter abbreviation (Sept / Sep. etc. fall back to {Mon}); {MM} two-digit
month, {M} month number; {DD} two-digit day, {D} day number, {Dth} ordinal day (1st, 2nd, 23rd, 11th).
"""
from __future__ import annotations

import calendar
import re
from datetime import date

MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
          "november", "december"]
MON_NUM = {m: i + 1 for i, m in enumerate(MONTHS)}
MON_NUM.update({m[:3]: i + 1 for i, m in enumerate(MONTHS)})
MON_NUM["sept"] = 9
_MON = "(?:" + "|".join(sorted(MON_NUM, key=len, reverse=True)) + r")"
_ORD = r"(?:st|nd|rd|th)"
_YEARWORDS = r"(?:(?:in\s+|of\s+)?(?:the\s+)?year\s+(?:of\s+)?)?"
_EPOCH = date(1970, 1, 1).toordinal()

# order matters: longer / more specific forms first; overlapping later matches are dropped
PATTERNS = [
    ("iso_d", r"(?P<y>\d{4})-(?P<m>\d{1,2})-(?P<d>\d{1,2})(?!\d)"),
    ("dot_d", r"(?P<d>\d{1,2})\.(?P<m>\d{1,2})\.(?P<y>\d{4})"),
    ("slash_d", r"(?P<d>\d{1,2})/(?P<m>\d{1,2})/(?P<y>\d{4})"),
    ("the_d_of_m_y", r"the\s+(?P<d>\d{1,2})" + _ORD + r"?\s+of\s+(?P<mn>" + _MON + r")\.?,?\s+" + _YEARWORDS +
     r"(?P<y>\d{4})"),
    ("d_m_y", r"(?P<d>\d{1,2})" + _ORD + r"?\s+(?:of\s+)?(?P<mn>" + _MON + r")\.?,?\s+" + _YEARWORDS + r"(?P<y>\d{4})"),
    ("m_d_y", r"(?P<mn>" + _MON + r")\.?\s+(?P<d>\d{1,2})" + _ORD + r"?,?\s+" + _YEARWORDS + r"(?P<y>\d{4})"),
    ("m_y", r"(?P<mn>" + _MON + r")\.?,?\s+" + _YEARWORDS + r"(?P<y>\d{4})"),
    ("iso_m", r"(?P<y>\d{4})-(?P<m>\d{1,2})(?![\d-])"),
    ("y", r"(?:the\s+year\s+(?:of\s+)?)?(?P<y>\d{4})(?![\d-])"),
]
_RX = [(k, re.compile(r"\b" + p, re.I)) for k, p in PATTERNS]


def ordinal(n: int) -> str:
    suf = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suf}"


def _month_ph(tok: str) -> str:
    t = tok.rstrip(".")
    if t.lower() in MONTHS:                  # full name ('May' is also its own abbreviation)
        base = "Month"
    elif t.lower() == "sept":
        base = "Sept"
    else:
        base = "Mon"
    if t.isupper():
        return "{" + base.upper() + "}"
    if t[0].islower():
        return "{" + base.lower() + "}"
    return "{" + base + "}"


def _shape(m: re.Match) -> str:
    text, a0 = m.group(0), m.start()
    parts, pos = [], 0
    spans = []
    for g in ("y", "m", "d", "mn"):
        if m.groupdict().get(g) is not None:
            spans.append((m.start(g) - a0, m.end(g) - a0, g))
    for a, b, g in sorted(spans):
        parts.append(text[pos:a])
        tok = text[a:b]
        if g == "y":
            parts.append("{YYYY}")
        elif g == "m":
            parts.append("{MM}" if len(tok) == 2 and tok[0] == "0" else ("{M}" if len(tok) == 1 else "{MM}"))
        elif g == "mn":
            parts.append(_month_ph(tok))
        else:
            rest = text[b:b + 2].lower()
            if rest in ("st", "nd", "rd", "th"):
                parts.append("{Dth}")
                b += 2
            else:
                parts.append("{DD}" if len(tok) == 2 and tok[0] == "0" else "{D}")
        pos = b
    parts.append(text[pos:])
    return re.sub(r"\s+", " ", "".join(parts))


def find(text: str) -> list[dict]:
    """Date expressions in text: start, end, kind, gran (day/month/year), lo/hi (day numbers), shape, text."""
    out, taken = [], []
    for kind, rx in _RX:
        for m in rx.finditer(text):
            a, b = m.span()
            if any(a < tb and ta < b for ta, tb in taken):
                continue
            g = m.groupdict()
            try:
                y = int(g["y"])
                if not 1900 <= y <= 2100:
                    continue
                if g.get("m"):
                    mo = int(g["m"])
                elif g.get("mn"):
                    mo = MON_NUM[g["mn"].lower()]
                else:
                    mo = None
                d = int(g["d"]) if g.get("d") else None
                if d is not None:
                    lo = hi = date(y, mo, d); gran = "day"
                elif mo is not None:
                    lo, hi, gran = date(y, mo, 1), date(y, mo, calendar.monthrange(y, mo)[1]), "month"
                else:
                    lo, hi, gran = date(y, 1, 1), date(y, 12, 31), "year"
            except (ValueError, KeyError):
                continue
            taken.append((a, b))
            out.append({"start": a, "end": b, "kind": kind, "gran": gran, "lo": lo.toordinal() - _EPOCH,
                        "hi": hi.toordinal() - _EPOCH, "shape": _shape(m), "text": m.group(0)})
    return sorted(out, key=lambda x: x["start"])


def shape_granularity(shape: str) -> str:
    if any(k in shape for k in ("{D}", "{DD}", "{Dth}")):
        return "day"
    if re.search(r"\{(M|MM|Month|month|MONTH|Mon|mon|MON|Sept|sept|SEPT)\}", shape):
        return "month"
    return "year"


def render(day: int, shape: str) -> str:
    """Write day number `day` in `shape` (a month or year shape writes the month / year that contains the day)."""
    dt = date.fromordinal(_EPOCH + int(day))
    full = calendar.month_name[dt.month]
    abbr = calendar.month_abbr[dt.month]
    sept = "Sept" if dt.month == 9 else abbr
    rep = {"{YYYY}": f"{dt.year:04d}", "{MM}": f"{dt.month:02d}", "{M}": str(dt.month), "{DD}": f"{dt.day:02d}",
           "{D}": str(dt.day), "{Dth}": ordinal(dt.day), "{Month}": full, "{month}": full.lower(),
           "{MONTH}": full.upper(), "{Mon}": abbr, "{mon}": abbr.lower(), "{MON}": abbr.upper(), "{Sept}": sept,
           "{sept}": sept.lower(), "{SEPT}": sept.upper()}
    return re.sub(r"\{[A-Za-z]+\}", lambda m: rep[m.group(0)], shape)


def period_of(day: int, gran: str) -> tuple[int, int]:
    """Inclusive [lo, hi] day numbers of the day / month / year containing `day`."""
    dt = date.fromordinal(_EPOCH + int(day))
    if gran == "day":
        lo = hi = dt
    elif gran == "month":
        lo, hi = dt.replace(day=1), dt.replace(day=calendar.monthrange(dt.year, dt.month)[1])
    else:
        lo, hi = date(dt.year, 1, 1), date(dt.year, 12, 31)
    return lo.toordinal() - _EPOCH, hi.toordinal() - _EPOCH


def period_label(day: int, gran: str) -> str:
    """MultiTQ's time answer at a granularity: 'YYYY-MM-DD', 'YYYY-MM' or 'YYYY'."""
    s = date.fromordinal(_EPOCH + int(day)).isoformat()
    return {"day": s, "month": s[:7], "year": s[:4]}[gran]
