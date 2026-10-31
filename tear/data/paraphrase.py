"""Paraphrase robustness set for MultiTQ event-anchored questions (pre-registered: results/prereg_paraphrase_robustness.md).

The population is defined without the anchor patterns: test questions of the three multi-constraint types whose text
has no date and no month name, so the template fixes the time by another event. A local LLM rewrites each sampled
question once (greedy decoding); an automatic filter keeps rewrites that are one question, differ from the original and
keep every KG entity name the original mentions, and (Amendment 2) that express the same time relation
(constraint_classes). make_variant writes cache/multitq_para, which shares the fact files
with cache/multitq_orig; its 'rewrite' field (the stage text) holds the paraphrase and 'question' the original.
"""
from __future__ import annotations

import os
import random
import re
from pathlib import Path

from ..utils.io import load_json, save_json
from .multitq import SHARED_FILES

MULTI = ("after_first", "before_last", "equal_multi")
DATE = re.compile(r"\b(\d{4}-\d{2}-\d{2}|\d{4}-\d{2}|\d{4})\b")
MONTH = re.compile(r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\b", re.I)
EXTRA_SHARED = ("entity_emb.npy",)   # entity-name embeddings of scripts/14 (same entities)

PROMPT = (
    "Rewrite the question below so that it asks exactly the same thing in different words.\n"
    "Rules:\n"
    "- Ask for the same thing: keep the same question word (who, which, what) and the same kind of answer.\n"
    "- Keep the time relation itself: the asked event must still happen before, after, or in the same period as the "
    "same other event, and first or last must keep its meaning.\n"
    "- Keep every name of a person, organisation, group or country exactly as written, with the same spelling.\n"
    "- Change the sentence structure, and express the time relation (for example before, after, first, last, "
    "in the same month as) with different wording where you can.\n"
    "- Do not add or remove any information, condition or date.\n"
    "- Output only the rewritten question, as one sentence ending with a question mark.\n\n"
    "Question: {question}\n"
    "Rewritten question:")


def in_population(q: dict) -> bool:
    return q["qtype"] in MULTI and not DATE.search(q["question"]) and not MONTH.search(q["question"])


def draw_sample(questions: list, n_per: int = 200, seed: int = 2026, exclude: set | None = None) -> list[str]:
    """quids (str), n_per per multi-constraint type, from the population sorted by quid."""
    rng, out = random.Random(seed), []
    for t in MULTI:
        pool = sorted(str(q["quid"]) for q in questions if q["qtype"] == t and in_population(q)
                      and str(q["quid"]) not in (exclude or set()))
        out += rng.sample(pool, n_per)
    return out


def prompt_for(question: str) -> str:
    return PROMPT.format(question=question)


def clean_output(text: str) -> str:
    """First non-empty line of the model output without a leading label or surrounding quotes."""
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    s = lines[0] if lines else ""
    s = re.sub(r"^(rewritten question|question|paraphrase)\s*:\s*", "", s, flags=re.I)
    return s.strip().strip('"').strip("'").strip()


# Amendment 2 (results/prereg_paraphrase_robustness.md): the time relation must survive the paraphrase. The lexicon
# is broader than the anchor patterns of tear/labeling/anchor.py on purpose, so wordings those patterns miss still pass.
CONSTRAINT_CLASSES = {
    "after": r"\b(?:after|afterwards?|following|followed|later than|subsequent(?:ly)? to|since|once|succeeding|"
             r"in the wake of)\b",
    "before": r"\b(?:before|prior to|earlier than|preceding|preceded|ahead of|until|previous to|in advance of)\b",
    "first": r"\b(?:first|earliest|initial(?:ly)?|next|foremost)\b",
    "last": r"\b(?:last|latest|most recent(?:ly)?|final(?:ly)?|closest)\b",
    "same_day": r"\bsame (?:day|date)\b|\bthat (?:very )?day\b",
    "same_month": r"\bsame month\b|\bthat (?:very )?month\b",
    "same_year": r"\bsame year\b|\bthat (?:very )?year\b",
}
_CONSTRAINT_RX = {k: re.compile(v, re.I) for k, v in CONSTRAINT_CLASSES.items()}


def constraint_classes(text: str) -> set:
    return {k for k, rx in _CONSTRAINT_RX.items() if rx.search(text or "")}


def _norm(s: str) -> str:
    return " ".join(s.lower().split())


def check(original: str, para: str, mentions) -> dict:
    """Pre-registered validity filter. mentions: rules.build_mentioner(entities)."""
    lost = sorted(set(mentions(original)) - set(mentions(para))) if para else []
    reasons = []
    if not para or "\n" in para or not para.endswith("?"):
        reasons.append("format")
    if para and _norm(para) == _norm(original):
        reasons.append("unchanged")
    if lost:
        reasons.append("entity_lost")
    co, cp = constraint_classes(original), constraint_classes(para)
    if co != cp:
        reasons.append("constraint_changed")
    return {"valid": not reasons, "reasons": reasons, "lost_entities": lost,
            "constraint_orig": sorted(co), "constraint_para": sorted(cp)}


def make_variant(src_dir: Path, out_dir: Path, questions: list, paraphrases: dict) -> list:
    """cache/multitq_para: symlinked fact files and questions.json with the paraphrase as the stage text."""
    src, out = Path(src_dir).resolve(), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if out.resolve() == src:
        raise ValueError("the variant must not be written into the source directory")
    for name in SHARED_FILES + EXTRA_SHARED:
        link = out / name
        if not link.is_symlink() and (src / name).exists():
            link.symlink_to(os.path.relpath(src / name, out.resolve()))
    by_id = {str(q["quid"]): q for q in questions}
    qs = []
    for quid, p in paraphrases.items():
        q = dict(by_id[quid])
        q["rewrite"] = p["paraphrase"]
        q["paraphrase_valid"] = p["valid"]
        qs.append(q)
    save_json(qs, out / "questions.json")
    return qs


def load_paraphrases(path: Path) -> dict:
    return load_json(path) if Path(path).exists() else {}
