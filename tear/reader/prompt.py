"""Reader prompts.

  timer4  reasoning prompt of TimeR4 (Qian et al., EMNLP 2024), also Figure 5 of the TEAR paper; written for a reader
          fine-tuned on the task
  v2      zero-shot instruction prompt for readers that were not fine-tuned (same text for every reranker). Rejected on
          the tuning set: the unconditional date-format rule made the 7B reader answer dates to entity questions
  v3      v2 with the answer form made conditional on the question (dates only for when / which year / which month
          questions, entities otherwise)
  v3i     v3 for interval facts (TimelineKGQA: 's p o from <start> to <end>'): only the fact-format line and the
          time-constraint line are rewritten for time spans; chosen against v3 on a validation tuning set with the
          'none' reranker (logs/run_timeline_reader.sh, logs/timeline_prompt_choice.json)
  v3g     v3 plus one line for questions whose predicted answer is a date (answer type = slot + granularity,
          tear/labeling/granularity.py): "This question asks for a year: answer with the year only, written as YYYY."
          For entity questions the prompt is v3 itself, so v3 outputs are reused for them (greedy decoding).
Prompts are compared on a tuning set disjoint from the evaluation sample, then one is frozen for every reranker.
Selection (2026-10-17, cache/multitq/tune_120.json, 20 per qtype, Qwen2.5-7B-Instruct-4bit, 'none' reranker so the
choice cannot favour TEAR): Hit@1 exact timer4 0.242 / v2 0.150 / v3 0.242; lenient 0.242 / 0.200 / 0.325 -> v3 frozen.
"""
from __future__ import annotations

PROMPT = ("Based on the historical facts, please answer the given question. Please keep the answer as simple as possible "
          "and return all the possible answers as a list.\nHistorical facts:{facts}\nQuestion:\n {question}?")
PROMPT_V2 = ("Historical facts (each line: subject, relation, object, date):\n{facts}\n\n"
             "Question: {question}?\n\n"
             "Answer using only the historical facts above.\n"
             "- Copy the answer exactly as it is written in the facts (an entity name or a date).\n"
             "- Respect the time constraint in the question (before/after a date, first/last, in a given day, month or year).\n"
             "- If the question asks for a year, answer YYYY; for a month, YYYY-MM; for a day, YYYY-MM-DD.\n"
             "- Return a Python list of answers with the most likely answer first. Never return an empty list.")
PROMPT_V3 = ("Historical facts (each line: subject, relation, object, date):\n{facts}\n\n"
             "Question: {question}?\n\n"
             "Answer the question using only the historical facts above.\n"
             "- If the question asks when, or in which year, month or day, the answer is a date: copy it from the facts "
             "(YYYY for a year, YYYY-MM for a month, YYYY-MM-DD for a day).\n"
             "- Otherwise the answer is an entity: copy its name exactly as it is written in the facts.\n"
             "- Respect the time constraint in the question (before/after a date, first/last, in a given period).\n"
             "- Return only a Python list of answers, most likely answer first. If unsure, give your best guess from "
             "the facts; never return an empty list.")
PROMPT_V3I = ("Historical facts (each line: subject, relation, object, from start date to end date; 'beginning of time' "
              "or 'end of time' means that the start or the end is not known):\n{facts}\n\n"
              "Question: {question}?\n\n"
              "Answer the question using only the historical facts above.\n"
              "- If the question asks when, or in which year, month or day, the answer is a date: copy it from the facts "
              "(YYYY for a year, YYYY-MM for a month, YYYY-MM-DD for a day).\n"
              "- Otherwise the answer is an entity: copy its name exactly as it is written in the facts.\n"
              "- Respect the time constraint in the question (before, after, during or at the same time as another fact, "
              "or a given number of days before or after it): compare the time spans of the facts.\n"
              "- Return only a Python list of answers, most likely answer first. If unsure, give your best guess from "
              "the facts; never return an empty list.")
GRANULARITY_LINE = "- This question asks for a {unit}: answer with the {unit} only, written as {fmt}.\n"
_V3_ANCHOR = "- Otherwise the answer is an entity:"
PROMPTS = {"timer4": PROMPT, "v2": PROMPT_V2, "v3": PROMPT_V3, "v3i": PROMPT_V3I, "v3g": PROMPT_V3}


def build_prompt(facts, question: str, version: str = "timer4", granularity: str | None = None) -> str:
    """granularity ('year' / 'month' / 'day' or None) is used by v3g only; v3g without it equals v3."""
    q = question.rstrip("?")
    if version == "timer4":
        return PROMPT.format(facts=str(facts), question=q)
    text = PROMPTS[version].format(facts="\n".join(f"{i + 1}. {f}" for i, f in enumerate(facts)), question=q)
    if version == "v3g" and granularity:
        from ..labeling.granularity import UNITS
        line = GRANULARITY_LINE.format(unit=granularity, fmt=UNITS[granularity][0])
        text = text.replace(_V3_ANCHOR, line + _V3_ANCHOR, 1)
    return text
