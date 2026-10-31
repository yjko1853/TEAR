# Addendum (2026-10-17): rule-based answer slots, exact-match ATC, LLM-free Hit@1

Why: the KG-oracle slot labels used in results.md vote over every fact linking the answer and a question entity,
regardless of relation; ICEWS codes many events in both directions (Make a visit / Host a visit, sign, consult), so
those labels are often reversed. They are replaced here by a rule-based classifier (then `src/rule_slots.py`, now `tear/labeling/rules.py`) that reads the
slot off the question grammar. No LLM, no API, no training.

## Rule classifier accuracy (MultiTQ)

| check | result |
|---|---|
| timestamp vs gold answer_type (all 54,584) | 99.8% (time questions detected 15,820/15,884; entity flagged as time 39) |
| subject/object, hand-judged held-out sample (67 entity questions, judged before the last fix) | 66/67 = 98.5% |
| topic entity, same sample (64 judgeable) | 60/64 = 94% before the sector-name fix (2 of the 4 errors fixed afterwards) |

The design sample (64 questions) and the held-out sample (68 questions) were judged by Claude in-session; a final
human check by the author is still required before reporting these numbers in the paper.

> Note (2026-10-31): the rule-accuracy numbers reported in the paper (Supplementary S9.1) come from the first
> author's own judgements of a design sample of 100 test questions and a blind sample of 100 (98 judgeable):
> `paper/derived/rule_check_accuracy.csv` (rewritten wording, written by `paper/derived/rule_check_accuracy.py`)
> and `paper/derived/rule_check_accuracy_ground.csv` (original wording, written by `paper/derived/derive_ground.py`).
> The labels of this addendum were a preliminary check and are not used in the paper.

## Answer coverage@15 (fixed all-mpnet-base-v2 retriever, shared top-1000 pool, extended date parser)

| reranker | all | equal | before_after | first_last | equal_multi | after_first | before_last | Multiple | entity |
|---|---|---|---|---|---|---|---|---|---|
| none | 0.6393 | 0.6714 | 0.7435 | 0.9377 | 0.3932 | 0.3256 | 0.3062 | 0.3317 | 0.5040 |
| TimeR4-style before rerank | 0.6698 | 0.6714 | 0.7579 | 0.9377 | 0.3935 | 0.3256 | 0.5467 | 0.4273 | 0.5469 |
| hard window | 0.6661 | 0.6714 | 0.7854 | 0.9377 | 0.3966 | 0.4076 | 0.3823 | 0.3953 | 0.5418 |
| ATC only (exact) | 0.6492 | 0.6611 | 0.7771 | 0.9374 | 0.4160 | 0.3439 | 0.3317 | 0.3538 | 0.5180 |
| TR only | 0.7127 | 0.6714 | 0.7435 | 0.9782 | 0.4390 | 0.6109 | 0.5700 | 0.5596 | 0.6007 |
| TEAR (exact) | 0.7242 | 0.6611 | 0.7772 | 0.9832 | 0.4549 | 0.6341 | 0.5992 | 0.5837 | 0.6169 |
| TEAR + window (exact) | 0.7329 | 0.6611 | 0.8131 | 0.9832 | 0.4553 | 0.6411 | 0.6038 | 0.5884 | 0.6291 |

ATC topic matching: the original TEAR code penalises when the topic string is a substring of the slot value (so topic
"Iraq" also penalises "Citizen (Iraq)"); `--atc_match exact` penalises only the identical entity. TEAR coverage@15:
substring 0.7164, exact 0.7242. Files: results/multitq_*_rule_anydate.csv, results/multitq_*_rule_exact_anydate.csv.

## LLM-free Hit@1 (answer = rule-slot value of the top-1 fact; results/multitq_llmfree_hit1_rule_exact_anydate.csv)

| reranker | exact | lenient | after_first | before_after | before_last | equal | equal_multi | first_last | Multiple | entity | time |
|---|---|---|---|---|---|---|---|---|---|---|---|
| none | 0.2854 | 0.2881 | 0.0555 | 0.1808 | 0.0503 | 0.3530 | 0.0564 | 0.6317 | 0.0536 | 0.0990 | 0.7395 |
| TimeR4-style | 0.3019 | 0.3048 | 0.0555 | 0.2030 | 0.1554 | 0.3530 | 0.0568 | 0.6317 | 0.0955 | 0.1223 | 0.7395 |
| hard window | 0.3097 | 0.3124 | 0.0914 | 0.2652 | 0.0767 | 0.3530 | 0.0577 | 0.6317 | 0.0787 | 0.1333 | 0.7395 |
| ATC only | 0.3503 | 0.3539 | 0.0986 | 0.3182 | 0.0844 | 0.4298 | 0.1082 | 0.6362 | 0.0949 | 0.1906 | 0.7395 |
| TR only | 0.3427 | 0.3453 | 0.1505 | 0.1809 | 0.1402 | 0.3530 | 0.0480 | 0.8226 | 0.1255 | 0.1301 | 0.8608 |
| TEAR | 0.4437 | 0.4471 | 0.3388 | 0.3183 | 0.3190 | 0.4298 | 0.1119 | 0.8378 | 0.2847 | 0.2725 | 0.8608 |
| TEAR + window | 0.4716 | 0.4748 | 0.3399 | 0.4550 | 0.3197 | 0.4298 | 0.1119 | 0.8378 | 0.2854 | 0.3119 | 0.8608 |

Caveat: questions are TimeR4's released rewrites (produced by gpt-3.5-turbo in the TimeR4 paper), so the pipeline is
LLM-free at run time but reuses an LLM rewrite step from prior work.

## Next (S6, API-free reader)

`src/qwen_reader.py` (now `scripts/07_read_with_llm.py` and `tear/reader/`): MLX local reader (default mlx-community/Qwen2.5-7B-Instruct-4bit), TimeR4 reasoning prompt,
top-15 facts, stratified sample cache/multitq/sample_6000.json (1,000 per qtype, seed 13), resumable jsonl output in
results/qwen/, scoring with Hit@1 exact / lenient and TimeR4-style any-match.
