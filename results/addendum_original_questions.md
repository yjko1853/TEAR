# Addendum (2026-10-17): no-rewrite ablation — original MultiTQ questions vs TimeR4's rewrites

**Tables 1-4 use answer-slot rule v1** (`tear/labeling/rules.py` before the v2 fix from the human review;
labels in `cache/multitq/slots_rule.json` (13:54) and `cache/multitq_orig/slots_rule.json` (21:38), tag
`rule_exact_anydate`, date parser v1). The section "Rule v2" repeats them with rule v2, date parser v2 and the
query fallback (tag `rule_v2_exact_anydate`), next to the v1 columns. The conclusions are the same.

Why: every MultiTQ number so far used TimeR4's released rewrites (`questions.json` field `rewrite`, produced by
gpt-3.5-turbo in the TimeR4 paper). A reviewer can object that TEAR relies on a prior LLM rewriting step. This
addendum reruns the whole retrieval-side pipeline on the original MultiTQ questions (`question` field) and puts the
two runs side by side. No LLM, no API, no training; all encoding and NER on CPU.

## Setup

- Dataset `multitq_orig` (`scripts/01_prepare_data.py --dataset multitq_orig`): `facts.npz`, `vocab.json`,
  `sentences.txt`, `fact_emb.npy` are symlinks to `cache/multitq` (same facts, same fact embeddings);
  `questions.json` is a copy whose `rewrite` field holds the original question (TimeR4's rewrite kept as
  `rewrite_timer4`), so every stage that reads `rewrite` now reads the original wording.
- Same pipeline and settings as `addendum_rule_labels.md`: all-mpnet-base-v2 question encoding (CPU), exact FAISS
  top-1000, NER topic fallback (`--labels all`, CPU) and rule-based answer slots / topic entities (rule v1) recomputed
  from the original questions, extended date parser (`--date_granularity any`), exact-match ATC, lambda1 = lambda2 = 0.8.
- Commands: README "No-rewrite ablation". Stage times: question encoding 1,606 s (CPU, 4 threads), FAISS 57 s,
  NER 1,019 s, rules 22 s, features + rerankers + lambda sweep 106 s.

## What the rewrite changes

The rewrite changes 14,691 of 54,584 questions (26.9%), all multi-constraint types: after_first 6,247 / 6,266,
before_last 6,070 / 6,247, equal_multi 2,374 / 3,207. before_after, equal and first_last questions are verbatim
(39,893 questions). The rewrite replaces a reference event with that event's date, e.g.
"After the Danish Ministry of Defence and Security, who was the first to visit Iraq?" ->
"After 2006-01-05, who was the first to visit Iraq?". The date is resolved by TimeR4's retrieve-rewrite step, so the
rewrite adds information that is not in the question. A few rewrites also change the constraint itself
("In 2011, ..." -> "In 2011-12-23, ..."; one before_last question became "After 2007-07-30, ... before the Ethiopian
leader?").

Broken rewrites: the rewrite numbers here are from `questions.json` as released (v1). That file includes 380 empty
(341) or truncated (39) rewrites, all among the changed questions (before_last 156, after_first 135, equal_multi 89).
A separate run adds `questions_v2.json` (field `query`), which falls back to the original question for these 380.
Leaving them out (54,204 questions) does not change the comparison. TEAR + window vs none, coverage@15: rewrite
0.7378 vs 0.6437 (+9.4 pp), original 0.6545 vs 0.6113 (+4.3 pp). Hit@1: rewrite 0.4749 vs 0.2874 (+18.8 pp),
original 0.4074 vs 0.2755 (+13.2 pp). On the 14,311 validly rewritten changed questions, TEAR + window Hit@1 is
0.307 (rewrite) vs 0.051 (original).

Rule labels: identical on the 39,893 verbatim questions. On the 14,691 changed questions, the answer slot agrees
95.1% and the topic entity 94.0% between the two wordings. Topic source on the changed questions: main clause
13,933 / NER fallback 757 / none 1 (original); 13,604 / 671 / 416 (rewrite).

## Table 1. All 54,584 questions: answer coverage@15 and LLM-free Hit@1 (exact)

| reranker | cov@15 rewrite | cov@15 original | Δ (pp) | Hit@1 rewrite | Hit@1 original | Δ (pp) |
|---|---|---|---|---|---|---|
| none | 0.6393 | 0.6084 | -3.1 | 0.2854 | 0.2737 | -1.2 |
| TimeR4-style before | 0.6698 | 0.6114 | -5.8 | 0.3019 | 0.2782 | -2.4 |
| hard window | 0.6661 | 0.6169 | -4.9 | 0.3097 | 0.2908 | -1.9 |
| ATC only | 0.6492 | 0.6199 | -2.9 | 0.3503 | 0.3306 | -2.0 |
| TR only | 0.7127 | 0.6415 | -7.1 | 0.3427 | 0.3132 | -2.9 |
| TEAR | 0.7242 | 0.6445 | -8.0 | 0.4437 | 0.3772 | -6.7 |
| TEAR + window | 0.7329 | 0.6518 | -8.1 | 0.4716 | 0.4049 | -6.7 |

Gain over `none` within each question set (pp):

| reranker | cov@15 gain, rewrite | cov@15 gain, original | Hit@1 gain, rewrite | Hit@1 gain, original |
|---|---|---|---|---|
| TimeR4-style before | +3.0 | +0.3 | +1.7 | +0.5 |
| hard window | +2.7 | +0.9 | +2.4 | +1.7 |
| ATC only | +1.0 | +1.1 | +6.5 | +5.7 |
| TR only | +7.3 | +3.3 | +5.7 | +4.0 |
| TEAR | +8.5 | +3.6 | +15.8 | +10.4 |
| TEAR + window | +9.4 | +4.3 | +18.6 | +13.1 |

Paired significance on the original questions, LLM-free Hit@1 vs `none` (McNemar, paired bootstrap 10,000 draws;
`results/multitq_orig_llmfree_significance_rule_exact_anydate.csv`):

| reranker | Δ Hit@1 (pp) | 95% CI | McNemar p | none wrong & reranker right / none right & reranker wrong |
|---|---|---|---|---|
| TimeR4-style before | +0.5 | [0.3, 0.6] | 4.0e-17 | 547 / 301 |
| hard window | +1.7 | [1.6, 1.8] | 7.3e-165 | 1,050 / 115 |
| ATC only | +5.7 | [5.5, 5.9] | < 1e-300 | 3,151 / 41 |
| TR only | +4.0 | [3.8, 4.1] | < 1e-300 | 2,315 / 155 |
| TEAR | +10.4 | [10.1, 10.6] | < 1e-300 | 5,812 / 161 |
| TEAR + window | +13.1 | [12.8, 13.4] | < 1e-300 | 7,368 / 203 |

Original minus rewrite, same reranker (paired over questions;
`results/multitq_orig_vs_rewrite_significance_rule_exact_anydate.csv`): none -1.2 pp [-1.3, -1.1], TimeR4-style -2.4,
hard window -1.9, ATC -2.0, TR -2.9, TEAR -6.7 [-6.9, -6.4], TEAR + window -6.7 [-6.9, -6.4]; all McNemar p < 1e-100.

## Table 2. Questions the rewrite changed vs left verbatim

| reranker | changed (n=14,691): cov@15 rewrite / original | changed: Hit@1 rewrite / original | verbatim (n=39,893): cov@15 rewrite / original | verbatim: Hit@1 rewrite / original |
|---|---|---|---|---|
| none | 0.3312 / 0.2165 | 0.0545 / 0.0112 | 0.7528 / 0.7528 | 0.3704 / 0.3703 |
| TimeR4-style before | 0.4335 / 0.2165 | 0.0993 / 0.0112 | 0.7568 / 0.7568 | 0.3765 / 0.3765 |
| hard window | 0.3992 / 0.2165 | 0.0813 / 0.0112 | 0.7644 / 0.7644 | 0.3938 / 0.3938 |
| ATC only | 0.3538 / 0.2448 | 0.0957 / 0.0227 | 0.7580 / 0.7580 | 0.4441 / 0.4441 |
| TR only | 0.5707 / 0.3060 | 0.1323 / 0.0227 | 0.7650 / 0.7650 | 0.4202 / 0.4202 |
| TEAR | 0.5947 / 0.2988 | 0.2980 / 0.0509 | 0.7719 / 0.7719 | 0.4974 / 0.4974 |
| TEAR + window | 0.5998 / 0.2988 | 0.2988 / 0.0509 | 0.7819 / 0.7819 | 0.5353 / 0.5353 |

Noise floor: the original questions were encoded on CPU, the rewrites on MPS (max embedding difference 3.1e-7 on the
verbatim questions). On the 39,893 verbatim questions the retriever's top-1 is identical for 99.997% and the top-15
set for 99.99%. After reranking, at most 1 top-1 fact and at most 5 top-15 sets differ (per reranker), so every
difference in Tables 1-3 comes from the changed questions.

## Table 3. Per question type: original (rewrite)

answer coverage@15

| reranker | after_first (n=6,266, 100% changed) | before_last (n=6,247, 97%) | equal_multi (n=3,207, 74%) | before_after (n=11,073, 0%) | equal (n=17,311, 0%) | first_last (n=10,480, 0%) |
|---|---|---|---|---|---|---|
| none | 0.191 (0.326) | 0.194 (0.306) | 0.349 (0.393) | 0.744 (0.744) | 0.671 (0.671) | 0.938 (0.938) |
| TimeR4-style before | 0.191 (0.326) | 0.194 (0.547) | 0.349 (0.394) | 0.758 (0.758) | 0.671 (0.671) | 0.938 (0.938) |
| hard window | 0.191 (0.408) | 0.194 (0.382) | 0.349 (0.397) | 0.785 (0.785) | 0.671 (0.671) | 0.938 (0.938) |
| ATC only | 0.219 (0.344) | 0.221 (0.332) | 0.376 (0.416) | 0.777 (0.777) | 0.661 (0.661) | 0.937 (0.937) |
| TR only | 0.294 (0.611) | 0.290 (0.570) | 0.391 (0.439) | 0.744 (0.744) | 0.671 (0.671) | 0.978 (0.978) |
| TEAR | 0.285 (0.634) | 0.276 (0.599) | 0.412 (0.455) | 0.777 (0.777) | 0.661 (0.661) | 0.983 (0.983) |
| TEAR + window | 0.285 (0.641) | 0.276 (0.604) | 0.412 (0.455) | 0.813 (0.813) | 0.661 (0.661) | 0.983 (0.983) |

LLM-free Hit@1 (exact)

| reranker | after_first | before_last | equal_multi | before_after | equal | first_last |
|---|---|---|---|---|---|---|
| none | 0.008 (0.056) | 0.005 (0.050) | 0.039 (0.056) | 0.181 (0.181) | 0.353 (0.353) | 0.632 (0.632) |
| TimeR4-style before | 0.008 (0.056) | 0.005 (0.155) | 0.039 (0.057) | 0.203 (0.203) | 0.353 (0.353) | 0.632 (0.632) |
| hard window | 0.008 (0.091) | 0.005 (0.077) | 0.039 (0.058) | 0.265 (0.265) | 0.353 (0.353) | 0.632 (0.632) |
| ATC only | 0.014 (0.099) | 0.013 (0.084) | 0.078 (0.108) | 0.318 (0.318) | 0.430 (0.430) | 0.636 (0.636) |
| TR only | 0.024 (0.150) | 0.021 (0.140) | 0.025 (0.048) | 0.181 (0.181) | 0.353 (0.353) | 0.823 (0.823) |
| TEAR | 0.049 (0.339) | 0.046 (0.319) | 0.079 (0.112) | 0.318 (0.318) | 0.430 (0.430) | 0.838 (0.838) |
| TEAR + window | 0.049 (0.340) | 0.046 (0.320) | 0.079 (0.112) | 0.455 (0.455) | 0.430 (0.430) | 0.838 (0.838) |

## Table 4. Constraint recognition: original / rewrite (share of questions)

| qtype | n | cue first | cue last | cue before | cue after | target date | window | TR fired | ATC penalised any |
|---|---|---|---|---|---|---|---|---|---|
| after_first | 6,266 | 1.000 / 0.973 | 0.000 / 0.000 | 0.000 / 0.000 | 1.000 / 0.980 | **0.000 / 0.964** | **0.000 / 0.964** | 1.000 / 0.973 | 0.991 / 0.959 |
| before_last | 6,247 | 0.001 / 0.000 | 1.000 / 0.966 | 1.000 / 0.973 | 0.000 / 0.073 | **0.000 / 0.911** | **0.000 / 0.911** | 1.000 / 0.966 | 0.995 / 0.957 |
| equal_multi | 3,207 | 0.181 / 0.176 | 0.181 / 0.173 | 0.000 / 0.002 | 0.000 / 0.147 | 0.000 / 0.148 | 0.000 / 0.148 | 0.362 / 0.348 | 0.996 / 0.958 |
| before_after | 11,073 | 0.000 / 0.000 | 0.000 / 0.000 | 0.497 / 0.497 | 0.503 / 0.503 | 0.489 / 0.489 | 0.489 / 0.489 | 0.000 / 0.000 | 0.996 / 0.996 |
| equal | 17,311 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 | 0.654 / 0.654 |
| first_last | 10,480 | 0.501 / 0.501 | 0.500 / 0.500 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 | 1.000 / 1.000 | 0.062 / 0.062 |

Cue words survive in the original wording: first/last/before/after are recognised at least as often as in the
rewrites, so TR still fires. What disappears is the target date: no after_first, before_last or equal_multi question
carries a date (0% vs 96%, 91%, 15%). TR therefore keeps the globally earliest or latest member of each group instead
of the earliest or latest one inside the before/after window. The TimeR4-style and hard-window baselines need the
same date and have no effect on these types either.

Other diagnostics, original questions: share of top-15 candidates whose answer slot holds the topic entity is 29.9%
(none) -> 0.2% (ATC) -> 0.8% (TEAR); rewrite: 30.6% -> 0.1% -> 1.4%. lambda sweep (coverage@15): default
(0.8, 0.8) 0.6445, best 0.6470 at (0.6, 0.5), both rules off 0.6084; rewrite: 0.7242, best 0.7290 at (0.7, 0.6).
`results/multitq_orig_timer4_public_top15.csv` is byte-identical to `multitq_timer4_public_top15.csv`: TimeR4's
released top-15 was retrieved with the rewrites, so it stays a rewrite-based reference (0.748), not a no-rewrite
number.

## Interpretation (rule v1; v2 gives the same picture, see "Rule v2")

TEAR does not depend on the rewrite. On the original questions, TEAR + window still beats `none` by +4.3 pp
coverage@15 and +13.1 pp LLM-free Hit@1 (McNemar p < 1e-300), and ATC keeps almost all of its Hit@1 gain
(+5.7 vs +6.5 pp), because it reads only the question grammar and the topic entity. Overall, the rewrite accounts
for 5.1 pp of the coverage gain (9.4 -> 4.3) and 5.5 pp of the Hit@1 gain (18.6 -> 13.1). All of this loss is on the
26.9% of questions the rewrite changed. On after_first and before_last, TEAR's Hit@1 drops from 0.34 / 0.32 to
0.05 / 0.05, because the original questions name a reference event, not a date, so no before/after window can be
built. The verbatim 73% of the questions (before_after, equal, first_last) are rewrite-free by construction and give
identical results (Hit@1 0.370 -> 0.535 with TEAR + window). For the paper: report the original-question numbers as
the rewrite-free setting next to the rewrite numbers, and state that the multi-constraint gains need the
reference-event date. Resolving that date from the KG without an LLM is the open step.

## Rule v2 (rule v2 + date parser v2 + query fallback)

Configuration as in `results/addendum_rule_v2.md` (tag `rule_v2_exact_anydate`). Rewrite side:
`cache/multitq/questions_v2.json` (the 380 empty or truncated rewrites fall back to the original question),
`slots_rule_v2.json`, `top1000_v2_*`. Original side: `cache/multitq_orig/slots_rule_v2.json` (rule v2 on the
original questions, 1,700 slot changes vs v1, 0 topic changes), date parser v2, and the same retrieval and NER
topics as the v1 ablation (v2 does not change them). Commands: `logs/build_orig_v2.sh`. Since the 380 fallback
questions now read the original question on both sides, the rewrite changes 14,311 questions and 40,273 are
verbatim.

All questions, v1 and v2 columns:

| reranker | cov@15 rewrite v1 | cov@15 rewrite v2 | cov@15 original v1 | cov@15 original v2 | Hit@1 rewrite v1 | Hit@1 rewrite v2 | Hit@1 original v1 | Hit@1 original v2 |
|---|---|---|---|---|---|---|---|---|
| none | 0.6393 | 0.6406 | 0.6084 | 0.6084 | 0.2854 | 0.2886 | 0.2737 | 0.2762 |
| TimeR4-style before | 0.6698 | 0.6716 | 0.6114 | 0.6119 | 0.3019 | 0.3054 | 0.2782 | 0.2808 |
| hard window | 0.6661 | 0.6692 | 0.6169 | 0.6187 | 0.3097 | 0.3160 | 0.2908 | 0.2960 |
| ATC only | 0.6492 | 0.6517 | 0.6199 | 0.6210 | 0.3503 | 0.3518 | 0.3306 | 0.3314 |
| TR only | 0.7127 | 0.7145 | 0.6415 | 0.6405 | 0.3427 | 0.3468 | 0.3132 | 0.3158 |
| TEAR | 0.7242 | 0.7274 | 0.6445 | 0.6454 | 0.4437 | 0.4463 | 0.3772 | 0.3785 |
| TEAR + window | 0.7329 | 0.7375 | 0.6518 | 0.6543 | 0.4716 | 0.4791 | 0.4049 | 0.4111 |

Gain over `none` (pp), v1 / v2:

| reranker | cov@15 gain, rewrite | cov@15 gain, original | Hit@1 gain, rewrite | Hit@1 gain, original |
|---|---|---|---|---|
| TimeR4-style before | +3.0 / +3.1 | +0.3 / +0.3 | +1.7 / +1.7 | +0.5 / +0.5 |
| hard window | +2.7 / +2.9 | +0.9 / +1.0 | +2.4 / +2.7 | +1.7 / +2.0 |
| ATC only | +1.0 / +1.1 | +1.1 / +1.3 | +6.5 / +6.3 | +5.7 / +5.5 |
| TR only | +7.3 / +7.4 | +3.3 / +3.2 | +5.7 / +5.8 | +4.0 / +4.0 |
| TEAR | +8.5 / +8.7 | +3.6 / +3.7 | +15.8 / +15.8 | +10.4 / +10.2 |
| TEAR + window | +9.4 / +9.7 | +4.3 / +4.6 | +18.6 / +19.0 | +13.1 / +13.5 |

v2, original questions vs `none` (LLM-free Hit@1; `results/multitq_orig_llmfree_significance_rule_v2_exact_anydate.csv`):
TimeR4-style +0.5 pp [0.3, 0.6], hard window +2.0 [1.9, 2.1], ATC +5.5 [5.3, 5.7], TR +4.0 [3.8, 4.1],
TEAR +10.2 [10.0, 10.5], TEAR + window +13.5 [13.2, 13.8]; McNemar p < 1e-14 for all. Original minus rewrite, v2:
none -1.2 pp, TimeR4-style -2.5, hard window -2.0, ATC -2.0, TR -3.1, TEAR -6.8 [-7.0, -6.6],
TEAR + window -6.8 [-7.0, -6.6]; all p < 1e-100.

v2, changed (n=14,311) vs verbatim (n=40,273) questions:

| reranker | changed: cov@15 rewrite / original | changed: Hit@1 rewrite / original | verbatim: cov@15 rewrite / original | verbatim: Hit@1 rewrite / original |
|---|---|---|---|---|
| none | 0.3395 / 0.2168 | 0.0588 / 0.0115 | 0.7476 / 0.7476 | 0.3702 / 0.3702 |
| TimeR4-style before | 0.4445 / 0.2168 | 0.1054 / 0.0115 | 0.7523 / 0.7523 | 0.3765 / 0.3765 |
| hard window | 0.4093 / 0.2168 | 0.0877 / 0.0115 | 0.7616 / 0.7616 | 0.3972 / 0.3972 |
| ATC only | 0.3657 / 0.2486 | 0.0995 / 0.0216 | 0.7534 / 0.7534 | 0.4415 / 0.4415 |
| TR only | 0.5847 / 0.3024 | 0.1410 / 0.0229 | 0.7606 / 0.7606 | 0.4199 / 0.4199 |
| TEAR | 0.6134 / 0.3006 | 0.3103 / 0.0518 | 0.7679 / 0.7679 | 0.4946 / 0.4946 |
| TEAR + window | 0.6179 / 0.3006 | 0.3112 / 0.0518 | 0.7800 / 0.7800 | 0.5387 / 0.5387 |

v2 per question type, original (rewrite). The share of changed questions is 98% for after_first, 95% for
before_last, 71% for equal_multi and 0% for the other types.

| reranker | cov@15 after_first | before_last | equal_multi | before_after | equal | first_last |
|---|---|---|---|---|---|---|
| none | 0.191 (0.330) | 0.194 (0.310) | 0.349 (0.400) | 0.744 (0.744) | 0.671 (0.671) | 0.938 (0.938) |
| TimeR4-style before | 0.191 (0.330) | 0.194 (0.550) | 0.349 (0.401) | 0.761 (0.761) | 0.671 (0.671) | 0.938 (0.938) |
| hard window | 0.191 (0.412) | 0.194 (0.386) | 0.349 (0.404) | 0.794 (0.794) | 0.671 (0.671) | 0.938 (0.938) |
| ATC only | 0.223 (0.352) | 0.223 (0.337) | 0.381 (0.428) | 0.778 (0.778) | 0.661 (0.661) | 0.938 (0.938) |
| TR only | 0.290 (0.617) | 0.285 (0.575) | 0.390 (0.448) | 0.744 (0.744) | 0.671 (0.671) | 0.978 (0.978) |
| TEAR | 0.284 (0.644) | 0.280 (0.607) | 0.413 (0.469) | 0.778 (0.778) | 0.661 (0.661) | 0.984 (0.984) |
| TEAR + window | 0.284 (0.650) | 0.280 (0.611) | 0.413 (0.468) | 0.822 (0.822) | 0.661 (0.661) | 0.984 (0.984) |

| reranker | Hit@1 after_first | before_last | equal_multi | before_after | equal | first_last |
|---|---|---|---|---|---|---|
| none | 0.007 (0.057) | 0.006 (0.054) | 0.042 (0.063) | 0.192 (0.192) | 0.353 (0.353) | 0.632 (0.632) |
| TimeR4-style before | 0.007 (0.057) | 0.006 (0.160) | 0.042 (0.063) | 0.214 (0.214) | 0.353 (0.353) | 0.632 (0.632) |
| hard window | 0.007 (0.095) | 0.006 (0.082) | 0.042 (0.064) | 0.290 (0.290) | 0.353 (0.353) | 0.632 (0.632) |
| ATC only | 0.013 (0.099) | 0.011 (0.086) | 0.080 (0.113) | 0.323 (0.323) | 0.430 (0.430) | 0.636 (0.636) |
| TR only | 0.024 (0.156) | 0.022 (0.147) | 0.027 (0.052) | 0.192 (0.192) | 0.353 (0.353) | 0.824 (0.824) |
| TEAR | 0.049 (0.346) | 0.046 (0.323) | 0.081 (0.116) | 0.323 (0.323) | 0.430 (0.430) | 0.839 (0.839) |
| TEAR + window | 0.049 (0.347) | 0.046 (0.324) | 0.081 (0.116) | 0.483 (0.483) | 0.430 (0.430) | 0.839 (0.839) |

v2 constraint recognition, original / rewrite (share of questions):

| qtype | n | cue first | cue last | cue before | cue after | target date | window | TR fired | ATC penalised any |
|---|---|---|---|---|---|---|---|---|---|
| after_first | 6,266 | 1.000 / 0.995 | 0.000 / 0.000 | 0.000 / 0.000 | 1.000 / 1.000 | **0.000 / 0.962** | **0.000 / 0.962** | 1.000 / 0.995 | 0.991 / 0.980 |
| before_last | 6,247 | 0.001 / 0.000 | 1.000 / 0.991 | 1.000 / 0.997 | 0.000 / 0.072 | **0.000 / 0.910** | **0.000 / 0.910** | 1.000 / 0.990 | 0.994 / 0.982 |
| equal_multi | 3,207 | 0.181 / 0.180 | 0.181 / 0.180 | 0.000 / 0.002 | 0.000 / 0.147 | 0.000 / 0.148 | 0.000 / 0.148 | 0.362 / 0.359 | 0.996 / 0.985 |
| before_after | 11,073 | 0.000 / 0.000 | 0.000 / 0.000 | 0.497 / 0.497 | 0.503 / 0.503 | 0.568 / 0.568 | 0.568 / 0.568 | 0.000 / 0.000 | 0.996 / 0.996 |
| equal | 17,311 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 | 0.654 / 0.654 |
| first_last | 10,480 | 0.501 / 0.501 | 0.500 / 0.500 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 | 1.000 / 1.000 | 0.062 / 0.062 |

Other v2 diagnostics: lambda sweep on the original questions, default (0.8, 0.8) 0.6454, best 0.6478 at (0.6, 0.5).
On the 40,273 verbatim questions the two retrieval runs agree as before: top-1 99.998%, top-15 set 99.99%.

v2 does not change the conclusions. On original questions, TEAR + window still beats `none` by +4.6 pp
coverage@15 and +13.5 pp Hit@1 (rewrite: +9.7 and +19.0). On the questions the rewrite changes, Hit@1 is 0.311 with
the rewrite and 0.052 without it. The before_after date parser fix (target date 0.489 -> 0.568) only affects verbatim
questions, so it helps both sides equally.

## Files

- Cache: `cache/multitq_orig/` (`questions.json`, `q_emb.npy`, `top1000_{ids,scores}.npy`, `ner_spans_all.json`,
  `topic_ner_all.json`, `slots_rule.json`, `rerank/*_rule_exact_anydate_top30_ids.npy`; fact files are symlinks).
- Results (v1 tag `rule_exact_anydate`, v2 tag `rule_v2_exact_anydate`):
  `results/multitq_orig_{coverage_main,ablation,lambda_sweep,constraint_recognition,slot_occupancy_top15,llmfree_hit1,llmfree_significance}_<tag>.csv`,
  `results/multitq_orig_vs_rewrite_{summary,qtype,constraints,significance}_<tag>.csv`,
  `results/multitq_orig_vs_rewrite_retrieval_agreement.csv` (v1) and `..._retrieval_agreement_rule_v2_exact_anydate.csv`.
  v2 labels: `cache/multitq_orig/slots_rule_v2.json`; v2 build: `logs/build_orig_v2.sh`, log `logs/multitq_orig_v2.log`. Logs: `logs/multitq_orig_{search,ner,rerank}.log`.
- Code: `tear/data/__init__.py` (`is_multitq`), `tear/data/multitq.py` (`make_original_variant`),
  `tear/data/candidates.py`, `scripts/01, 03, 04, 05` (MultiTQ checks accept `multitq_orig`), `scripts/06`
  (`--dataset`, default `multitq`), new `scripts/11_compare_rewrite.py`, new `tests/test_multitq_variant.py`, README.

## Verification

- `pytest`: 44 passed after the v1 ablation (the 41 existing tests plus 3 new ones for the variant builder and
  `is_multitq`); 90 passed at 22:49, after the v2 run, including tests added by separate runs.
- `11_compare_rewrite.py` recomputes per-question coverage@15 and Hit@1 from the top-30 arrays and asserts that
  they match the stage 05/06 CSVs of both sides (v1 and v2). After its v2 options were added, rerunning it with
  the defaults reproduces the 5 v1 comparison CSVs byte for byte.
- Existing behaviour unchanged: `05_rerank.py --dataset multitq` (rule labels, exact ATC, any-date, sweep) and
  `06_llmfree_eval.py` rerun into a scratch directory reproduce all 7 existing `results/multitq_*_rule_exact_anydate`
  / `multitq_timer4_public_top15.csv` files and the 9 top-30 arrays + `question_order.json` in
  `cache/multitq/rerank/` byte for byte.
- Do-not-modify files: modification times and sizes of all 98 protected files (`cache/multitq/**`, `results/qwen/*`,
  `results/multitq_qwen_*.csv`, `tear/__init__.py`, `tear/paths.py`, `tear/utils/io.py`, `tear/text.py`,
  `tear/reader/*`, `tear/evaluation/*`, `scripts/07_read_with_llm.py`, `scripts/08_score_reader.py`) were recorded at
  20:50 before any work and compared after it. All are unchanged except
  `results/qwen/tear_window_rule_exact_anydate__Qwen2.5-7B-Instruct-4bit__k15__p-v3.jsonl`, which the running Qwen
  reader (pid 43414) appends to itself; no process from this ablation opened that file. Final check at 22:05:
  the same result. The only other change under `cache/multitq` is new `*_v2*` files (questions_v2.json,
  q_emb_v2.npy, top1000_v2_*, slots_rule_v2.json, rerank/*_rule_v2_*) written by the parallel questions_v2 session;
  none of the 98 existing files changed. Check after the v2 rerun (22:49): additionally `scripts/07_read_with_llm.py`,
  `tear/reader/mlx_reader.py` and `tear/evaluation/hits.py` have mtime 22:29. That is when the reader patches
  (`patches/*_stage_query.patch`) were applied after the v1 reader run was stopped (its tear_window chunk had
  finished); the v2 reader run `logs/run_qwen_chunks_v2.sh` started at 22:30. No process from this ablation wrote
  these files. No MPS/GPU was used; the v2 rerun used 2 CPU threads.
