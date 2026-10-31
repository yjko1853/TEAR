# Addendum (2026-10-18): anchor grounding — original questions, no TimeR4 rewrite, no LLM

Why: TEAR's multi-constraint gains so far depended on TimeR4's gpt-3.5 rewrite, which replaces a reference event with
its date ("After the Danish Ministry of Defence and Security, ..." -> "After 2006-01-05, ...";
`addendum_original_questions.md`). Here TEAR resolves that reference event itself, by a KG lookup, and runs on the
original MultiTQ questions. No training, no LLM, no API; CPU only (2 threads).

Setting for every run: fixed all-mpnet-base-v2 retriever and exact top-1000 pool of the run's own question text,
answer-slot rule v2, date parser v2, exact-entity ATC, lambda1 = lambda2 = 0.8, MultiTQ test (54,584 questions).

## Method (tear/labeling/anchor.py, scripts/14_ground_anchors.py)

1. **Anchor phrase.** Leading "Before/After X,", trailing "before/after X [did]?", or "in/on the same day/month/year
   as/of X". A phrase that starts with a date or month name is left to the date parser.
2. **Anchor entity.** The anchor plays the answer's role in the same kind of event with the same partner (the topic
   entity). The candidates are the entities that share a fact with the topic in the question's slot layout. Take the
   KG entities named in the phrase; otherwise take the most similar candidate name (all-mpnet cosine >= 0.5). The
   fallback catches descriptions such as "the Malaysian police" -> Police (Malaysia).
3. **Reference event.** Among the facts linking the anchor and the topic in that layout, the one with the highest
   retriever score against the question fixes the anchor and the relation.
4. **Date.** If that (anchor, relation, topic) event happened on several days, keep the widest window: earliest day
   for "after" and same-period questions, latest for "before".
5. **Use.**
   - Before/after: the anchor date drives the window, TR's valid candidates and TimeR4-style phi, exactly as an
     explicit date would. It is used only when the question has no explicit date.
   - Same period (`--equal_window`, new): "same day/month/year as X" restricts the window and TR's valid candidates to
     that period. The same window applies to explicit same-period dates ("in the same month of 2015-09-03" in the
     rewrites), so the rewrite runs get it too.
   - Anchor entity: it is named by the question like the topic entity, so it joins the topic in the ATC penalty.

**Design on dev only.** The design choices were fixed on the MultiTQ dev split: 600 questions per multi-constraint
type, seed 13, rule-v2 labels; `scripts/14_ground_anchors.py --dev_design`, `results/anchor_dev_design.csv`. They are
the name threshold, relation by fact score, the date rule, no topic guard, and the anchor-entity penalty. Criterion:
the resolved constraint yields a gold answer in the KG. Share of anchored dev questions:

| setting | after_first | before_last | equal_multi (same period) |
|---|---|---|---|
| **final: tau 0.5, widest window, no topic guard** | **0.824** | **0.807** | **0.902** |
| no name-embedding fallback (tau off) | 0.722 | 0.695 | 0.795 |
| tau 0.7 / 0.6 / 0.4 | 0.814 / 0.821 / 0.824 | 0.805 / 0.807 / 0.809 | 0.893 / 0.902 / 0.902 |
| date: earliest / latest / best-scored fact | 0.824 / 0.823 / 0.828 | 0.804 / 0.807 / 0.802 | 0.902 / 0.905 / 0.899 |
| topic guard (no fallback when the phrase names only the topic) | 0.821 | 0.802 | 0.896 |

- In the prototype, picking the relation by relation-name similarity instead of fact score lowered after_first from
  0.81 to 0.73.
- The anchor entity is a gold answer for only 4 of 1,524 resolved dev questions, so penalising it like the topic is
  safe.
- The test split was used only for evaluation.

## 1. Anchor resolution and date accuracy (test; `results/anchor_grounding_accuracy.csv`)

The reference is the date TimeR4's rewrite adds. The 380 empty or truncated rewrites are excluded, and so is any
rewrite without a full date. "Needed" = the granularity the question asks for (day for before/after).

| qtype | anchor | anchored | resolved | compared | agree day | agree month | agree year | agree at needed granularity | differ | differ: ours gives a gold answer | differ: TimeR4 gives a gold answer |
|---|---|---|---|---|---|---|---|---|---|---|---|
| after_first | after | 6,244 | 0.990 | 6,020 | 0.625 | 0.642 | 0.690 | 0.625 | 2,260 | 0.524 | 0.119 |
| before_last | before | 6,232 | 0.991 | 5,801 | 0.634 | 0.649 | 0.699 | 0.634 | 2,121 | 0.573 | 0.116 |
| equal_multi | same day | 446 | 0.998 | 353 | 0.731 | 0.737 | 0.776 | 0.731 | 95 | 0.674 | 0.032 |
| equal_multi | same month | 988 | 0.997 | 610 | 0.757 | 0.774 | 0.815 | 0.774 | 138 | 0.732 | 0.362 |
| equal_multi | same year | 336 | 0.997 | 232 | 0.668 | 0.685 | 0.737 | 0.737 | 61 | 0.738 | 0.410 |
| before_after | after / before | 2,287 / 2,329 | 0.976 / 0.976 | - | - | - | - | - | - | - | - |

- 18,863 test questions carry an event anchor and 98.8% are resolved.
- 4,616 of them are before_after questions that TimeR4 never rewrote ("Who visited China before Thailand did?"). So
  far no reranker could use a date for them.
- "Gives a gold answer" applies our relation and slot layout to each date and checks whether the first / last /
  same-period partner in the KG is a gold answer. It is a consistency check, not ground truth: TimeR4's date is
  judged under our relation.
- TimeR4's dates can be wrong too. Some rewrites point at another event between the same entities (other relation
  or reverse direction), some flip the constraint ("Which country hosted Chen Zhili's visit last, before Thailand?"
  -> "After 2005-06-12, ..."), and some contain leftover model output ("Fact: ... Modified Question: ...").

**Manual check of 20 random date disagreements** (seed 20261017). The correct anchor event was taken from the
question's own verb and the KG facts between the anchor and the topic:

| # | quid | qtype | question (shortened) | ours | TimeR4 | verdict |
|---|---|---|---|---|---|---|
| 1 | 3027814 | after_first | After Seyoum Mesfin, who first received the visit from China? | 2010-07-20 (Seyoum Mesfin hosts China) | 2006-11-02 (Seyoum visits China, reverse) | ours |
| 2 | 3002799 | equal_multi | ... optimistic comments about ASEAN in the same year as Malaysia? | 2015 (Praise, wrong relation) | 2012 (no Malaysia-ASEAN fact) | both wrong (true: 2005-03-17) |
| 3 | 3042899 | before_last | Before the Japanese Self-Defence Forces, ... last to receive humanitarian aid from Iraq? | 2005-12-05 (our slot label is wrong) | 2011-06-23 (no fact) | both wrong (true: 2008-04-18) |
| 4 | 3017771 | after_first | After the UNHCR, with which country did Iraq first express an interest in cooperation? | OHCHR (wrong entity) | 2011-12-17 (High Commission for Refugees) | TimeR4 |
| 5 | 3011929 | after_first | After Thailand, with whom did Wen Jiabao first express the intention to cooperate? | 2007-01-14 (intent to cooperate) | 2005-07-08 (negotiation, other relation) | ours |
| 6 | 3047252 | equal_multi | Which country accused Cambodia in the same month of the Military Personnel of Thailand? | 2008-10 (only Accuse fact) | 2011-04-22 (no fact) | ours |
| 7 | 3051267 | after_first | After France's Foreign Minister, which country did Malaysia first express an intention to meet with? | France (wrong entity and relation) | 2006-07-18 (Foreign Affairs (France)) | TimeR4 |
| 8 | 3051867 | before_last | Which country hosted Chen Zhili's visit last, before Thailand? | 2011-01-09 (only date) | "After 2005-06-12" (no fact, cue flipped) | ours |
| 9 | 3051074 | equal_multi | Who condemned Thailand in the same year of international governmental organisations? | 2005 (Make statement, wrong relation) | 2009-08-19 (Criticize) | TimeR4 |
| 10 | 3054315 | before_last | ... Malaysia last seek diplomatic cooperation before Barack Obama? | 2009-01-22 (Malaysia -> Obama) | 2013-05-14 (Obama -> Malaysia, reverse) | ours |
| 11 | 3054151 | before_last | Which country did China last blame before the citizens of the International? | 2015-02-06 (only date) | 2008-02-26 (no fact) | ours |
| 12 | 3032766 | before_last | Before the Chinese businessman, which country did China last want to negotiate with? | 2007-02-08 | 2005-08-31 (no fact) | ours |
| 13 | 3017277 | before_last | Before the head of government of South Korea, ... want to negotiate with Cambodia last? | 2012-03-30 | 2009-05-20 (no fact) | ours |
| 14 | 3046707 | after_first | ... first to speak negatively about China after the citizens of Singapore? | 2005-02-01 | 2015-02-06 (no fact) | ours |
| 15 | 3050219 | after_first | After the Malaysian police, ... first to want to cooperate with Thailand? | 2012-02-29 (Police (Malaysia), name fallback) | 2010-05-19 (no fact) | ours |
| 16 | 3039035 | before_last | Who was the last country to visit the UAE before Thailand? | 2010-02-05 (visit) | 2015-03-02 (Consult, other relation) | ours |
| 17 | 3053811 | equal_multi | ... negotiate with Malaysia in the same month of Tsai Ming-hsien? | 2008-05 (only fact) | 2011-11-11 (no fact) | ours |
| 18 | 3054945 | before_last | Which country made the last request to Kuwait before Oman? | 2008-06-05 (visit, wrong relation) | "After 2014-07-11" (Consult, cue flipped) | both wrong (true: 2010-06-01) |
| 19 | 3008268 | equal_multi | With whom did China telephone in the same year as Roh Moo Hyun? | 2006-10-15 (telephone) | 2006-04-17 (other relation, same year) | both (same year) |
| 20 | 3017417 | after_first | After China, ... South Korean business community wanted to establish diplomatic cooperation? | topic labelled South Korea (should be Business (South Korea)) | 2013-11-29 (no fact) | both wrong |

Ours is right in 13 of 20 and TimeR4 in 4 (#19 counts for both); in 4 cases both are wrong. Our errors come from the
relation choice (#2, #9, #18), the anchor entity (#4 acronym, #7 "France's Foreign Minister") and rule-label errors
(#3 slot, #20 topic).

## 2. Main comparison (LLM-free; `results/anchor_grounding_summary.csv`)

Runs:
- **original**: original questions, no grounding (`rule_v2_exact_anydate` on multitq_orig).
- **original+grounding**: anchors + equal window + anchor penalty (`rule_v2_ground_exact_anydate`).
- **rewrite**: TimeR4 rewrite v2, `questions_v2.json` (`rule_v2_exact_anydate` on multitq).
- **rewrite+equal_window**: the same-period window for the rewrites' explicit dates, the fair counterpart of
  grounding (`rule_v2_eqwin_exact_anydate`).

Supplementary runs:
- **original+anchors**: no equal window.
- **rewrite+grounding**: our anchors wherever the rewrite left an event anchor (mostly before_after).

Answer coverage@15:

| reranker | original | original+grounding | rewrite | rewrite+equal_window | original+anchors | rewrite+grounding |
|---|---|---|---|---|---|---|
| none | 0.6084 | 0.6084 | 0.6406 | 0.6406 | 0.6084 | 0.6406 |
| TimeR4-style before | 0.6119 | 0.6560 | 0.6716 | 0.6716 | 0.6560 | 0.6816 |
| hard window | 0.6187 | 0.6702 | 0.6692 | 0.6730 | 0.6564 | 0.6977 |
| ATC only | 0.6210 | 0.6445 | 0.6517 | 0.6517 | 0.6445 | 0.6644 |
| TR only | 0.6405 | 0.6834 | 0.7145 | 0.7145 | 0.6834 | 0.7170 |
| TEAR | 0.6454 | 0.7255 | 0.7274 | 0.7274 | 0.7255 | 0.7424 |
| TEAR + window | 0.6543 | **0.7561** | 0.7375 | 0.7408 | 0.7454 | 0.7722 |

LLM-free Hit@1 (exact):

| reranker | original | original+grounding | rewrite | rewrite+equal_window | original+anchors | rewrite+grounding |
|---|---|---|---|---|---|---|
| none | 0.2762 | 0.2762 | 0.2886 | 0.2886 | 0.2762 | 0.2886 |
| TimeR4-style before | 0.2808 | 0.3017 | 0.3054 | 0.3054 | 0.3017 | 0.3130 |
| hard window | 0.2960 | 0.3189 | 0.3160 | 0.3187 | 0.3188 | 0.3346 |
| ATC only | 0.3314 | 0.3665 | 0.3518 | 0.3518 | 0.3665 | 0.3757 |
| TR only | 0.3158 | 0.3276 | 0.3468 | 0.3468 | 0.3276 | 0.3474 |
| TEAR | 0.3785 | 0.4472 | 0.4463 | 0.4463 | 0.4472 | 0.4705 |
| TEAR + window | 0.4111 | **0.5015** | 0.4791 | 0.4831 | 0.4949 | 0.5297 |

Paired tests of LLM-free Hit@1 (McNemar; paired bootstrap, 10,000 draws;
`results/anchor_grounding_significance.csv`). "a only" / "b only" count the questions only that run answers
correctly.

| reranker | grounding vs original: Δ pp [95% CI], p | grounding vs rewrite+equal_window: Δ pp [95% CI], p (b only / a only) | grounding vs rewrite: Δ pp [95% CI], p |
|---|---|---|---|
| none | 0 (same retrieval) | -1.2 [-1.3, -1.1], < 1e-4 (753 / 74) | -1.2 [-1.3, -1.1], < 1e-4 |
| TimeR4-style before | +2.1 [2.0, 2.2], < 1e-4 | -0.4 [-0.5, -0.2], < 1e-4 (1,037 / 832) | -0.4 [-0.5, -0.2], < 1e-4 |
| hard window | +2.3 [2.2, 2.4], < 1e-4 | +0.0 [-0.1, 0.2], 0.83 (1,040 / 1,051) | +0.3 [0.1, 0.4], 3.7e-4 |
| ATC only | +3.5 [3.3, 3.7], < 1e-4 | +1.5 [1.3, 1.7], < 1e-4 (931 / 1,731) | +1.5 [1.3, 1.7], < 1e-4 |
| TR only | +1.2 [1.1, 1.3], < 1e-4 | -1.9 [-2.1, -1.8], < 1e-4 (1,437 / 391) | -1.9 [-2.1, -1.8], < 1e-4 |
| TEAR | +6.9 [6.6, 7.1], < 1e-4 | +0.1 [-0.2, 0.3], 0.49 (2,330 / 2,378) | +0.1 [-0.2, 0.3], 0.49 |
| TEAR + window | **+9.0 [8.8, 9.3], < 1e-4** | **+1.8 [1.6, 2.1], < 1e-4 (2,456 / 3,462)** | **+2.2 [2.0, 2.5], < 1e-4** |

Within original+grounding, against `none` (`results/multitq_orig_llmfree_significance_rule_v2_ground_exact_anydate.csv`):

| reranker | Δ pp vs none |
|---|---|
| TimeR4-style | +2.6 |
| hard window | +4.3 |
| ATC | +9.0 |
| TR | +5.2 |
| TEAR | +17.1 |
| TEAR + window | +22.5 [22.2, 22.9] |

All p < 1e-4.

Other comparisons:
- Equal window alone adds +0.7 pp for TEAR + window on original questions (original+anchors 0.4949 -> 0.5015)
  and +0.4 pp on rewrites.
- Grounding on top of the rewrites: +4.7 pp [4.5, 4.8] for TEAR + window (rewrite+grounding vs
  rewrite+equal_window).
- lambda sweep, original+grounding, coverage@15: default (0.8, 0.8) gives 0.7255; the best, 0.7289, is at
  (0.7, 0.5); with both rules off, 0.6084.

## 3. Per question type: original / **original+grounding** / rewrite+equal_window

Answer coverage@15:

| reranker | after_first (n=6,266) | before_last (n=6,247) | equal_multi (n=3,207) | before_after (n=11,073) | equal (n=17,311) | first_last (n=10,480) |
|---|---|---|---|---|---|---|
| none | 0.191 / **0.191** / 0.330 | 0.194 / **0.194** / 0.310 | 0.349 / **0.349** / 0.400 | 0.744 / **0.744** / 0.744 | 0.671 / **0.671** / 0.671 | 0.938 / **0.938** / 0.938 |
| TimeR4-style before | 0.191 / **0.191** / 0.330 | 0.194 / **0.525** / 0.550 | 0.349 / **0.349** / 0.401 | 0.761 / **0.791** / 0.761 | 0.671 / **0.671** / 0.671 | 0.938 / **0.938** / 0.938 |
| hard window | 0.191 / **0.292** / 0.412 | 0.194 / **0.289** / 0.386 | 0.349 / **0.582** / 0.468 | 0.794 / **0.870** / 0.794 | 0.671 / **0.671** / 0.671 | 0.938 / **0.938** / 0.938 |
| ATC only | 0.223 / **0.276** / 0.352 | 0.223 / **0.274** / 0.337 | 0.381 / **0.421** / 0.428 | 0.778 / **0.824** / 0.778 | 0.661 / **0.661** / 0.661 | 0.938 / **0.938** / 0.938 |
| TR only | 0.290 / **0.475** / 0.617 | 0.285 / **0.474** / 0.575 | 0.390 / **0.390** / 0.448 | 0.744 / **0.744** / 0.744 | 0.671 / **0.671** / 0.671 | 0.978 / **0.978** / 0.978 |
| TEAR | 0.284 / **0.578** / 0.644 | 0.280 / **0.584** / 0.607 | 0.413 / **0.453** / 0.469 | 0.778 / **0.824** / 0.778 | 0.661 / **0.661** / 0.661 | 0.984 / **0.984** / 0.984 |
| TEAR + window | 0.284 / **0.584** / 0.650 | 0.280 / **0.589** / 0.611 | 0.413 / **0.635** / 0.525 | 0.822 / **0.916** / 0.822 | 0.661 / **0.661** / 0.661 | 0.984 / **0.984** / 0.984 |

LLM-free Hit@1 (exact):

| reranker | after_first | before_last | equal_multi | before_after | equal | first_last |
|---|---|---|---|---|---|---|
| none | 0.007 / **0.007** / 0.057 | 0.006 / **0.006** / 0.054 | 0.042 / **0.042** / 0.063 | 0.192 / **0.192** / 0.192 | 0.353 / **0.353** / 0.353 | 0.632 / **0.632** / 0.632 |
| TimeR4-style before | 0.007 / **0.007** / 0.057 | 0.006 / **0.131** / 0.160 | 0.042 / **0.042** / 0.063 | 0.214 / **0.247** / 0.214 | 0.353 / **0.353** / 0.353 | 0.632 / **0.632** / 0.632 |
| hard window | 0.007 / **0.045** / 0.095 | 0.006 / **0.047** / 0.082 | 0.042 / **0.044** / 0.110 | 0.290 / **0.357** / 0.290 | 0.353 / **0.353** / 0.353 | 0.632 / **0.632** / 0.632 |
| ATC only | 0.013 / **0.066** / 0.099 | 0.011 / **0.071** / 0.086 | 0.080 / **0.135** / 0.113 | 0.323 / **0.416** / 0.323 | 0.430 / **0.430** / 0.430 | 0.636 / **0.636** / 0.636 |
| TR only | 0.024 / **0.072** / 0.156 | 0.022 / **0.076** / 0.147 | 0.027 / **0.027** / 0.052 | 0.192 / **0.192** / 0.192 | 0.353 / **0.353** / 0.353 | 0.824 / **0.824** / 0.824 |
| TEAR | 0.049 / **0.246** / 0.346 | 0.046 / **0.255** / 0.323 | 0.081 / **0.137** / 0.116 | 0.323 / **0.416** / 0.323 | 0.430 / **0.430** / 0.430 | 0.839 / **0.839** / 0.839 |
| TEAR + window | 0.049 / **0.247** / 0.347 | 0.046 / **0.255** / 0.324 | 0.081 / **0.249** / 0.184 | 0.483 / **0.651** / 0.483 | 0.430 / **0.430** / 0.430 | 0.839 / **0.839** / 0.839 |

On the 18,863 anchored questions, TEAR + window Hit@1 is 0.044 for original, **0.305** for original+grounding and
0.252 for rewrite+equal_window; coverage@15 is 0.367 / **0.662** / 0.617. On the 35,721 questions without an anchor,
the three runs give the same numbers (Hit@1 0.605).

## 4. Constraint recognition: original / **original+grounding** / rewrite+equal_window (share of questions)

| qtype | target date | window | TR fired | ATC penalised any | anchor date used | anchor penalised |
|---|---|---|---|---|---|---|
| after_first | 0.000 / **0.986** / 0.962 | 0.000 / **0.986** / 0.962 | 1.000 / **0.999** / 0.995 | 0.991 / **0.996** / 0.980 | 0.986 | 0.970 |
| before_last | 0.000 / **0.989** / 0.910 | 0.000 / **0.989** / 0.910 | 1.000 / **1.000** / 0.990 | 0.994 / **0.997** / 0.982 | 0.989 | 0.976 |
| equal_multi | 0.000 / **0.000** / 0.148 | 0.000 / **0.550** / 0.324 | 0.362 / **0.362** / 0.359 | 0.996 / **0.999** / 0.985 | 0.550 | 0.543 |
| before_after | 0.568 / **0.975** / 0.568 | 0.568 / **0.975** / 0.568 | 0.000 / **0.000** / 0.000 | 0.996 / **0.998** / 0.996 | 0.407 | 0.394 |
| equal | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0.654 / 0.654 / 0.654 | 0 | 0 |
| first_last | 0 / 0 / 0 | 0 / 0 / 0 | 1.000 / 1.000 / 1.000 | 0.062 / 0.062 / 0.062 | 0 | 0 |

"Target date" counts before/after dates only; for equal_multi the grounded window is a same-period window. The anchor
penalty matters most for same-period questions. Before it was added, the top-1 fact's answer slot held the anchor
entity in 96% of grounded same-period questions, and equal_multi Hit@1 did not move (0.081 -> 0.080).

## 5. Reader inputs on the 6,000-question Qwen sample (`results/anchor_grounding_reader_reuse.csv`)

Suppose the reader switches from the running v2 run (rewrite v2, `rule_v2_exact_anydate`) to original+grounding. A
prediction can be reused only when the question text and the top-15 fact sentences are both unchanged.

| setting | reusable | to re-run |
|---|---|---|
| none | 3,350 | 2,650 |
| tear_window | 2,742 | 3,258 |
| timer4 | 3,110 | 2,890 |
| tear | 2,777 | 3,223 |
| atc | 2,780 | 3,220 |
| tr | 3,277 | 2,723 |
| timer4_public | 3,352 | 2,648 |

The question text differs for 2,648 sample questions (the rewritten ones). For timer4_public (TimeR4's released
top-15), only the text changes. In total, 20,612 reader calls would need re-running, about 29 h at 5 s per question.

## Interpretation

With the reference event looked up in the KG, TEAR on the original questions no longer needs TimeR4's rewrite.
TEAR + window reaches LLM-free Hit@1 0.502, against 0.411 without grounding (+9.0 pp) and 0.479 / 0.483 with
TimeR4's rewrites (+2.2 / +1.8 pp, p < 1e-4). Coverage@15 is 0.756 against 0.738 / 0.741, and TEAR without the window
ties the rewrite (0.447 vs 0.446, p = 0.49).

The lead over the rewrite comes from questions TimeR4 never rewrote: 4,616 before_after questions anchored on an
event, where Hit@1 rises from 0.483 to 0.651. It also comes from same-period questions (equal_multi 0.184 -> 0.249),
where the anchor penalty keeps the anchor's own fact out of first place.

On after_first and before_last the rewrite still leads (Hit@1 0.247 / 0.255 vs 0.347 / 0.324). One reason is that
the original wording retrieves a weaker pool (no reranking: coverage@15 0.191 vs 0.330), and TEAR only reranks it.
The other is that the relation or anchor choice fails on part of the anchors. Still, where the two dates differ, ours
yields a gold answer in the KG more often (52–74% vs 3–41%), and the manual check gives ours 13 of 20 and TimeR4 4.

Grounding also adds +4.7 pp on top of the rewrites (0.530). For the paper, the original questions with grounding can
be the main, fully LLM-free setting, with the rewrite as a reference.

## Files and commands

- Code:
  - `tear/labeling/anchor.py` (new)
  - `tear/temporal.py`: `same_period`, `explicit_same_period`, `period_bounds`
  - `tear/rerank/tr.py`: `period_validity`
  - `tear/rerank/features.py`: `anchors`, `equal_window` and the anchor penalty, all off by default
  - `scripts/05_rerank.py`: `--anchors`, `--equal_window`
  - `scripts/14_ground_anchors.py` and `scripts/15_anchor_report.py` (new)
  - `tear/analysis/runs.py` (new; per-question run metrics, now also used by `scripts/11`)
  - `tests/test_anchor.py` (17 tests)
  - README section "Anchor grounding"
- Data:
  - `cache/multitq_orig/anchors.json`, `cache/multitq_orig/entity_emb.npy`
  - rewrite-side arrays in `cache/multitq_orig/rerank_on_rewrite/`; `cache/multitq/` is untouched
- Results:
  - `results/anchor_dev_design.csv`, `results/multitq_orig_anchor_resolution.csv`
  - `results/anchor_grounding_{accuracy,summary,significance,reader_reuse}.csv`
  - `results/multitq_orig_*_rule_v2_{ground,anchor}_exact_anydate.csv`
  - `results/multitq_*_rule_v2_{eqwin,ground}_exact_anydate.csv`
- Build: `scripts/14_ground_anchors.py --dev_design`, `scripts/14_ground_anchors.py --dataset multitq_orig`,
  `bash logs/build_anchor.sh`, `scripts/15_anchor_report.py`. Logs: `logs/anchor_build.log`, `logs/anchor_report.log`.

## Verification

- `pytest`: 107 passed. That includes the 17 new tests: anchor phrases on original questions, dates left to the date
  parser, the date rule, same-period parsing and bounds, the KG lookup on a toy KG, and `build_features` with
  anchors, the equal window and the anchor penalty.
- Defaults unchanged: `05_rerank.py` without `--anchors` / `--equal_window` reproduces, byte for byte, 12 CSVs and 20
  top-30 arrays. They are the multitq v1 `rule_exact_anydate` outputs and the multitq_orig v2 `rule_v2_exact_anydate`
  outputs.
- `15_anchor_report.py` asserts that its per-question recomputation matches the stage 05/06 CSVs of all six runs.
- `scripts/11_compare_rewrite.py` now uses `tear/analysis/runs.py`. Rerunning it reproduces its 10 v1 and v2
  comparison CSVs byte for byte.
- Protected files (reader, evaluation, `text.py`, `paths.py`, `utils/io.py`, scripts 07/08, `cache/multitq/`,
  `results/qwen*/`, `results/multitq_qwen_*.csv`; 124 files) were recorded at 22:56 and checked after the runs. The
  only change is `results/qwen_v2/timer4_rule_v2_exact_anydate__...jsonl`, which the running v2 reader appends to.
  No new files were written under `cache/multitq/`, `tear/reader/` or `tear/evaluation/`. No GPU was used.
