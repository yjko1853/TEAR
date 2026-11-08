# Pre-registration: TR-promote (within-group reorder) variant of the representative-fact rule

Registered 2026-10-31 09:50, before any reranked list or reader output of this variant existed. The result is
reported in the paper whatever it shows (supplementary material; the main text refers to it in the discussion of the
representative-fact rule). Later changes are listed as deviations with their reason.

## Question

On the reader, TR lowers Hit@1 of `first_last` questions (reader+date, MultiTQ sample: none 0.682 -> TR 0.629,
TEAR+window 0.632; ICEWS18: 0.726 -> 0.684) while it raises the LLM-free Hit@1 of the same questions
(0.655 -> 0.837). Hypothesis: the reader judges first/last by comparing the several dated occurrences of the same
triple in its list, and TR removes exactly those occurrences from the top-15 (every non-representative is multiplied
by lambda2 = 0.8, so other groups' facts move in front of them). If so, a variant that lifts the representative to the
head of its group but leaves the other members where the retriever put them should keep the LLM-free gain and remove
the reader loss. If the reader loss is caused by something else (for example, the representative itself being the
wrong fact), the variant will not restore the reader.

## Variant (fixed before the run; no tuning)

`tr_promote`: for a question with a first/last cue, groups and representatives are those of TR (same validity mask,
same group keys). The representative's score becomes the maximum similarity over its group's members (+1e-6 to break
the tie with that member); every other candidate keeps its score. Nothing is down-weighted. `tear_promote` = ATC +
`tr_promote`; `tear_window_promote` = ATC + `tr_promote` + the hard window (score -10 outside the window), i.e.
TEAR+window with TR replaced by the variant. lambda1 = 0.8 and the window are unchanged. Code:
`tear/rerank/tr.py: representative_of`, `tear/rerank/core.py: promote_scores`, tag `rule_v2_ground_exact_anydate`.

## Runs

- LLM-free Hit@1 (`scripts/06_llmfree_eval.py --rerankers ...`), all MultiTQ test questions (54,584) and all 6,000
  ICEWS18 questions, for none / tr / tear / tear_window and the three variants; paired McNemar and bootstrap of each
  variant against its TR counterpart (tr_promote vs tr, tear_promote vs tear, tear_window_promote vs tear_window).
  Outputs: `results/tr_promote/`.
- Reader (Qwen2.5-7B-Instruct-4bit, prompt v3, k = 15, greedy, same code and shuffled order as the other settings):
  - MultiTQ: `tear_window_promote` on the 3,000 sample questions of the three types with a first/last cue
    (`cache/multitq_orig/sample_firstlast_cue_3000.json`: first_last, after_first, before_last; TR does not fire on
    the other types, so their lists are identical to TEAR+window); `tr_promote` on the 1,000 `first_last` questions
    (`sample_first_last_1000.json`). Predictions: `results/qwen_ground_promote/`.
  - ICEWS18: `tear_window_promote` on the first_last questions among the first 2,000 of the reader order (354;
    `cache/icews18/sample_reader2000_first_last.json`). Predictions: `results/qwen_icews18_promote/`.
  - Lossless reuse (`scripts/14_reuse_across_settings.py --source_dir`): where a variant's top-15 list equals that of a
    setting already read, the output is copied.

## Measures

- Reader Hit@1 (exact, first answer) and reader+date, per question type, paired against TEAR+window (for
  `tear_window_promote`) and against TR and none (for `tr_promote`) on the same questions: McNemar with continuity
  correction and paired bootstrap (10,000 draws, seed 0), as in `scripts/27_hardwindow_reader_report.py`.
- LLM-free Hit@1 of the same questions and of the full test set, paired the same way.
- Diagnostic: how many of the gold-answer facts' other occurrences (same group) TEAR+window keeps in the top-15,
  versus the variant.

## What is written whatever the result

- A supplementary table with the three variants next to TR / TEAR / TEAR+window (LLM-free on the full test set,
  reader on the first/last-cue sample), and one paragraph in the main text where the representative-fact rule's reader
  effect is discussed.
- If `tear_window_promote` recovers the reader loss on `first_last` (difference to TEAR+window positive with the
  interval above 0) while its LLM-free Hit@1 stays within 0.01 of TEAR+window on these types, the paper says that the
  reader loss comes from removing the other occurrences and recommends the promote form of the rule for a reader
  pipeline, keeping the down-weighting form for direct (LLM-free) answering. If the LLM-free Hit@1 falls by more than
  0.01, the trade-off is reported as such and both forms are kept as options. If the reader loss is not recovered,
  the paper says the loss is not explained by the removal of the other occurrences and leaves it as an open limitation.
- The variant is not used anywhere else in the paper; no main-table number changes.


## Outcome (written 2026-10-31 18:00, after the runs; nothing above was changed)

- LLM-free, full sets (`results/tr_promote/tr_promote_llmfree_pairs.csv`): MultiTQ TEAR+window 0.5015 -> promote 0.5347
  (+0.033 [0.031, 0.035]; after_first 0.247 -> 0.385, before_last 0.255 -> 0.395, first_last 0.839 -> 0.845; equal,
  before_after unchanged); TR 0.328 -> TR-promote 0.414; ICEWS18 TEAR+window 0.472 -> 0.523. Coverage@15 on first_last
  0.984 -> 0.958, higher on after_first / before_last.
- Reader, MultiTQ first/last-cue sample (3,000; `tr_promote_reader_significance.csv`, reader+date): first_last
  TEAR+window 0.632 -> promote 0.714 (+0.082 [0.052, 0.112], p < 1e-4), above none / hard window (0.682; +0.032 [0.013,
  0.051], p = 0.002); before_last 0.194 -> 0.161 (-0.033 [-0.059, -0.006], p = 0.019); after_first 0.209 -> 0.206
  (p = 0.88); the three types together 0.345 -> 0.360 (+0.015 [-0.001, 0.031], p = 0.066). LLM-free on the same
  questions 0.443 -> 0.535.
- Reader, TR alone on the 1,000 first_last sample questions: reader+date TR -> TR-promote 0.629 -> 0.709 (+0.080 [0.050, 0.110], p < 1e-4; gains 161, losses 81); none -> TR-promote 0.682 -> 0.709 (+0.027 [0.008, 0.046], p = 0.008; gains 61, losses 34); none -> TR 0.682 -> 0.629 (-0.053 [-0.083, -0.023], p = 8.1e-04; gains 94, losses 147). LLM-free on the same questions: none 0.655, TR 0.837, TR-promote 0.843. Distinct groups in the top-15: none 6.9, TR 14.7, TR-promote 6.7.
- Reader, ICEWS18 first_last questions among the first 2,000 of the reader order (354): reader+date TEAR+window -> promote 0.684 -> 0.749 (+0.065 [0.020, 0.110], p = 0.008; gains 46, losses 23); none -> promote 0.726 -> 0.749 (+0.023 [-0.011, 0.056], p = 0.243; gains 22, losses 14); none -> TEAR+window 0.726 -> 0.684 (-0.042 [-0.085, 0.000], p = 0.068; gains 22, losses 37); hard window = none on these questions (no date). LLM-free: none 0.655, TEAR+window 0.825, promote 0.836. Distinct groups: none 8.6, TEAR+window 14.9, promote 8.4. Reader speed on ICEWS18 lists: 22.5 s/question (longer fact sentences), 321 read + 33 reused.
- Diagnostic (`tr_promote_group_diagnostic.csv`, `tr_promote_reader_diagnostic.csv`): other occurrences of the top-1
  fact's group in the top-15, first_last test set: none 1.71, TEAR+window 0.27, promote 1.91. Distinct groups in the
  top-15 on the reader sample: TEAR+window 13.6, promote 5.0 (after_first 13.1 -> 4.3, before_last 13.2 -> 4.2,
  first_last 14.6 -> 6.6). The reader follows the top-ranked fact on 18% of before_last questions under promote.
- Decision rule: the first_last condition holds (interval above 0) and LLM-free Hit@1 does not fall (it rises), so the
  paper says that the reader loss of TR on first_last comes from the removal of the other occurrences and names the
  promote form as the one to use before a reader on single-answer first/last questions. The rule did not anticipate the
  before_last loss, which is reported next to it with the diagnostic (the occurrences crowd out other candidate entities
  on the Multiple types); a type-dependent choice between the two forms would be a second post-hoc step and is not made.
  Written up as registered: supplementary Table S16 and paragraph "A within-group variant of the representative-fact
  rule" in the reader section, one sentence in Section 5.6 of the main text. No main-table number changed.
- Deviations: none in the runs. Added after the runs started, before any reader output was read: the report also pairs
  each comparison setting against `none` on the same questions (for the table's Δ column) and a reader diagnostic
  (follows-top-1 rate, distinct groups, gold in top-15); both are descriptive and change no registered measure.
