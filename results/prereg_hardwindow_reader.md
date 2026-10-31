# Pre-registration: reader with the hard window and anchor grounding (MultiTQ, ICEWS18)

Registered 2026-10-31, before any reader output of this setting existed. The result is reported in the paper
whatever it shows (decision recorded in TEAR_작업_히스토리.md, 2026-10-31). Later changes are listed as deviations
with their reason.

## Question

TEAR+window contains three parts: the window (and its grounded dates), ATC and TR. Table 3 has no reader result
for the hard window alone, so the reader gain of TEAR+window over no reranking (+0.077, reader+date, sample) cannot
be split into the share of the window with grounding and the share of ATC and TR. How much of the reader gain does
the label-free hard window with anchor grounding give, and how much do ATC and TR add on top of it?

## Setting (nothing new is designed or tuned)

- Reranker: `hardwindow` with tag `rule_v2_ground_exact_anydate`, i.e. the `Hard window` row of Table 3: candidates
  that violate the question's constraint (explicit or grounded dates) are moved behind the valid ones, the similarity
  order is kept otherwise. The rerank files already exist (cache/multitq_orig/rerank/, cache/icews18/rerank/) and are
  not regenerated.
- Reader: Qwen2.5-7B-Instruct-4bit (MLX), prompt v3, k = 15, greedy, batch 1, original questions; the same code
  (scripts/07_read_with_llm.py) and the same shuffled order (seed 2026) as the other settings.
- Data: MultiTQ 6,000-question sample (1,000 per type; cache/multitq_orig/sample_6000.json, all 6 chunks) and the
  first 2,000 questions of the ICEWS18 reader order (cache/icews18/sample_6000.json), as for the other ICEWS18
  settings.
- Lossless reuse (scripts/14_reuse_across_settings.py): where the hard window's top-15 list is identical to that of a
  setting already read, its output is copied (dry run: 2,744 of 6,000 MultiTQ and 763 of 2,000 ICEWS18 questions).

## Measures

- Exact Hit@1 of the first answer, reader and reader+date (date cut chosen by the gold answer type, as in Table 3),
  on the balanced sample and re-weighted to the test distribution of types (MultiTQ).
- Paired McNemar and bootstrap (10,000 draws, seed 0) for: hard window vs none, TEAR+window vs hard window, and
  hard window vs distance penalty.
- Per type (MultiTQ qtype and entity/time).
- Decomposition reported in the paper: reader gain of TEAR+window over none = (hard window - none) + (TEAR+window -
  hard window).

## What is written whatever the result

- The numbers fill the `Hard window` row of Table 3 (Reader columns) and the ICEWS18 table in the supplementary
  material.
- If TEAR+window - hard window is small or not significant, the text says that the window with grounding gives most
  of the reader gain and that ATC and TR contribute mainly to the top-ranked fact (LLM-free Hit@1) and coverage; the
  contribution bullets and the conclusion are adjusted to that. If it is large, the text says that ATC and TR add a
  reader gain on top of the window.

## Outcome (added 2026-10-31 07:40, after the run; nothing above was changed)

Run: logs/run_hardwindow_reader.sh, 2026-10-31 01:00-06:40 (MultiTQ 6,000, ICEWS18 2,000; 2,744 and 763 outputs
copied by lossless reuse, as in the dry run). Report: scripts/27_hardwindow_reader_report.py ->
results/hardwindow_reader.md, hardwindow_reader_{summary,significance}.csv. No deviation from the setting above.
Added after the run, outside the registered measures: list-level any-match and the Multiple mean (for Supplementary
Table S13) and the hybrid rule on ICEWS18 (manuscript/derived/derive_reader_icews18.py, hardwindow added to SETTINGS).

Reader+date Hit@1 (sample; MultiTQ re-weighted in brackets):

| | none | distance penalty | hard window | TEAR | TEAR+window |
|---|---|---|---|---|---|
| MultiTQ | 0.309 (0.400) | 0.337 (0.421) | 0.358 (0.444) | 0.346 (0.425) | 0.386 (0.464) |
| ICEWS18 | 0.328 | 0.332 | 0.399 | 0.364 | 0.427 |

- MultiTQ: hard window - none +0.049 [0.042, 0.056]; TEAR+window - hard window +0.027 [0.017, 0.038], p 5.5e-7;
  hard window - distance penalty +0.021 [0.013, 0.029]; TEAR - hard window -0.013, p 0.03.
- ICEWS18: hard window - none +0.071; TEAR+window - hard window +0.028 [0.010, 0.047], p 0.003.
- Per type (MultiTQ), TEAR+window - hard window: after_first +0.112, before_last +0.070, before_after +0.049,
  equal +0.001, equal_multi -0.018, first_last -0.050.
- LLM-free Hit@1 on the same sample: none 0.213, hard window 0.253, TEAR+window 0.440.

Reading: TEAR+window - hard window is significant but about a third of the gain (0.027 of 0.077). The text therefore
states both parts of the registered wording: the window with grounding gives about two thirds of the reader gain, ATC
and TR add a significant third on top of it, and their larger effect is on the top-ranked fact. Changed in the
manuscript: Table 3 (Hard window reader), Table 4 (Hard window reader row), Section 5.6 (new paragraph), Section 5.9
(ICEWS18), abstract, contribution bullet, limitations (readers) and conclusion; Supplementary Tables S13, S14, S18 and
their text.
