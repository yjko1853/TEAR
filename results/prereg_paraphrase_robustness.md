# Pre-registration: paraphrase robustness of anchor grounding (MultiTQ)

Registered 2026-10-20, before any paraphrase was generated or scored. The user decided that the result is reported in
the paper whatever it is (2026-10-20). Nothing below may be changed after the first paraphrase is scored; any later
change is reported as a deviation with its reason.

## Question

Anchor grounding (AG) finds event-anchored constraints with patterns that match MultiTQ's wording ("After X, ...",
"... before X did?", "in the same month as X"). Does AG still find and resolve the constraint when the same question
is worded differently, and how much of its end-to-end gain survives?

## Population and sample

- MultiTQ test questions of the three multi-constraint types (after_first, before_last, equal_multi) whose original
  text contains no date (YYYY, YYYY-MM, YYYY-MM-DD) and no month name. This definition does not use the anchor
  patterns, so questions the patterns miss on the original wording stay in the population.
  Population size: after_first 6,266, before_last 6,247, equal_multi 2,046.
- Sample: 200 questions per type, 600 in all, drawn with random.Random(2026) from the population sorted by quid.

## Paraphrases

- Generator: local Qwen2.5-7B-Instruct-4bit (mlx-community), greedy decoding, max 80 new tokens, one paraphrase per
  question, prompt fixed in tear/data/paraphrase.py (PROMPT) before generation. The prompt asks for the same question
  in different words, with every name kept exactly, the time relation expressed with different wording where
  possible, and no information added or removed.
- The prompt may be smoke-tested on at most 5 dev-split questions for output format only. Test questions are not
  used to choose or change the prompt.
- Automatic validity filter, fixed in advance: one line ending with "?"; differs from the original after lower-casing
  and whitespace normalisation; every KG entity name that the rule mentioner finds in the original is also found in
  the paraphrase. Primary results use the valid paraphrases; results on all 600 are reported as well.
- Meaning check: 50 random valid paraphrases (random.Random(7)) are judged for "same question and same time
  constraint" yes / no, and the judgements are saved with the sample. The share is reported.

## Pipeline

Unchanged code and settings of the main setting (original questions + grounding, tag rule_v2_ground_exact_anydate):
question encoding and exact top-1000 search (scripts/03), NER topic fallback and rule-v2 labels (scripts/04),
grounding (scripts/14, tau 0.5), reranking with --date_granularity any --date_parser v2 --atc_match exact --anchors
--equal_window, lambda1 = lambda2 = 0.8 (scripts/05), LLM-free evaluation (scripts/06). The paraphrases are a new
dataset variant cache/multitq_para that shares the fact files with cache/multitq_orig. No rule, pattern, threshold or
lambda is tuned on the paraphrases.

## Measures (on the same questions, original wording vs paraphrase)

Primary
1. Anchor detection rate: share of questions where an anchor phrase is found.
2. Resolution rate: share where the reference event is resolved.
3. Date agreement: share of questions resolved on both wordings whose reference dates are equal.
4. LLM-free Hit@1 and coverage@15 of TEAR+window with grounding, and its gain over no reranking (none), with paired
   bootstrap (10,000, seed 2026) and McNemar, original vs paraphrase.

Secondary
5. Rule-v2 answer-slot agreement between the two wordings.
6. The same measures per question type.

## Reporting

- All measures are reported, whatever their values, in the paper (AG results, robustness paragraph) and in
  results/paraphrase_robustness.md.
- If the patterns are extended after seeing the results, the extension is a post-hoc row, evaluated on a second,
  disjoint sample of 600 drawn with seed 2027, and it does not replace the pre-registered numbers.

## Amendment 1 (2026-10-20, before any test paraphrase was generated)

The format smoke test on 5 dev questions (scripts/21 --smoke 5 --smoke_seed 1) gave well-formed output, but 2 of the 5
paraphrases changed what is asked ("Who ... on the same day as X?" became "On what day did X ...?"; "Who was the last
to visit France before X?" became a question about the order of X's visits). A paraphrase that asks another question
cannot test grounding, so the prompt was revised once, on dev questions only, by adding two rules: keep the same
question word and kind of answer, and keep the time relation itself (before / after / same period, first / last).
A second smoke test on 5 other dev questions (--smoke_seed 2) checks the revised prompt; after it the prompt is frozen
whatever that test shows. No test question was paraphrased or scored before this amendment.

## Amendment 2 (2026-10-20, after reading the first 50 generated paraphrases, before any paraphrase was scored)

Reading the first 50 test paraphrases (no grounding, ranking or answer was computed) showed that the generator often
reverses the time relation: "Which country visited Malaysia first, after Tony Blair?" became "... before Tony Blair
did?", and "first ... after" became "prior to" in several others. Such a paraphrase asks another question, so a drop
in accuracy would not measure grounding. The validity filter therefore gets one more automatic condition: the set of
time-relation classes found in the paraphrase must equal the set found in the original (after / before / first /
last / same day / same month / same year; lexicon in tear/data/paraphrase.py CONSTRAINT_CLASSES). The lexicon is
deliberately broader than the anchor patterns ("subsequent to", "later than", "once", "earlier than", "preceding",
"ahead of", "most recent", "latest", "next", ...), so paraphrases that the patterns would miss still pass and still
test grounding. The lexicon was written once, before the filter was run on the sample, and is not changed afterwards.

Also added, as a further subset and not a replacement: every paraphrase is judged for "same question, same time
constraint" by a second LLM (Claude, in a separate agent that sees only the original and the paraphrase, no results).
The 50-question manual meaning check remains and also measures agreement with that judge. Reported subsets: valid
(primary, with Amendment 2), all 600, judged same meaning, and manually checked same meaning.

## Deviation (2026-10-20, after the results)

The 50-pair manual meaning check was not carried out (author decision). The LLM judge's labels are therefore reported
without a human validation, and only as a secondary subset; the primary results use the automatic filter, which does
not depend on any judgement. The prepared sheet stays in 검수/바꿔쓰기_의미검수_50.xlsx, unfilled.
