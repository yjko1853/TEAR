# Addendum (2026-10-18): a second benchmark, MultiTQ-style questions over ICEWS18

Why: TEAR's effects have been measured on MultiTQ (ICEWS05-15) only. No other public benchmark has point-in-time event
facts, the same (s, r, o) repeated over days, first/last and dated before/after cues, and subject/object answers.
ICEWS18 (RE-NET release, 2018-01-01..10-31) has those facts and does not
overlap ICEWS05-15 in time, but has no questions. This addendum builds them with MultiTQ's own procedure, checks the
procedure by regenerating ICEWS05-15 questions and comparing them with the MultiTQ test split, and prepares the GPU
runs. CPU only (2 threads); no encoding, search or reader run was made.

Result in one paragraph. The generator reproduces MultiTQ's answer procedure exactly for 79% of the dev questions (97%
of `equal` entity questions); the 22 relations, templates, date formats, surface forms and type-internal mix come from
MultiTQ train and dev questions and were frozen before any ICEWS18 question existed. Regenerated ICEWS05-15 questions
match the MultiTQ test split in every question-form statistic TEAR's rules depend on (first/last and before/after cues,
dated before/after questions and the share TEAR's date parser reads, event anchors, answer counts, answer types,
date formats) and in TEAR's answer-slot accuracy per type. The remaining differences come from MultiTQ's entity-fold
split (25% of the test questions are about China) and are reported, not reproduced. ICEWS18 gives 6,000 test and
1,200 dev questions with the same distributions, except that ICEWS18 spans one calendar year, so it has no year-level
questions.

## 1. MultiTQ's procedure, recovered from its questions

MultiTQ (Chen et al., ACL 2023, Sec. 3.2) generated questions from 246 core templates over the 22 most frequent
relations, expanded to 7,334 templates by question type and time granularity, each with "a procedure executed over the
TKG to extract all possible answers"; questions with more than 20 answers and pseudo-temporal questions were removed.
The generation code is not public (`czy1999/MultiTQ` holds the model code and the data). The procedures below were
recovered by aligning MultiTQ dev questions to the KG and keeping the variant that reproduces the gold answer set
exactly (exploration on 1,000 dev questions per type, before any template was frozen):

| qtype | shape | answers | evidence on dev |
|---|---|---|---|
| equal | Who visited {O} in {T}? | every partner with a fact in period T, on the whole KG | 988 / 1,000 reproduced |
| equal (time) | When did {S} visit {O}? | the one period of (S, r, O); triples spanning two periods are not asked | 1,131 of 1,143 reproduced only with a single period |
| before_after | Who visited {O} before {T} / before {Y}? | partners with a fact strictly before / after T, or before / after the anchor's earliest day | strict 516 vs inclusive 293; anchor earliest 349 vs latest 4 |
| first_last | Who first visited {O}? / When did {S} first visit {O}? | partners on the first / last day; the first / last period | 692 / 698 |
| equal_multi | Who visited {O} in the same month as {Y}? / In {T}, who first visited {O}? | partners in the anchor's period minus the anchor / first-last day within year T | same-period without the anchor 511 vs with it 9 |
| after_first, before_last | After {Y}, who first visited {O}? / ... last before {Y} did? | the first day after / last day before the anchor's **earliest** day | earliest 708 / 740 vs latest 112 / 104 |

The anchor event is (Y, r, O) with the anchor in the answer's role (object-side questions: (S, r, Y)). These
procedures are `tear/data/qgen/semantics.py`; the aligner and the generator call the same functions.

## 2. Templates: alignment and freeze

**Alignment** (`tear/data/qgen/align.py`). For each MultiTQ train or dev question with gold answers, every structure
(answer side, relation, topic mention, anchor mention, date expression, cue direction) whose procedure reproduces the
gold answer set exactly is a hypothesis. Hypotheses come from the facts that link the answers to the entities
mentioned in the question (aliases, Sec. 2.2), so only a few are evaluated per question. When several structures
reproduce the answers, which happens for reciprocal relations (Make a visit / Host a visit), symmetric ones (Engage in
negotiation, Sign formal agreement) and swapped topic and anchor, a bigram model of delexicalised frames
(placeholders are tokens) plus a naive-Bayes relation model, both trained on the unambiguous questions, choose one if
it is ahead by a log-odds margin of log 20; otherwise the question is left out. TEAR's answer-slot rules are not used,
so the generator's answer slot is independent of the classifier it later checks. The chosen structure is
delexicalised (spans -> `{S}`, `{O}`, `{Y}`, `{T}`) and identical frames are merged with counts per split.

[Table 1] Questions whose gold answers the frozen procedure reproduces exactly (unique or model-resolved structure)

| qtype | answer | train | dev | test (aligned after the freeze, comparison only) |
|---|---|---|---|---|
| equal | entity | 0.884 | 0.968 | 0.960 |
| equal | time | 0.691 | 0.741 | 0.734 |
| before_after | entity | 0.741 | 0.820 | 0.814 |
| first_last | entity | 0.736 | 0.858 | 0.850 |
| first_last | time | 0.720 | 0.771 | 0.764 |
| equal_multi | entity | 0.758 | 0.796 | 0.799 |
| after_first | entity | 0.634 | 0.623 | 0.588 |
| before_last | entity | 0.657 | 0.641 | 0.598 |
| all | | 0.736 (284,791 / 386,787) | 0.789 (45,751 / 57,979) | 0.775 (42,312 / 54,584) |

Unaligned questions are mostly mentions the alias matcher does not read ('the Ministry of Foreign Affairs of Vietnam'
for Foreign Affairs (Vietnam)) and MultiTQ questions whose wording dropped the anchor ('In the same month, which
country made negative comments about Japan?').

**Relations.** Dev questions aligned without a relation restriction show 22 relations with >= 0.2% of the aligned
questions and a sharp drop after the 22nd (Discuss by telephone 361, fight with small arms and light weapons 208,
then Make statement 55, Consult 13); this matches the paper's "22 most frequent relations"
(`results/qgen_relation_selection.csv`). All 22 exist in ICEWS18. Their most frequent wordings are in
`results/qgen_relation_phrases.csv` (e.g. Make optimistic comment: 'made optimistic remarks about {O}'; Use
conventional military force: 'made {O} suffer from conventional military forces').

**Surface forms and dates.** Sector actors 'Head (Country)' are written in eight forms (of, of the, possessive,
country + head, in, from, adjective, verbatim) with number, case and article; their frequencies are measured on the
aligned dev mentions. Adjectives are mined from train questions ('the Kenyan ministry' -> Kenya; 79 countries, the only
irregular one kept is British). Date expressions are kept as shapes ('{D} {Month} {YYYY}', '{Mon}, {YYYY}',
'the {Dth} of {Month} {YYYY}', ...) with the template, so MultiTQ's format mix is reproduced.

**Which split weights the templates.** MultiTQ's dev and test questions share their wording statistics (mean length
10.00 / 10.18 words for equal, 12.96 / 13.02 for after_first; paper Table 6: 11.28 / 11.41 tokens) while train
questions are longer (12.60 for equal) and differently mixed (first_last entity questions 3.8% of train, 1.2% of dev,
1.1% of test). Generation therefore weights a template by the number of dev questions it reproduced (14,454 usable
dev templates) and takes the type-internal bucket mix (answer type, time level, variant) from all dev questions. Train
questions contribute the demonyms, the alignment models and the article rates.

**Freeze record** (`resources/qgen/FREEZE_LOG.md`). Templates: `resources/qgen/multitq_templates_v1.json.gz`, 46,923
templates (46,902 usable), SHA-256 `a886f3b884a7622f72f26959dc6e55044ee028f6cb9ade70019fb5fe64d47878`, created
2026-10-18 18:19 from train (sha256 89ce10b4...) and dev (ad0dfe7c...) only. Before the freeze the test split's
question lengths and type proportions (not its answers) were compared once with train and dev to choose the weighting
split above; the test questions were aligned to the KG for the first time at 18:22 (`align-test`). A first build at 18:09 was superseded before any question was generated from it because
three mined adjectives were wrong (South Korea -> 'Grand', United States -> 'Mahdi', Africa -> 'Constitutional'); it is
kept under `resources/qgen/superseded/` with its hash. `load_frozen` checks the SHA-256 before every generation.

## 3. Generator

For a type, a bucket is drawn with MultiTQ dev's shares, then a template of the bucket by its dev count, then a seed
fact uniformly among the KG facts of the template's relation. The seed fixes the topic (its non-answer side), the anchor
(its answer side, anchored variants) and the period (its day / month / year, written in the template's date shape).
Answers come from Sec. 1. A draw is rejected for no answer, more than 20 answers, an `equal` entity question whose time
constraint does not change the answer (MultiTQ's pseudo-temporal filter), an `equal` time question whose triple spans
several periods, a repeated question text, and the following regularities of MultiTQ dev (generator v2,
`resources/qgen/generator_v2.json`, SHA-256 `a7f13805...`, fixed at 22:19 before any retrieval):

- **one answer** for after_first, before_last, first_last (entity) and equal_multi within a year: all 6,499 / 6,532 /
  698 / 1,063 dev questions of these kinds have exactly one answer;
- **anchor events seen on one day**: 98.7% of the 13,737 aligned anchored dev questions (also removes the conflict
  between MultiTQ's earliest-day rule and TEAR's widest-window rule);
- **sector vs other entities per role**: dev never names a sector entity as the topic of an entity question without an
  anchor (18,801 / 18,801) and never two sector entities in one question; with an anchor the topic / anchor pattern is
  other-other 0.719, other-sector 0.162, sector-other 0.119; time questions S / O 0.721, 0.088, 0.191. Each question
  draws a pattern and keeps only seed facts that match it;
- **seed weighting of time questions**: a triple is drawn with probability ~ (its fact count)^0.5, the exponent chosen
  on ICEWS05-15 so that first/last time questions fall in one period as often as on dev (dev 0.568; 0.534 at 0.5,
  0.704 at 0, 0.290 at 1).

The first generator version (18:21, without these rules) and the first v2 config (22:14, without role patterns, with
two surface-form errors: 'of the' for every country, 'affairses') produced sets that were used only for the fidelity
and rule statistics that led to the changes; no retrieval or reading was run on them (log entries in
`FREEZE_LOG.md`).

**Degenerate time level.** ICEWS18 covers one calendar year: a year-level time answer would always be '2018', a
year-level before/after constraint would have no answer, and 'in 2018' or 'in the same year as' would not change the
answer. Year buckets are therefore left out on ICEWS18 and their weight moves to the same answer type and variant at
day and month (so the share of time answers and of dated before/after questions stays MultiTQ's). equal_multi's
'first/last within a year' variant has no other level, so ICEWS18 equal_multi questions are all same-period questions.

**Sets** (`.meta.json` next to each file records the seed, config, rejections and SHA-256):

| set | file | questions | seed | quids |
|---|---|---|---|---|
| ICEWS18 test | `cache/icews18/questions.json` | 6,000 (1,000 per type) | 2026 | 5,000,000- |
| ICEWS18 dev | `cache/icews18/questions_dev.json` | 1,200 (200 per type), no text shared with test (94 share a seed fact) | 2027 | 5,100,000- |
| ICEWS05-15 regenerated | `cache/icews0515_regen/questions.json` | 6,000 (1,000 per type) | 2025 | 7,000,000- |

The question records have the MultiTQ format of `scripts/01` (quid, question, rewrite = question, answers,
answer_type, qtype, qlabel, time_level, empty TimeR4 top-15) plus `gen`: answer slot, topic entity or subject /
object, anchor entity and date, date expression (text, shape, from / to), first/before flag, gold fact ids, seed fact,
template id and frame, surface forms. `cache/icews18/{facts.npz, vocab.json, sentences.txt}` follow the
`scripts/01` layout (468,558 facts = train + valid + test, 23,033 entities, 256 relations, 304 days). `is_multitq`
now accepts `icews18` and `icews0515_regen`, so stages 02-08 run on them unchanged (checked with a synthetic candidate
pool: stages 05 and 06 run through and write their tables).

## 4. Fidelity: regenerated ICEWS05-15 vs MultiTQ test

Generated sets have 1,000 questions per type; "reweighted" rows average the per-type values with MultiTQ test's type
proportions. Text-side values use every question; KG-side values for MultiTQ test use its 42,312 aligned questions.

[Table 2] Question form (MultiTQ test / regenerated ICEWS05-15 / ICEWS18 test)

| qtype | time answer | words | answers per question | single answer | first/last cue | before/after cue | date expression | before/after date read by TEAR's parser v2 | event anchor phrase |
|---|---|---|---|---|---|---|---|---|---|
| equal | 0.348 / 0.345 / 0.389 | 10.2 / 10.6 / 10.8 | 2.62 / 2.17 / 2.74 | 0.644 / 0.691 / 0.611 | 0 / 0.003 / 0 | 0 / 0 / 0 | 0.652 / 0.655 / 0.611 | – | 0 / 0 / 0 |
| before_after | 0 / 0 / 0 | 10.1 / 10.7 / 10.9 | 5.17 / 7.07 / 6.69 | 0.109 / 0.154 / 0.154 | 0 / 0.002 / 0 | 1 / 1 / 1 | 0.588 / 0.543 / 0.563 | 0.568 / 0.535 / 0.543 | 0.417 / 0.455 / 0.435 |
| first_last | 0.941 / 0.939 / 0.947 | 12.4 / 12.9 / 12.9 | 1 / 1 / 1 | 1 / 1 / 1 | 1 / 1 / 1 | 0 / 0 / 0 | 0 / 0 / 0 | – | 0 / 0 / 0 |
| equal_multi | 0 / 0 / 0 | 13.0 / 13.6 / 14.5 | 2.23 / 2.43 / 3.97 | 0.628 / 0.657 / 0.369 | 0.362 / 0.340 / 0 | 0 / 0 / 0 | 0.362 / 0.340 / 0 | – | 0.552 / 0.593 / 0.912 |
| after_first | 0 / 0 / 0 | 13.0 / 13.6 / 13.6 | 1 / 1 / 1 | 1 / 1 / 1 | 1 / 1 / 1 | 1 / 1 / 1 | 0 / 0 / 0 | – | 0.996 / 0.993 / 0.995 |
| before_last | 0 / 0 / 0 | 12.9 / 13.4 / 13.4 | 1 / 1 / 1 | 1 / 1 / 1 | 1 / 1 / 1 | 1 / 1 / 1 | 0 / 0 / 0 | – | 0.998 / 0.998 / 0.998 |
| reweighted to MultiTQ test | 0.291 / 0.290 / 0.305 | 11.4 / 11.9 / 12.0 | 2.43 / 2.69 / 2.88 | 0.685 / 0.710 / 0.668 | 0.443 / 0.443 / 0.421 | 0.432 / 0.432 / 0.432 | 0.347 / 0.338 / 0.308 | 0.115 / 0.109 / 0.110 | 0.346 / 0.355 / 0.370 |

Time levels match as well (MultiTQ test day / month / year 0.616 / 0.220 / 0.163; regenerated 0.625 / 0.213 / 0.162
reweighted), and so do the date formats (day level '{D} {Month} {YYYY}' 0.867 / 0.884, month level '{Month} {YYYY}'
0.581 / 0.650, '{YYYY}-{MM}' 0.154 / 0.146; `results/qgen_fidelity_date_shapes.csv`).

[Table 3] KG side (MultiTQ test aligned / regenerated / ICEWS18 test)

| qtype | gold fact in an (s, r, o) group seen on >= 2 days | anchor event on several days | first/last time triple in one period |
|---|---|---|---|
| equal | 0.662 / 0.614 / 0.492 | – | – |
| before_after | 0.930 / 0.849 / 0.758 | 0.001 / 0 / 0 | – |
| first_last | 0.545 / 0.540 / 0.415 | – | 0.588 / 0.552 / 0.647 |
| equal_multi | 0.838 / 0.734 / 0.591 | 0.002 / 0 / 0 | – |
| after_first | 0.730 / 0.560 / 0.396 | 0.018 / 0 / 0 | – |
| before_last | 0.727 / 0.549 / 0.410 | 0.022 / 0 / 0 | – |
| all | 0.719 / 0.641 / 0.510 | 0.013 / 0 / 0 | – |

**What is not reproduced.** MultiTQ generated each split on its own entity fold, which concentrates the test questions
on a few hubs: the ten most frequent topics carry 66.8% of the aligned test questions (China alone 10,480, 25%) against
24.4% in the regenerated set and 12.2% in ICEWS18. Hub topics have more repeated facts and shorter names: without its
top-10 topics, MultiTQ test's repeated-gold share falls from 0.719 to 0.499, and its equal questions are 11.3 words long
instead of 9.6. The remaining KG-side gap (0.719 vs 0.641) and the +0.5 words come from this; the generator does not
imitate the fold split. On ICEWS18 the repeated-gold share is lower again (0.510) because the KG itself repeats less
(56% of its facts are in (s, r, o) groups seen on >= 2 days, against 75% for ICEWS05-15; 10 months against 11 years):
TR has less to do there, which works against TEAR, not for it.

## 5. ICEWS18 test: the forms ATC and TR act on

Per type, ICEWS18 has the same cue rates as MultiTQ (Table 2). In absolute numbers the 6,000 test questions hold 3,000
first/last questions (first_last 1,000 with 94.7% time answers, after_first 1,000, before_last 1,000), 563 dated
before/after questions (543 read by TEAR's date parser v2) and 3,437 anchored questions (before/after 437, after_first
1,000, before_last 1,000, same-period 1,000). The first 2,000 questions of the reader's fixed shuffle hold 327 / 331 /
354 / 330 / 328 / 330 of equal / before_after / first_last / equal_multi / after_first / before_last (180 dated and 151
anchored before/after questions).

Question-level rule firing without anchor grounding (the parts of stage 05's constraint table that do not depend on
the candidates; MultiTQ original questions / ICEWS18): first cue after_first 1.000 / 1.000, last cue before_last
1.000 / 1.000, first or last in first_last 0.501+0.500 / 0.502+0.498, explicit before/after date 0.568 / 0.543, TR
fires in after_first, before_last and first_last 1.000 / 1.000 and in equal_multi 0.362 / 0 (the year-only first/last
variant). With grounding the date of the 3,437 anchored questions is looked up in the KG (stage R5 below).

## 6. TEAR's rules against the generator's record

The generator knows each question's answer slot, topic, date and anchor, so TEAR's CPU components can be scored
without labelling. Rule v2 without the NER fallback (stage 04 with NER runs in the GPU script). The MultiTQ reference
scores the same rule against the answer side of the aligned test questions; it includes MultiTQ's own label noise
(e.g. 'Who was the first to praise the High Commission for Refugees after ...?' whose gold answer is the one praised),
which the generator reproduces through the templates: the rule agreed with a human judgement of the slot on 97 of 98
questions in the blind check, against 97.7% with the aligned MultiTQ answer side.

[Table 4] Answer-slot agreement of rule v2 (MultiTQ test aligned / regenerated / ICEWS18 test / ICEWS18 dev)

| qtype | slot | topic entity |
|---|---|---|
| equal | 0.998 / 0.997 / 0.995 / 1.000 | 0.993 / 0.996 / 0.975 / 0.990 |
| before_after | 0.972 / 0.979 / 0.970 / 0.970 | 0.976 / 0.985 / 0.982 / 0.970 |
| first_last | 0.992 / 0.988 / 0.988 / 0.985 | 0.983 / 0.981 / 0.950 / 0.935 |
| equal_multi | 0.965 / 0.974 / 0.986 / 0.980 | 0.965 / 0.956 / 0.941 / 0.955 |
| after_first | 0.919 / 0.915 / 0.914 / 0.915 | 0.962 / 0.965 / 0.956 / 0.945 |
| before_last | 0.936 / 0.929 / 0.931 / 0.935 | 0.966 / 0.968 / 0.970 / 0.955 |
| all (natural mix / balanced) | 0.977 / 0.964 / 0.964 / 0.964 | 0.980 / 0.975 / 0.962 / 0.958 |

ICEWS18 test reweighted to MultiTQ's type mix: slot 0.971. By gold slot (ICEWS18 test): subject 0.988 (3,044),
object 0.899 (1,620; 164 read as subject), timestamp 0.989. By template source: slot 0.972 on templates whose structure
the KG fixed (3,891 questions) and 0.950 on templates whose structure the frame model chose (2,109), so the model
decisions do not inflate the agreement. Other components on ICEWS18 test: first/last cue detected in 3,000 / 3,000;
before/after date parsed and equal to the generator's boundary day in 543 / 563 (0.964; the misses are shapes the
parser does not read: '2018-10-8', 'Aug, the year of 2018', 'the 14th of March 2018'); anchor phrase found for 0.971 of the anchored
questions with the right type (before / after / day / month). A mention-only anchor lookup (no retriever scores, so the
relation is the first candidate's) finds the right anchor entity for 0.826 and the right date for 0.494; the full
grounding with fact scores and the name-embedding fallback needs the encoder and is stage R5/R8 below.

## 7. Remaining runs (GPU; after the MultiTQ reader run)

`logs/run_icews18.sh` holds every remaining stage; each GPU step first checks that no `07_read_with_llm.py` /
`run_qwen_chunks*` process is running (tested while the MultiTQ run was active: exit 2, nothing started) and skips
outputs that exist.

```bash
caffeinate -i bash logs/run_icews18.sh retrieval   # R1 encode 468,558 facts (MPS, ~30 min); R2 search test + dev;
                                                   # R3 NER topic fallback; R4 rule v2 labels; R5 anchor grounding
                                                   # (dev, test); R6 rerank with grounding + equal window (main) and
                                                   # without; R7 LLM-free Hit@1 + significance; R8 rulecheck with the
                                                   # final labels and anchors
caffeinate -i bash logs/run_icews18.sh difficulty  # D1-D4 the same on the regenerated ICEWS05-15 set (no fact encoding:
                                                   # MultiTQ embeddings); D5 scripts/17 difficulty: coverage@15 and
                                                   # LLM-free Hit@1 per type, MultiTQ (multitq_orig) vs regenerated vs
                                                   # ICEWS18 -> results/qgen_difficulty.csv
caffeinate -i bash logs/run_icews18.sh reader      # Qwen2.5-7B-4bit, prompt v3, k = 15, first 2,000 of the fixed
                                                   # shuffle, settings none / timer4 / tear / tear_window with
                                                   # cross-setting reuse (scripts/14_reuse_across_settings.py);
                                                   # scored by scripts/08 -> results/qwen_icews18_scores/
```

Tag of the main run: `rule_v2_ground_exact_anydate` (rule v2 + NER fallback, date parser v2, exact ATC, anchor
grounding, equal window), as for MultiTQ's original questions. The reader sample `cache/icews18/sample_6000.json` was
made with the reader's own `make_sample` (seed 13). The `icews18_timer4_public_top15.csv` table that stage 05 writes
is empty by construction (no TimeR4 output exists for generated questions). Expected time: ~1 h for retrieval and
difficulty, 2,000 x 4 reader calls minus reuse (~55% on MultiTQ) at ~4.5 s, about 4-5 h.

## 8. Caveats

- The questions are generated, with MultiTQ's procedure and wording; they inherit its noise (grammar slips, voice
  flips, as in Table 4's reference) and carry no human check. The ICEWS18 set supports controlled comparisons (same
  retriever and reader, rerankers swapped), not leaderboard numbers.
- ICEWS18 has no year-level questions and no 'first/last within a year' variant (Sec. 3); MultiTQ test has 16.3%
  year-level questions.
- The fidelity comparison was iterated: the first generator's statistics led to the v2 rules. Every rule was read off
  MultiTQ dev, and generation and calibration never used test answers, but the test comparison is not a blind check of
  the first design.
- KG-side statistics of MultiTQ test cover the 77.5% of questions the procedure reproduces; the remaining questions
  may differ (e.g. anchors not written by name).
- ICEWS data are licensed for research and education only; RE-NET's repository has no license file.

## 9. Files

| file | content |
|---|---|
| `tear/data/qgen/` | `kg.py` (event index, RE-NET loader), `timeexpr.py`, `aliases.py`, `semantics.py`, `align.py`, `freeze.py`, `generate.py` |
| `tear/analysis/qgen_fidelity.py` | fidelity statistics, rule check, MultiTQ reference |
| `scripts/17_generate_questions.py` | extract, calibrate, prepare, generate, align-test, fidelity, rulecheck, difficulty |
| `resources/qgen/` | frozen templates, generator v2 config, SHA-256 files, `FREEZE_LOG.md`, superseded builds |
| `results/qgen_alignment_{traindev,test}.csv` | alignment status per split / type |
| `results/qgen_relation_selection.csv`, `qgen_relation_phrases.csv` | the 22 relations, their wordings |
| `results/qgen_fidelity_{text,kg,date_shapes,topics}.csv`, `qgen_kg_summary.csv` | Tables 2-3 and more |
| `results/{icews18,icews18_dev,icews0515_regen}_qgen_rulecheck_*.csv`, `multitq_test_qgen_rulecheck_reference.csv` | Table 4 and details |
| `cache/icews18/`, `cache/icews0515_regen/`, `cache/qgen/` | KG files, question sets (+ `.meta.json`), `slots_rule_v2_noner*.json`, reader sample, alignment rows (git-ignored) |
| `external/ICEWS18_RE-Net/`, `external/MultiTQ_hf/` | downloaded data (RE-NET `data/ICEWS18`; MultiTQ train / dev from Hugging Face `chenziyang/MultiTQ`; dev identical to TimeR4's copy) |
| `logs/run_icews18.sh` | remaining GPU stages |
| `tests/test_qgen.py` | 25 unit tests (synthetic KG) |
