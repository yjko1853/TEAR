# Addendum (2026-10-18): anchor grounding on TimelineKGQA (ICEWS-Actor, CronQuestions-KG)

Why: the first TimelineKGQA runs (`results.md`, TimelineKGQA sections) found almost no TEAR effect. The questions
contain no first/last cue and no "before/after <date>", so TR and the date windows had nothing to act on.
TimelineKGQA medium / complex questions do, however, fix the time by another event ("Who Affiliation To Far Left
while Takis Kliridis Affiliation To Ministry of Finance?"). The anchor grounding that worked on MultiTQ
(`addendum_anchor_grounding.md`) might therefore apply. Here it is extended to interval facts and Allen relations.
No training, no LLM; CPU only (2 threads). Writes went only to `cache/icews`, `cache/cron`, `results/` and new code
files.

Update 2026-10-19: §6 repeats the runs with rule labels instead of the dataset's answer types; §7 prepares the reader
experiment (not run).

Setting:
- Test splits as before: ICEWS-Actor 14,621 questions (`split` column), CronQuestions-KG 8,344.
- The same top-1000 pools (`cache/<ds>/top1000_*`, read only), all-mpnet-base-v2.
- Answer slot from the dataset's `answer_type` (`slots_oracle.json`; gold information, see §6 for rule labels),
  exact-entity ATC, lambda1 = lambda2 = 0.8.
- Baseline rerankers are recomputed in the same script with exact ATC, so they differ slightly from the
  substring-ATC numbers in `results.md`.

## 1. Where event anchors occur (test; `results/<ds>_grounding_patterns.csv`)

| dataset | level | question type | answer types | share with an event anchor | most frequent connectives |
|---|---|---|---|---|---|
| ICEWS | medium | timeline_recovery + temporal_constrained_retrieval | subject 1,179 / object 1,137 | 100% | during 673; same_time(+finishes) 590; N days {relation} 342 |
| ICEWS | complex | timeline_position_retrieval*2 + temporal_constrained_retrieval | subject 661 / object 669 | 100% | during+during 197; pairs with equal 141; finishes+finishes 31 |
| ICEWS | medium / complex | timeline_recovery*2, timeline_position*3 | union/intersection 3,011, duration 1,458, ranking 512 | 0–48% | same_time (union / intersection: both events are the answer) |
| ICEWS | simple | timeline_recovery, temporal_constrained_retrieval | 5,994 | 0% (timestamp_range 46% "during") | – |
| Cron | medium | timeline_position + temporal_constrained_retrieval | subject 764 / object 795 | 100% | N days {relation} 739; before 374; after 186; {relation} 201 |
| Cron | complex | timeline_position*2 + temporal_constrained_retrieval | subject 758 / object 805 | 100% | after+after 247; before+before 229; N days before / after (x2) 454 |
| Cron | other | ranking, duration, union/intersection, simple | 5,222 | 0–53% | – |

Relations of the asked event X to the anchor Y in the medium/complex subject/object questions (`temporal_relation`
field):

| dataset | relations (count) |
|---|---|
| ICEWS | d 712, di 715, = 696, f 414, fi 402, > 344, < 306, N days before 191, X > Y with offset 173, X < Y with offset 169, during (no offset) 162, N days after 157, si 134, s 110 |
| Cron | > 1,193, < 1,094, N days after 719, N days before 710, X > Y with offset 371, X < Y with offset 370, the rest < 50 each |

Grounding targets the 3,646 (ICEWS) and 3,122 (Cron) medium/complex questions with a subject or object answer. The
other types do not fit:
- **ranking** answers a rank number;
- **union / intersection / range** answer an interval made of both events;
- **duration** answers a length;
- **simple** questions name their own event.

## 2. Method (tear/labeling/interval_anchor.py, scripts/16_timeline_grounding.py)

1. **Connective -> Allen relation set.** The wording only fixes a family of relations, so each connective maps to a
   set. Mappings fixed on the validation split:
   - before / prior to / ahead of / preceding -> {<}
   - after -> {>}
   - during / while / in the course of / the period when / in the midst of -> {d, di, o, oi}
   - at the same time -> any overlap {=, s, si, f, fi, d, di, o, oi}
   - at the same start and end time / equal -> {=}
   - finishes / finishedby / starts / startedby -> {f} / {fi} / {s} / {si}
2. **Day offsets.** "N days after Y" means X.start - Y.end = N; "N days before Y" means Y.start - X.end = N. In some
   questions the generator printed its relation into the text ("{'relation': 'X < Y', ...}"); that relation is read
   as written.
3. **Anchor event.** The anchor is the KG fact whose subject and object are both named in the clause that follows
   the connective (Cron: the named predicate is preferred). If that (s, p, o) has several intervals, a candidate
   counts as valid when it stands in the relation to *any* of them.
4. **Window.** A candidate must satisfy every constraint (complex questions have two). The window is used by the
   hard window and TEAR + window, exactly as a date window would be. Grounding applies only when the asked slot is a
   subject or object.
5. **No anchor penalty here.** Unlike MultiTQ, the anchor event is an unrelated event. On validation the answer is
   one of its entities in 3.5% (ICEWS) / 12.4% (Cron) of questions, so the anchor-entity ATC penalty is off (see
   §5).

**Validation check** (`--dev_design`, `results/<ds>_grounding_dev.csv`; medium/complex subject/object questions with
gold events):

| | ICEWS (4,626) | Cron (3,172) |
|---|---|---|
| a constraint is parsed | 100% | 100% |
| gold anchor (s, p, o) found | 97.7% | 91.5% |
| connective family contains the gold Allen relation | 100% | 99.7% |
| gold labels agree with the strict Allen definitions | 100% | 99.2% (year-level point intervals) |
| day offsets agree with the gold events (written out / printed relation) | 100% (610 / 328) | 100% (1,521 / 744) |
| gold answer event passes the window: any interval / best-scored interval only | **96.9%** / 88.1% | **87.7%** / 76.9% |

The Cron column was recomputed after event parsing was made deterministic. `tear/data/timelinekgqa.parse_events`
iterated a set of predicate names, so under some hash seeds 27–36 Cron validation rows failed to parse; it now tries
the longest predicate first. The validation set went from 3,150 to 3,172 questions; every share changed by at most
0.1 pp, and ICEWS is unchanged. Successful parses are identical in every order, and the test questions were not
affected.

## 3. Test results (`results/<ds>_grounding_{coverage,llmfree,significance,event_recall,constraints}.csv`)

"Grounded" = questions with a resolved anchor window: ICEWS 3,646 and Cron 3,110 of 3,122 medium/complex
subject/object questions. Answer coverage@15 is defined for subject / object / timestamp answers (ICEWS 7,671
questions, Cron 4,739). LLM-free Hit@1 reads the asked slot of the top-1 fact (subject, object, start or end date)
and counts an exact match.

**Answer coverage@15**

| reranker | ICEWS all | ICEWS grounded | ICEWS medium | ICEWS complex | Cron all | Cron grounded | Cron medium | Cron complex |
|---|---|---|---|---|---|---|---|---|
| none | 0.576 | 0.306 | 0.185 | 0.517 | 0.536 | 0.391 | 0.276 | 0.505 |
| hard window | 0.607 | 0.306 | 0.185 | 0.517 | 0.570 | 0.391 | 0.276 | 0.505 |
| TEAR | 0.576 | 0.307 | 0.186 | 0.517 | 0.535 | 0.390 | 0.275 | 0.504 |
| TEAR + window | 0.608 | 0.307 | 0.186 | 0.517 | 0.568 | 0.390 | 0.275 | 0.504 |
| hard window + grounding | 0.654 | 0.404 | 0.269 | 0.639 | 0.635 | 0.490 | 0.348 | 0.631 |
| **TEAR + window + grounding** | **0.654** | **0.404** | **0.270** | **0.638** | **0.633** | **0.489** | **0.344** | **0.632** |
| TEAR + window + grounding, anchors kept | 0.656 | 0.408 | 0.278 | 0.634 | 0.645 | 0.506 | 0.384 | 0.627 |

**LLM-free Hit@1 (exact)**

| reranker | ICEWS all | ICEWS grounded | ICEWS subject | ICEWS object | Cron all | Cron grounded | Cron subject | Cron object |
|---|---|---|---|---|---|---|---|---|
| none | 0.374 | 0.163 | 0.172 | 0.225 | 0.332 | 0.228 | 0.268 | 0.216 |
| hard window | 0.421 | 0.163 | 0.237 | 0.287 | 0.380 | 0.228 | 0.317 | 0.281 |
| TEAR | 0.377 | 0.164 | 0.179 | 0.226 | 0.336 | 0.230 | 0.276 | 0.216 |
| TEAR + window | 0.423 | 0.164 | 0.240 | 0.288 | 0.384 | 0.230 | 0.327 | 0.281 |
| hard window + grounding | 0.463 | 0.251 | 0.290 | 0.345 | 0.444 | 0.325 | 0.393 | 0.357 |
| **TEAR + window + grounding** | **0.464** | **0.252** | **0.293** | **0.346** | **0.449** | **0.329** | **0.404** | **0.359** |
| TEAR + window + grounding, anchors kept | 0.443 | 0.207 | 0.273 | 0.309 | 0.449 | 0.329 | 0.397 | 0.367 |

By level, Hit@1 of TEAR + window -> + grounding: ICEWS medium 0.059 -> 0.120, complex 0.347 -> 0.483; Cron medium
0.140 -> 0.173, complex 0.318 -> 0.482. Simple questions are unchanged: they have no event anchor, and their
"from X to Y" window was already used.

**Paired tests on the grounded questions** (McNemar; paired bootstrap, 10,000 draws; all p < 1e-4):

| comparison | ICEWS Δ coverage@15 | ICEWS Δ Hit@1 | Cron Δ coverage@15 | Cron Δ Hit@1 |
|---|---|---|---|---|
| TEAR + window + grounding vs TEAR + window | +9.8 pp [8.8, 10.8] | +8.8 pp [7.8, 9.8] | +9.9 pp [8.7, 11.1] | +9.9 pp [8.7, 11.1] |
| hard window + grounding vs hard window | +9.8 pp | +8.8 pp | +9.9 pp | +9.7 pp |
| TEAR + window vs none (no grounding) | +0.1 pp (p = 0.63) | +0.1 pp (p = 0.25) | -0.1 pp (p = 0.39) | +0.2 pp (p = 0.21) |

- Over all slot-answer questions, TEAR + window + grounding vs TEAR + window: ICEWS +4.2 pp Hit@1 / +4.6 pp coverage,
  Cron +6.5 / +6.5 pp.
- Without grounding, TEAR does nothing on the grounded questions.
- TEAR and hard window give the same result once grounded (ATC adds at most +0.5 pp), because the gain comes from
  the window.

**Answer event vs anchor events in the top-15** (grounded questions with gold events, `results/<ds>_grounding_event_recall.csv`):

| reranker | ICEWS answer event | ICEWS all anchor events | Cron answer event | Cron all anchor events |
|---|---|---|---|---|
| none / TEAR + window | 0.292 | 0.495 | 0.361 | 0.211 |
| TEAR + window + grounding | **0.393** | 0.261 | **0.459** | 0.067 |
| TEAR + window + grounding, anchors kept | 0.398 | 0.577 | 0.491 | 0.310 |

TimelineKGQA's gold-event recall counts the anchor events too. The window demotes them on purpose, since they don't
stand in the relation to themselves. So gold-event recall@15 ("any") falls on medium / complex (ICEWS medium
0.893 -> 0.784, Cron medium 0.740 -> 0.626) even though the answer event rises by 10 pp.

The "anchors kept" variant adds the anchor facts back into the window. It raises both answer-event and anchor-event
recall (ICEWS gold-event "all" 0.452 -> 0.472, "any" 0.861 -> 0.884). LLM-free Hit@1 then drops on ICEWS (0.252 ->
0.207), because an anchor fact can take first place. For an LLM reader, which also needs to see the anchor, it is
probably the better input; this is not tested here.

**Window size.** The window keeps on average 54% / 53% of the 1,000 candidates for ICEWS medium subject / object,
21–25% for complex, and 25–34% for Cron. Overlap relations with open intervals (46% of ICEWS facts start at the
beginning of time and 62% end at the end of time) filter little. Day offsets and before/after are sharp.

## 4. Can TimelineKGQA be the second benchmark?

**Yes, for event-anchored temporal constraints (grounding + window); no, for ATC or TR.**

Evidence for:
- On both KGs the grounded window lifts the 3,000+ medium/complex entity questions by about +10 pp in coverage@15
  and LLM-free Hit@1. The gain is significant, comes from a component designed on another dataset, and has its
  settings fixed on TimelineKGQA's validation split.
- The connective lexicon and Allen definitions are validated against gold events.
- Without grounding, TEAR does nothing here, as the earlier runs showed. That matches the claim that TEAR's
  temporal component needs the reference time, and that the KG can supply it.

Limits:
1. ATC and TR are not exercised.
   - ICEWS-Actor has a single predicate.
   - Neither KG has first/last questions with an entity answer: ranking answers a number.
   - TEAR equals "none" and the hard window equals TEAR + window.
   - So TimelineKGQA cannot replicate MultiTQ's ATC/TR results. It replicates only the window and grounding part.
2. The text is template-generated and has generator artefacts.
   - Relation dicts printed into the question.
   - Answers leaked into some complex templates ("Who Bahrain National Gas Company ..., finishes ...?").
   - The `split` column (57,453 / 17,298 / 14,621) does not match the README.
   - These need stating, and they make it look less like a natural-language benchmark.
3. Absolute levels stay low, because the retriever rarely brings the answer event into the pool: answer-event
   recall@15 is only 0.39 / 0.46 after grounding.
4. No published baseline uses this split with retrieval-side metrics, so the comparison is internal: the same pool,
   with and without grounding.

## 5. Ablation: anchor-entity ATC (off)

With the anchor entities added to the ATC penalty, as on MultiTQ, TEAR + window + grounding is lower on the grounded
questions (`logs/<ds>_grounding_anchoratc.log`):

| | coverage@15 | Hit@1 |
|---|---|---|
| ICEWS, with the anchor penalty | 0.389 | 0.241 |
| ICEWS, without | 0.404 | 0.252 |
| Cron, with the anchor penalty | 0.458 | 0.294 |
| Cron, without | 0.489 | 0.329 |

TEAR + anchor penalty without a window loses 4.3 pp Hit@1 on Cron. This matches the validation finding that the
answer is often an anchor entity in TimelineKGQA: in this dataset the anchor is just another event.

## 6. Rule labels instead of the dataset's answer types (2026-10-19)

Sections 1–5 took the answer slot from the dataset's `answer_type` (`slots_oracle.json`, tag `oracle_ground`). That
is gold information, so the TEAR runs were repeated with the MultiTQ rule-v2 labels (`tear/labeling/rules.py`,
question grammar only; `scripts/16_timeline_grounding.py --make_rule_labels` -> `cache/<ds>/slots_rule_v2.json`;
tag `rule_ground`). The answer slot and topic entity come from the rules. Timestamp questions read the end date when
the question says end / finish / stop / conclude / terminate. On validation this word list separates the two
exactly: ICEWS 950/950 end and 1,020/1,020 start questions, Cron 418/418 and 411/411. A first list that also had
"left" misread 6 ICEWS start questions naming "Far Left".

**Rule accuracy against the dataset's answer type** (test, `results/<ds>_rule_label_accuracy.csv`):

| level / answer type | ICEWS n | ICEWS slot accuracy | Cron n | Cron slot accuracy |
|---|---|---|---|---|
| complex / object | 669 | **0.000** | 805 | **0.000** |
| complex / subject | 661 | 0.992 | 758 | 0.997 |
| medium / object | 1,137 | 0.182 | 795 | 0.348 |
| medium / subject | 1,179 | 0.881 | 764 | 0.904 |
| simple / object | 1,021 | 0.504 | 403 | 0.531 |
| simple / subject | 1,051 | 1.000 | 433 | 1.000 |
| simple / timestamp start / end (incl. start vs end) | 1,953 | 1.000 | 781 | 1.000 |
| all slot questions | 7,671 | 0.707 | 4,739 | 0.665 |
| topic entity agrees with the oracle topic | | 0.522 | | 0.693 |

The MultiTQ grammar rules do not fit TimelineKGQA's object questions. TimelineKGQA writes them as
"<subject> <relation> which organisation, ...?" or "Which organisation is Affiliation Toed <subject> ...?", forms
that do not occur in MultiTQ, so the rules fall back to "subject". The rules were left as they are (no
TimelineKGQA-specific rule was added).

**Oracle vs rule labels** (same pools and anchors; grounded = 3,646 / 3,110 questions):

| reranker | ICEWS cov@15 grounded, oracle / rule | ICEWS Hit@1 grounded, oracle / rule | Cron cov@15 grounded, oracle / rule | Cron Hit@1 grounded, oracle / rule |
|---|---|---|---|---|
| none | 0.306 / 0.306 | 0.163 / 0.116 | 0.391 / 0.391 | 0.228 / 0.151 |
| hard window | 0.306 / 0.306 | 0.163 / 0.116 | 0.391 / 0.391 | 0.228 / 0.151 |
| TEAR | 0.307 / 0.276 | 0.164 / 0.105 | 0.390 / 0.297 | 0.230 / 0.126 |
| TEAR + window | 0.307 / 0.276 | 0.164 / 0.105 | 0.390 / 0.297 | 0.230 / 0.126 |
| hard window + grounding | 0.404 / **0.404** | 0.251 / 0.150 | 0.490 / **0.490** | 0.325 / 0.200 |
| TEAR + window + grounding | 0.404 / 0.380 | 0.252 / 0.141 | 0.489 / 0.420 | 0.329 / 0.172 |

Over all slot questions, rule labels give TEAR + window + grounding Hit@1 0.368 (ICEWS) / 0.315 (Cron), against
0.464 / 0.449 with oracle labels.

- **Coverage of the label-free rerankers is unchanged.** none, the hard window and the hard window + grounding do
  not use labels. The grounded window keeps its +9.8 / +9.9 pp coverage@15 (hard window + grounding vs hard window,
  p < 1e-4).
- **TEAR's ATC hurts with these labels.** With a wrong slot or topic it penalises the correct facts: TEAR + window vs
  none is -3.0 pp (ICEWS) / -9.4 pp (Cron) coverage on the grounded questions. Grounding still adds +10.4 pp
  [9.4, 11.4] / +12.3 pp [11.0, 13.6] coverage and +3.6 / +4.6 pp Hit@1 to TEAR + window (p < 1e-4).
- **LLM-free Hit@1 drops sharply for object questions.** It reads the slot the labels give, so a subject read on an
  object question is always wrong. On TimelineKGQA it measures the labels as much as the ranking.
- **Rule-label runs also ground some non-entity questions.** The rules give an entity slot to many union /
  intersection questions, so the window is also applied there. Those questions have no coverage or Hit@1, so the
  tables above are unaffected. The event-recall CSV of the rule run (`*_event_recall_rule_ground.csv`), however,
  covers 5,523 / 4,112 questions instead of 3,630 / 3,110.
- **Conclusion for the paper.** On TimelineKGQA, report the grounded window (hard window + grounding, label-free)
  as the transferable component. TEAR's slot-dependent parts need labels that the MultiTQ rules cannot give for
  this question style.

## 7. Reader experiment (prepared, not run)

Does grounding carry through to the final answer of the local reader?

- **Data.** `scripts/19_timeline_reader_data.py` builds `cache/{icews,cron}_rd/`: questions in `scripts/07`'s format
  (quid, rewrite = original question, answers, qtype, qlabel = level, answer_type, answer_in_question). The fact
  sentences and the `rule_ground` top-30 arrays are symlinks to `cache/<ds>/`. No reader code is changed.
- **Sample.** `sample_1000.json`: 1,000 random targets per dataset (medium/complex, subject/object; seed 13). 07
  shuffles them with seed 2026, so every prefix is a paired subset.
  - Caveat: the answer string occurs in the question for 231 (ICEWS) / 373 (Cron) of the 1,000. That is every
    complex-subject question (661/661, 758/758, a generator artefact) and 7–26% of the medium ones.
  - Each prediction can be split by `answer_in_question`. `sample_1000_noleak.json` is the same draw from the 2,814 /
    1,950 targets without the answer in the text; use it with `SAMPLE=sample_1000_noleak.json`, and predictions go
    to `results/qwen_timeline_<ds>_noleak/`.
- **Settings.**
  - none (the retriever's order);
  - tear_window = TEAR + window + grounding (main comparison);
  - hardwindow = hard window + grounding (label-free).
  - All use rule labels (tag `rule_ground`), Qwen2.5-7B-Instruct-4bit, k = 15, batch 1, greedy.
- **Prompt.**
  - v3 describes each fact as "subject, relation, object, date". TimelineKGQA facts are "s p o from t1 to t2", with
    open ends written "beginning of time" / "end of time".
  - `patches/reader_prompt_v3i.patch` adds v3i: the same prompt with only the fact-format line and the
    time-constraint line written for time spans. It passes a dry run and is not applied, since `tear/reader/` is
    in use.
  - The driver chooses between v3 and v3i on `cache/{icews,cron}_rd_tune/` (60 validation questions each, seed 7,
    their own CPU retrieval) with the `none` reranker only. Rule, fixed before the run: v3i only if its exact Hit@1
    over the 120 questions is strictly higher, ties broken by lenient Hit@1, then v3.
  - The choice is written to `logs/timeline_prompt_choice.json` and reused on restart. Without the patch, v3 is used.
- **Driver.** `logs/run_timeline_reader.sh`.
  - It exits at once if 07, `run_icews18.sh` or `run_qwen_chunks` is running.
  - It works in chunks of 250 questions: ICEWS then Cron, settings in the order none -> tear_window -> hardwindow.
  - Before each setting, `scripts/14_reuse_across_settings.py` copies predictions with an identical top-15
    (tear_window from none; hardwindow from tear_window or none).
  - After each chunk, `scripts/08_score_reader.py --baseline none_rule_ground` writes
    `results/qwen_timeline_<ds>_scores/multitq_qwen_{hit1,significance}.csv`. It scores exact and lenient matches on
    the first answer; all answers are entities.
- **Expected time.**
  - Identical top-15 on the samples: tear_window = none 21.6% / 13.2%; hardwindow = tear_window or none 92.1% /
    78.8%.
  - Reader calls: about 1,860 (ICEWS) + 2,080 (Cron) + 240 for the prompt choice = about 4,180.
  - At 5.4 s per call (the ICEWS18 reader's measured speed) to 6.5 s, that is **about 6.5–7.5 h**.

## Files and commands

- Code:
  - `tear/labeling/interval_anchor.py` (new): connective lexicon, Allen masks, offsets, anchor lookup, window,
    LLM-free answer reading
  - `scripts/16_timeline_grounding.py` (new)
  - `tests/test_interval_anchor.py` (new, 10 tests)
  - §6–7: `scripts/16_timeline_grounding.py --make_rule_labels` and `--read_slot`; `scripts/19_timeline_reader_data.py`
    (new); `logs/run_timeline_reader.sh` (new); `patches/reader_prompt_v3i.patch` (new, not applied);
    `tear/data/timelinekgqa.py` (deterministic predicate order in `parse_events`)
- Cache: `cache/{icews,cron}/interval_anchors.json`, `rerank/{none,hardwindow,tear,tear_window}_{oracle,rule}_ground_top30_ids.npy`,
  `slots_rule_v2.json`; reader folders `cache/{icews,cron}_rd/`, `cache/{icews,cron}_rd_tune/`.
- Results:
  - `results/{icews,cron}_grounding_{dev,patterns,constraints,coverage,llmfree,significance,event_recall}.csv`
  - rule labels: `results/{icews,cron}_rule_label_accuracy.csv`, `results/{icews,cron}_grounding_*_rule_ground.csv`
  - logs `logs/{icews,cron}_grounding.log`, `logs/{icews,cron}_grounding_anchoratc.log`, `logs/{icews,cron}_grounding_rule.log`
- Commands (CPU, `OMP_NUM_THREADS=2`):
  ```
  .venv/bin/python -I scripts/16_timeline_grounding.py --dataset icews --dev_design
  .venv/bin/python -I scripts/16_timeline_grounding.py --dataset icews
  .venv/bin/python -I scripts/16_timeline_grounding.py --dataset cron --dev_design
  .venv/bin/python -I scripts/16_timeline_grounding.py --dataset cron
  ```

## Verification

- `pytest`: 118 passed after §1–5 and 143 passed after §6–7 (separate runs also added tests), including the 10
  new tests. They cover connective parsing on validation questions, simple
  questions without constraints, exclusive and exhaustive Allen masks (with the point-interval case), the
  any-interval window, offsets, and answer reading.
- No existing code file was edited for §1–5. `build_features`, `coverage_rows`, `rerank` and the significance tests
  are used as they are. For §6–7 only `tear/data/timelinekgqa.parse_events` changed (deterministic predicate order).
- The oracle run reproduces byte for byte with the extended `scripts/16` (`--read_slot dataset`, all 10 CSVs).
- The reader wiring was checked without the model: `07`'s argument handling with `run_reader` stubbed, prompts built
  from `cache/<ds>_rd` and `cache/<ds>_rd_tune`, the reuse script as a dry run, and the driver's guard, which exited
  while the ICEWS18 reader was running. One check started the real `run_reader` by mistake: it stopped at the MLX
  import (not installed in `.venv`), loaded no model, and left an empty `results/qwen_timeline_x/`, which was
  removed.
- Writes: `cache/icews`, `cache/cron` (new files only; `top1000_*`, `q_emb.npy` and `fact_emb.npy` are read only),
  `results/`, `logs/` and the new code files.
- Protected reader / evaluation files were not touched by this task. `scripts/07` was patched in the main analysis
  for the grounding reader run, as planned.
- No GPU was used.
