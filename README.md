# TEAR: training-free temporal-constraint and answer-type aware reranking for temporal KGQA

TEAR reranks the facts that a retriever returns for a temporal question before an LLM reads them. It needs no
training and no API; every rule reads the question text and the knowledge graph:

- **ATC (answer-slot penalty).** The question's answer slot (subject, object or timestamp) and its topic entity are
  predicted from the question grammar (`tear/labeling/rules.py`). Facts whose answer slot holds the topic entity
  cannot contain the answer there and are down-weighted (`score x lambda1`).
- **TR (representative-fact rule).** For `first` / `last` questions, candidates are grouped by the slots that must
  stay fixed. Within the before/after window, each group keeps its earliest or latest fact; the other members are
  down-weighted (`score x lambda2`).
- **AG (anchor grounding).** Constraints anchored on another event ('After the Danish Ministry ..., who was the first
  to visit Iraq?') are resolved in the KG: the anchor event's date becomes the reference date
  (`tear/labeling/anchor.py`; interval-valued graphs: `tear/labeling/interval_anchor.py`).
- **Window.** `tear_window` adds a before/after (and, with `--equal_window`, same day/month/year) window for every
  dated or grounded question.

All rerankers, including the baselines, operate on the same top-1000 candidate pool of an off-the-shelf retriever,
so differences between them are not confounded by retrieval. The paper evaluates on MultiTQ, TimelineKGQA
(ICEWS-Actor, CronQuestions-KG) and MultiTQ-style questions generated over ICEWS18 (`tear/data/qgen`).

This repository holds the code, the generated ICEWS18 questions, compact copies of the cached candidate lists and of
the rule and grounding outputs (`artifacts/`), per-question predictions and result tables (`results/`) and the
pre-registration notes (`results/prereg_*.md`). Versions of this repository before 2026 held code of an earlier,
trained version of TEAR; that state is kept under the tag `v1-legacy`.

## Repository layout

```
tear/                       library (lower layers never import higher ones)
  paths.py                  ROOT / EXTERNAL / CACHE / RESULTS / LOGS  (TEAR_HOME, TEAR_LOGS override)
  utils/                    io.py (JSON), timing.py (logs/timing.csv), device.py (torch device)
  text.py                   answer matching: normalize, match, exact_match, lenient_match (original TEAR rule)
  temporal.py               dates, cue words, before/after target dates (original regex + extended parser)
  data/                     multitq.py, timelinekgqa.py, paraphrase.py (preparation), candidates.py (CandidatePool),
                            qgen/ (MultiTQ-style question generator: templates frozen from MultiTQ train/dev)
  retrieval/                encoder.py (all-mpnet-base-v2; --model for others), search.py (exact FAISS / numpy top-N)
  labeling/                 rules.py (answer-slot + topic rules), ner.py (original NER port), oracle.py (diagnostic),
                            anchor.py (event-anchored constraints: reference date looked up in the KG),
                            interval_anchor.py (the same for interval facts), granularity.py (answer unit)
  rerank/                   atc.py, tr.py, baselines.py, features.py, core.py (reranker registry)
  reader/                   prompt.py (reader prompts), parse.py, mlx_reader.py (local LLM reader)
  evaluation/               coverage.py (coverage@k, gold recall), hits.py (Hit@1), significance.py
  analysis/                 motivation.py, runs.py (per-question metrics of a run), qgen_fidelity.py
  reporting.py              results/results.md builder
scripts/                    command-line stages 01-26 (table below)
logs/*.sh                   driver scripts that ran the stages for the paper; logs/timing.csv (wall-clock times);
                            logs/timeline_prompt_choice.json (TimelineKGQA prompt choice); the reader logs cited
                            for cost (qwen_ground_run.log, qwen_ground_dedup_run.log, timeline_reader.log)
resources/qgen/             frozen templates + generator config with SHA-256 and FREEZE_LOG.md
artifacts/                  released intermediate files (artifacts/README.md): reranked top-30 lists, rule labels,
                            anchors, ICEWS18 questions, reader samples, paraphrase set
results/                    CSV tables, reports (*.md), reader predictions (qwen*/ *.jsonl) and their scores
paper/derived/              scripts that compute the paper's remaining cells from cache/ and results/, with their
                            outputs (paper/derived/README.md)
tests/                      pytest unit tests
external/   cache/          third-party data clones and generated intermediate files (git-ignored)
```

| script | stage |
|---|---|
| `01_prepare_data.py` | fact table, fact sentences, test questions -> `cache/<ds>/` (`multitq`, `multitq_orig`, `multitq_bge`, `icews`, `cron`) |
| `02_encode_facts.py` | fact embeddings (all-mpnet-base-v2, or `--model`) |
| `03_search.py` | exact top-1000 per question |
| `04_label_slots.py` | answer-slot / topic labels: `--method rule` (default), `ner` (topic fallback), `oracle` (diagnostic) |
| `05_rerank.py` | all rerankers on the shared pool; coverage@k, ablation, lambda sweep, diagnostics, top-30 arrays |
| `06_llmfree_eval.py` | LLM-free Hit@1 (answer read off the top-1 fact) and significance against a baseline |
| `07_read_with_llm.py` | local LLM reader (MLX) over each setting's top-k facts |
| `08_score_reader.py` | reader Hit@1 (exact, lenient, any-match), McNemar and paired bootstrap |
| `09_motivation.py` | similarity of facts that differ only in time |
| `10_make_report.py` | `results/results.md` |
| `11_compare_rewrite.py` | original questions vs TimeR4's rewrites |
| `12_reuse_reader_predictions.py` | reuse predictions across question/label versions when the reader input is identical |
| `13_compare_versions.py` | v1 vs v2 labels and questions |
| `14_ground_anchors.py` | anchor grounding (MultiTQ format) -> `cache/<ds>/anchors.json` |
| `14_reuse_across_settings.py` | copy predictions between rerank settings whose top-k lists are identical (greedy decoding) |
| `15_anchor_report.py` | anchor grounding vs no grounding vs rewrites |
| `16_timeline_grounding.py` | TimelineKGQA: interval anchors, rule labels, rerankers with grounding |
| `17_generate_questions.py` | MultiTQ-style questions over ICEWS18 (extract, calibrate, prepare, generate, fidelity, rulecheck, difficulty) |
| `18_final_reader_report.py` | final MultiTQ reader tables (reader, reader+date, direct, hybrid) |
| `19_timeline_reader_data.py` / `20_timeline_reader_report.py` | TimelineKGQA reader inputs / report |
| `21_paraphrase_questions.py` / `22_paraphrase_report.py` | paraphrase robustness set (local LLM) / report |
| `23_retriever_report.py` | second retriever (BAAI/bge-base-en-v1.5) |
| `24_granularity_report.py` | answer granularity at the reader |
| `25_topic_kg_check.py` | automatic KG check of the rule topic entities |
| `26_make_figures.py` | paper figures (`figures/`, or `TEAR_FIGURES=<dir>`) |
| `27_hardwindow_reader_report.py` | reader with the hard window and anchor grounding (pre-registered): gain decomposition |

The two scripts numbered 14 are independent: `14_ground_anchors.py` is a CPU labelling stage, and
`14_reuse_across_settings.py` is a reader utility used by the `logs/run_*` drivers.

## Setup

Two environments are used on Apple silicon, because the MLX reader pulls newer numpy/transformers versions than the
retrieval stack:

```bash
# retrieval, reranking, evaluation (tested: Python 3.11.4, torch 2.0.1, sentence-transformers 3.4.1,
# transformers 4.49.0, faiss-cpu 1.15.1, numpy 1.26.4, pandas 1.5.3, rapidfuzz 3.14.6, pytest 7.4.0)
python3.11 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"        # add ".[dev,figures]" for scripts/26 (matplotlib)

# local LLM reader (tested: mlx 0.32.4, mlx-lm 0.32.2)
python3.11 -m venv .venv-mlx
.venv-mlx/bin/python -m pip install mlx-lm numpy pandas

# data (read only; no code from these clones is executed)
mkdir -p external && cd external
GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 https://github.com/qianxinying/TimeR4 TimeR4   # MultiTQ KG, test questions, TimeR4 rewrites
git clone --depth 1 https://github.com/PascalSun/TimelineKGQA TimelineKGQA                # ICEWS-Actor, CronQuestions-KG
```

The question generator also reads MultiTQ train / dev (`external/MultiTQ_hf/{train,dev}.json`, Hugging Face dataset
`chenziyang/MultiTQ`) and the RE-NET release of ICEWS18 (`external/ICEWS18_RE-Net/`, the `data/ICEWS18` folder of the
RE-NET repository). ICEWS data are licensed for research and education only.

Models are downloaded from the Hugging Face hub on first use: `sentence-transformers/all-mpnet-base-v2` (retriever),
`BAAI/bge-base-en-v1.5` (second retriever), `dslim/bert-base-NER` (NER port), `mlx-community/Qwen2.5-7B-Instruct-4bit`
(reader and paraphrases).

## Released artefacts

`artifacts/` holds compact copies of the intermediate files that are expensive to recompute (GPU encoding, hours of
reader time) or that define the evaluation data: for the main setting (`multitq_orig`), the second retriever
(`multitq_bge`) and ICEWS18, the reranked top-30 fact ids of every reranker (tag `rule_v2_ground_exact_anydate`),
the rule-v2 labels, the anchor grounding output, the reader samples, the generated ICEWS18 questions and the
paraphrase set. `artifacts/README.md` lists each file, its shape and the script that writes it.

```bash
.venv/bin/python -I artifacts/artifacts.py unpack     # -> cache/<ds>/..., byte-identical to the pipeline's files
```

Fact ids index lines of `cache/<ds>/sentences.txt`, which `scripts/01` (MultiTQ) and `scripts/17 prepare` (ICEWS18)
rebuild from the public data. With the questions (`scripts/01`), the unpacked arrays are enough to run the reader
(`scripts/07`) and the LLM-free evaluation (`scripts/06`) without encoding or search. The full top-1000 pools and the
embeddings are not included (they are rebuilt by `scripts/02` and `scripts/03`).

## Reproducing the paper

Run every command from the repository root with `P=".venv/bin/python -I"` and `M=".venv-mlx/bin/python -I"`.
Every stage is resumable and appends its wall-clock time to `logs/timing.csv`. On a fanless machine, cap the
threads (`export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2`); the drivers in `logs/` do this.

### Main setting: original MultiTQ questions with anchor grounding

```bash
# TimeR4-rewritten questions (needed for the comparison with rewriting)
$P scripts/01_prepare_data.py --dataset multitq                   # facts, sentences, test questions (TimeR4 copy)
$P scripts/02_encode_facts.py --dataset multitq                   # fact embeddings (MPS, ~30 min for 461k facts)
$P scripts/03_search.py       --dataset multitq                   # exact top-1000
$P scripts/04_label_slots.py  --dataset multitq --method ner --labels all   # NER topic fallback for the rules
bash logs/build_v2.sh          # questions_v2.json, changed rows of search and NER, rule v2 labels, rerank, Hit@1
# original questions (main setting)
$P scripts/01_prepare_data.py --dataset multitq_orig              # original questions; fact files are symlinks
$P scripts/03_search.py       --dataset multitq_orig --device cpu
$P scripts/04_label_slots.py  --dataset multitq_orig --method ner --labels all --device cpu --verify_sample 0
bash logs/build_orig_v2.sh     # rule v2 labels -> cache/multitq_orig/slots_rule_v2.json; run without grounding
$P scripts/14_ground_anchors.py --dev_design                      # design check on the dev split
bash logs/build_anchor.sh      # anchors.json; rerank + LLM-free Hit@1 (tag rule_v2_ground_exact_anydate); rewrite runs
$M scripts/07_read_with_llm.py --make_sample                      # reader sample: 1,000 questions per type (seed 13)
ln -s ../multitq/sample_6000.json cache/multitq_orig/sample_6000.json   # same quids for both wordings
$P scripts/15_anchor_report.py                                    # results/anchor_grounding_*.csv
bash logs/run_qwen_chunks_ground_dedup.sh v3                      # Qwen reader, 6,000 questions x 7 settings
$P scripts/18_final_reader_report.py                              # results/qwen_ground_final_*.csv
```

The main reranking call inside `logs/build_anchor.sh` is

```bash
$P scripts/05_rerank.py --dataset multitq_orig --slots cache/multitq_orig/slots_rule_v2.json \
   --date_granularity any --date_parser v2 --atc_match exact \
   --anchors cache/multitq_orig/anchors.json --equal_window --sweep --tag rule_v2_ground_exact
$P scripts/06_llmfree_eval.py --dataset multitq_orig --tag rule_v2_ground_exact_anydate \
   --slots cache/multitq_orig/slots_rule_v2.json --baseline none
```

### Other experiments

| experiment | commands | main outputs |
|---|---|---|
| second retriever (pre-registered, part A) | `bash logs/run_bge_and_granularity.sh` (A1-A7) | `results/retriever_comparison.{csv,md}`, `results/multitq_bge_*` |
| answer granularity (pre-registered, part B) | same driver, part B; `scripts/24` | `results/granularity*.{csv,md}` |
| paraphrase robustness (pre-registered) | `scripts/21 --select / --generate / --build`, `bash logs/run_paraphrase_pipeline.sh`, `scripts/22` | `results/paraphrase_robustness*` |
| TimelineKGQA grounding | `scripts/01`-`03 --dataset icews` (and `cron`); `scripts/16 --dataset <ds> --dev_design`; `scripts/16 --dataset <ds>` (dataset answer types, tag `oracle_ground`); `scripts/16 --dataset <ds> --make_rule_labels`, then `--slots cache/<ds>/slots_rule_v2.json --tag rule_ground` | `results/{icews,cron}_grounding_*.csv`, `results/{icews,cron}_rule_label_accuracy.csv` |
| TimelineKGQA reader | `scripts/19`, `bash logs/run_timeline_reader.sh`, `scripts/20` | `results/qwen_timeline_final*` |
| ICEWS18 questions | `scripts/17_generate_questions.py extract / calibrate / prepare / generate / fidelity / rulecheck` (below) | `cache/icews18/questions*.json` (released in `artifacts/icews18/`), `results/qgen_*`, `results/addendum_icews18_benchmark.md` |
| ICEWS18 retrieval, rerank, reader | `caffeinate -i bash logs/run_icews18.sh` | `results/icews18_*`, `results/qwen_icews18*` |
| hard window reader (pre-registered) | `caffeinate -i bash logs/run_hardwindow_reader.sh`, `scripts/27` | `results/hardwindow_reader*`, `hardwindow_*` in `results/qwen_ground/` and `results/qwen_icews18/` |
| no-rewrite ablation, v1 vs v2 | `scripts/11`, `scripts/13`, `logs/build_v2*.sh` | `results/multitq_orig_vs_rewrite_*`, `results/addendum_*.md` |
| motivation, topic check, report | `scripts/09`, `scripts/25`, `scripts/10` | `results/multitq_same_sro_similarity.csv`, `results/topic_kg_check.csv`, `results/results.md` |
| figures | `scripts/26_make_figures.py` | `figures/fig_overview.pdf`, `figures/fig_gains.pdf` and `fig_gains_values.csv` |
| remaining paper cells | `paper/derived/*.py`, `paper/derived/llmfree_significance.sh` (run from the repository root) | `paper/derived/*.csv`, `*.txt` |

### Where each table's numbers come from

Unless stated otherwise the tag is `rule_v2_ground_exact_anydate`, `<ds>_` is `multitq_orig_` and files are in
`results/`. Cells marked "derived" come from `paper/derived/` (scripts and outputs listed in `paper/derived/README.md`).

| paper item | results files |
|---|---|
| MultiTQ data (`tab:data`) | `cache/multitq_orig/questions.json` (scripts/01); anchored counts `anchor_grounding_accuracy.csv`, `addendum_anchor_grounding.md`; samples `artifacts/multitq_orig/sample_6000.json.gz` |
| Rerankers on MultiTQ (`tab:main`) | `<ds>_coverage_main_<tag>.csv` (k = 15), `<ds>_llmfree_hit1_<tag>.csv`, `<ds>_llmfree_significance_<tag>.csv`, `<ds>_slot_occupancy_top15_<tag>.csv`, reader `qwen_ground_final_summary.csv` / `qwen_ground_final_significance.csv`, reference row `multitq_timer4_public_top15.csv`; derived: `llmfree_significance_vs_distance_penalty_ground.csv` (difference to the distance penalty), `timer4_released_rescored_sample6000.csv` (fine-tuned TimeR4 reader row) |
| Hard window reader (Tables 3, 4; S13, S14, S18) | `scripts/27_hardwindow_reader_report.py` -> `hardwindow_reader.md`, `hardwindow_reader_summary.csv`, `hardwindow_reader_significance.csv`; pre-registration `prereg_hardwindow_reader.md` |
| Hit@1 by question type (`tab:types`) | `<ds>_llmfree_hit1_<tag>.csv` (`exact_<qtype>`), `qwen_ground_final_summary.csv` (`reader_date`, `qt_*`) |
| Grounding vs rewriting (`tab:grounding`) | `anchor_grounding_summary.csv`, `anchor_grounding_significance.csv`, `anchor_grounding_accuracy.csv`, `addendum_anchor_grounding.md` |
| Second retriever (`tab:retriever`) | `retriever_comparison.csv`, `retriever_comparison.md`, `multitq_bge_*_<tag>.csv` |
| Answer granularity (`tab:granularity`) | `granularity_summary.csv`, `granularity_paired.csv`, `granularity.md` |
| LLM-free errors (`tab:errors`) | derived: `error_breakdown_ground.csv` (`error_analysis_ground.py`), `reader_errors_ground.csv` (`qualitative_cases_ground.py`) |
| Published results (`tab:leaderboard`) | numbers from the cited papers; TEAR rows from `<ds>_llmfree_hit1_<tag>.csv`; derived: `reader_leaderboard_ground.csv` (TEAR + Qwen row), `timer4_released_rescored.csv` (TimeR4 under exact Hit@1) |
| TimelineKGQA grounding (`tab:tl`) | `{icews,cron}_grounding_coverage.csv`, `_coverage_rule_ground.csv`, `_event_recall.csv`, `_significance_rule_ground.csv`; reader `qwen_timeline_final_summary.csv`, `qwen_timeline_final_paired.csv` |
| Conditions for ATC and TR (`tab:timeline`) | `{icews,cron}_rule_label_accuracy.csv`, `{icews,cron}_slot_occupancy_top15_oracle.csv`, `{icews,cron}_grounding_coverage*.csv`, `multitq_orig_slot_occupancy_top15_rule_v2_exact_anydate.csv`, `anchor_grounding_summary.csv`; derived: `group_structure.csv`, `top15_redundancy{,_ground}.csv`, `rule_firing_shares{,_ground}.csv` |
| ICEWS18 checks (main text and supplementary section) | `icews18_coverage_main_<tag>.csv`, `icews18_llmfree_hit1_<tag>.csv`, `icews18_llmfree_hit1_rule_v2_exact_anydate.csv`, `icews18_final_qgen_rulecheck_*.csv`, `qgen_fidelity_*.csv`, `qgen_difficulty.csv`, `qwen_icews18_final_summary.csv` (scoring rules of scripts/18 on `qwen_icews18/`), `addendum_icews18_benchmark.md`; derived: `reader_icews18_summary.csv`, `reader_icews18_significance.csv` (`derive_reader_icews18.py`, which also checks the summary file) |
| Sensitivity to k and lambda (supplementary) | `<ds>_coverage_main_<tag>.csv` (k = 5..30), `<ds>_lambda_sweep_<tag>.csv` |
| Exact vs substring topic match (supplementary) | `<ds>_*_rule_v2_ground_substring_anydate.csv`, `substring_vs_exact_main_setting.csv` |
| Rule accuracy and constraint recognition (supplementary) | `<ds>_constraint_recognition_<tag>.csv`, `topic_kg_check.csv`, `addendum_rule_v2.md`; derived: `rule_check_accuracy.csv`, `rule_check_accuracy_ground.csv` (first author's blind check), `labels_ground.csv`, `error_propagation_ground.csv` |
| Reader details (supplementary) | `qwen_ground_final_summary.csv`, `qwen_ground_final_significance.csv`, `qwen_ground_final.md`, prompts in `tear/reader/prompt.py`; derived: `reader_rules_ground.csv`, `reader_vs_llmfree_ground.csv`, `timer4_released_rescored_sample6000.csv`, `timer4_prompt_bench_check.csv` |
| Qualitative cases (supplementary) and the example figure | derived: `qualitative_cases_ground.{csv,txt}`, `figure1_example_ground.txt`, `figure1_flags_ground.txt` |
| Paraphrase robustness (supplementary) | `paraphrase_robustness{,_anchors,_llmfree,_paired,_breakdown}.csv`, `paraphrase_robustness.md`, `prereg_paraphrase_robustness.md`; derived: `paraphrase_miss_forms.csv` |
| TimelineKGQA details (supplementary) | `{icews,cron}_grounding_{patterns,constraints,coverage,llmfree,significance,event_recall}[_rule_ground].csv`, `qwen_timeline_final*`, `logs/timeline_prompt_choice.json`, `addendum_timelinekgqa_grounding.md` |
| Metrics and TimeR4's published numbers (supplementary) | `timer4_metric_audit.md`; derived: `timer4_released_rescored.csv` (TimeR4's released predictions rescored per type with `tear.evaluation.hits.score_reader_record`) |
| Cost (supplementary) | `logs/timing.csv`, `logs/qwen_ground_run.log`, `logs/qwen_ground_dedup_run.log`, `logs/timeline_reader.log` |
| Figures | `scripts/26_make_figures.py` from the CSVs above |

Pre-registration notes: `results/prereg_retriever_and_granularity.md` (second retriever, answer granularity),
`results/prereg_paraphrase_robustness.md` (paraphrase robustness, with its amendments and deviation) and
`results/prereg_hardwindow_reader.md` (reader with the hard window and anchor grounding; `scripts/27`). Each was
written before the outputs it covers existed; later changes are listed in the note itself.

### Outputs and settings

`05_rerank.py` writes `results/<ds>_{coverage_main,ablation,lambda_sweep,constraint_recognition,slot_occupancy_top15}_<tag>.csv`
and `cache/<ds>/rerank/<reranker>_<tag>_top30_ids.npy` (row order in `question_order.json`; fact *i* is line *i* of
`cache/<ds>/sentences.txt`). `--results_dir` / `--rerank_dir` redirect the outputs. `_anydate` is appended to the tag
when `--date_granularity any`.

Rerankers: `none` (similarity order), `timer4` (TimeR4 Eq. 9-10, before only), `timer4_sym` (also after),
`chrono` (chronological sort of the top-P), `hardwindow` (before/after window, no labels), `atc`, `tr`, `tear`
(= ATC + TR), `tear_window` (TEAR + window).

Reader: `mlx-community/Qwen2.5-7B-Instruct-4bit`, greedy, k = 15. Prompts are in `tear/reader/prompt.py`: `timer4`
(the TimeR4 prompt, written for a fine-tuned reader), `v2` / `v3` (zero-shot instruction prompts; v3 is used for the
MultiTQ and ICEWS18 runs), `v3i` (v3 for interval facts, TimelineKGQA) and `v3g` (v3 plus the predicted answer
unit; the pre-registered granularity variant). v3 and v3i were chosen on tuning sets disjoint from the evaluation
samples with the `none` reranker (`cache/multitq/tune_120.json` from `--make_tune`; `logs/timeline_prompt_choice.json`)
and then frozen for every reranker.
The evaluation sample is read in one fixed shuffled order shared by every setting, so `--start/--end` chunks (and any
prefix) are paired across settings. Predictions go to
`results/<pred_dir>/<setting>[_<tag>]__<model>__k<k>__p-<prompt>[__bench][__tune].jsonl` and are resumable; records
copied by `scripts/14_reuse_across_settings.py` carry `copied_from`.

Metrics: **answer coverage@k** (a top-k fact element matches a gold answer under the TEAR `match` rule), **gold-event
recall@k** (TimelineKGQA), **LLM-free Hit@1** (answer read off the top-1 fact's predicted slot), **reader Hit@1**
(exact and lenient on the first listed answer; TimeR4-style any-match over the list), McNemar and paired bootstrap
tests (`tear/evaluation/significance.py`).

### Question and label versions

Version 2 of the questions and answer-slot rules is the paper's configuration. 380 TimeR4 rewrites are empty (341)
or truncated (39, e.g. `After 2012-11-21, who was`); `questions_v2.json` adds a `query` field = the rewrite when
valid, otherwise the original question (`tear.data.multitq.query_text`), and every stage reads `stage_query(q)`
(`query` when present, else `rewrite`, so v1 files and `multitq_orig` are unchanged). Rule v2 (`--rule_version v2`,
the default) fixes five classifier errors found in two checks (see `tear/labeling/rules.py`). Date parser v2
(`--date_parser v2`) also reads before/after dates such as `Sep, 2011`, `Sept. 2011`, `Sep 5, 2011` and
`5 Sep 2011` (v1 stays the default so earlier runs reproduce). Some addenda and driver comments mention
`patches/*.patch`; those changes are already part of the code in this repository.

### ICEWS18 questions

No public benchmark other than MultiTQ has point-in-time event facts, (s, r, o) repeated over days, first/last and
dated before/after cues, and subject/object answers. `tear/data/qgen` re-implements MultiTQ's template procedure (its
generation code is not public) and applies it to ICEWS18 (RE-NET release, 2018-01-01..10-31, 468,558 facts, no
overlap with ICEWS05-15). Details and checks: `results/addendum_icews18_benchmark.md`.

- **Templates** (`extract`): MultiTQ train + dev questions (the test split is not read) are aligned to ICEWS05-15: a
  structure (answer side, relation, topic / anchor / date spans) is kept only when MultiTQ's answer procedure
  (`tear/data/qgen/semantics.py`) reproduces the gold answer set exactly; structures the KG cannot tell apart
  (reciprocal or symmetric relations) are decided by a frame bigram model and a relation word model trained on the
  unambiguous questions (not by TEAR's rules). Frozen in `resources/qgen/multitq_templates_v1.json.gz` (+ `.sha256`,
  `FREEZE_LOG.md`).
- **Generator v2** (`calibrate`): regularities of MultiTQ *dev* questions; `resources/qgen/generator_v2.json`.
- **Sets**: `cache/icews18/questions.json` (6,000 = 1,000 per type, seed 2026), `questions_dev.json` (1,200, seed
  2027, no text shared with test); both are in `artifacts/icews18/` with their `.meta.json` (seed, config, rejections,
  SHA-256). The question format is the MultiTQ one of `scripts/01` (`rewrite` = the question, empty TimeR4 top-15; the
  `timer4_public` setting does not apply) plus a `gen` record: answer slot, topic, anchor and its date, date
  expression, gold fact ids, template. ICEWS18 spans one year, so year-level buckets are left out.

```bash
$P scripts/17_generate_questions.py extract      # templates (refuses to overwrite the frozen file)
$P scripts/17_generate_questions.py calibrate    # generator v2 config from MultiTQ dev
$P scripts/17_generate_questions.py prepare      # external/ICEWS18_RE-Net -> cache/icews18 (MultiTQ cache format)
$P scripts/17_generate_questions.py generate --dataset icews0515_regen --n_per 1000 --seed 2025
$P scripts/17_generate_questions.py generate --dataset icews18 --n_per 1000 --seed 2026
$P scripts/17_generate_questions.py generate --dataset icews18 --n_per 200 --seed 2027 --out questions_dev.json \
    --exclude questions.json --quid_start 5100000
$P scripts/17_generate_questions.py align-test   # MultiTQ test aligned with the frozen models (comparison only)
$P scripts/17_generate_questions.py fidelity     # results/qgen_fidelity_{text,kg,date_shapes,topics}.csv
$P scripts/17_generate_questions.py rulecheck --dataset icews18    # rule v2 / dates / cues / anchors vs the record
caffeinate -i bash logs/run_icews18.sh           # encode, search, NER, anchors, rerank, LLM-free Hit@1, reader
$P scripts/17_generate_questions.py difficulty   # coverage@15 / Hit@1: MultiTQ vs regenerated vs ICEWS18
```

`tear.data.is_multitq` treats `icews18` and `icews0515_regen` as MultiTQ-format datasets, so stages 02-08 run on them
with `--dataset`.

## Changes relative to the earlier TEAR code (tag `v1-legacy`)

| Earlier code | This repository | Reason |
|---|---|---|
| answer type by gpt-4o-mini few-shot voting (`answer_slot_multi_vote_1call.py`) | rule-based answer-slot classifier (`tear/labeling/rules.py`, v2 default, v1 kept) | no API; deterministic; KG-derived labels are unreliable on reciprocal ICEWS events |
| TimeR4 rewrite used as is (341 empty, 39 truncated) | original questions with anchor grounding (main setting); `query` = rewrite when valid, else the original question (`questions_v2.json`) for the rewrite runs | no LLM rewriting step; empty or truncated questions cannot be answered |
| ATC topic match by substring (`basic_result`) | `--atc_match exact` (entity identity); substring kept as an option | substring also penalises e.g. `Citizen (Iraq)` for topic `Iraq` |
| learnable temporal-cue embedding added to SBERT (`fine_tuned_retriever_*`) | removed | it was not used by the retrieval code at inference |
| fine-tuned SBERT + FAISS IVF index | off-the-shelf all-mpnet-base-v2 + exact `IndexFlatIP` | no training; deterministic search |
| `before\|after YYYY-MM-DD` regex only | `--date_granularity any`: also `12 May 2011`, `May 2011`, `2011-05`, `2011` | the TimeR4 rewrites use these forms (0.6% vs 48.9% of before/after questions parsed) |
| TR only (cue and answer type required) | adds `tear_window`: before/after window for every dated question | TR never fires on plain before/after questions |
| fine-tuned Llama2 reader (checkpoint lost) | local `Qwen2.5-7B-Instruct-4bit` via MLX, zero-shot | no GPU training, no API |
| — | anchor grounding, LLM-free Hit@1, paired significance tests, unit tests | evaluation without a reader; implicit constraints without rewriting |

Ported verbatim: answer `normalize` / `match`, cue extraction, the original date regex, group keys
(subject -> (r, o), object -> (s, r), timestamp -> (s, r, o)), before/after validity, first -> earliest /
last -> latest representative, `lambda1 = lambda2 = 0.8`.

## Implementation notes

- **torch + faiss on macOS.** Loading torch and faiss in one process hangs at `index.search` on Apple silicon (two
  OpenMP runtimes). `03_search.py` encodes questions in a child process; `--engine numpy` is an exact fallback.
- **TimelineKGQA facts are intervals.** First/last use `start_time`; `before d` requires `end < d`, `after d`
  requires `start > d`; `from X to Y` questions get a containment window; open ends are +-1e8 days. The released
  `split` column gives 57,453 / 17,298 / 14,621 questions for ICEWS-Actor (the dataset README reports
  53,624 / 17,874 / 17,874); the `split` column is used. The `paraphrased_question` column is misaligned for
  medium/complex questions, so the original `question` is used.
- **NER port.** The original keeps label ids 1 and 2, which are B-MISC / I-MISC in `dslim/bert-base-NER`
  (`--labels faithful`); `--labels all` keeps every entity label. Fuzzy matching uses rapidfuzz `WRatio`, the same
  scorer as fuzzywuzzy's `extractOne`.
- **KG-oracle labels** (`--method oracle`) vote over every fact linking the answer and a question entity regardless of
  the relation; ICEWS codes many events in both directions, so they are kept for diagnostics only.

## Verification

`pytest` (151 tests) covers the answer-slot rules (hand-judged cases; rule v2 fixes with the v1 error each
documents), the query fallback, date parsing, ATC exact vs substring, TR representatives, the TimeR4 phi, anchor and
interval-anchor parsing, answer granularity, answer-list parsing, the significance tests, the multitq_orig variant,
prediction reuse and paraphrase handling (synthetic fixtures), and the question generator (`tests/test_qgen.py`: date
formats, surface forms, every answer procedure, alignment, generator rules, RE-NET loading, the frozen-file hash
check). The tests need no data in `cache/` or `external/`.

The library reproduces the outputs of the flat scripts that produced the first results of this reimplementation
(not released) exactly: data preparation (MultiTQ, ICEWS, Cron), search (first 300 questions, ids and scores), rule /
NER / oracle labels, all reranker CSVs and top-30 id arrays (MultiTQ rule+exact and oracle+substring, ICEWS oracle),
LLM-free Hit@1, the motivation analysis, the reader sample and tuning set, reader prompts / question order / file
names, the scoring of existing predictions, and `results.md` (apart from its timestamp line). Fact encoding agrees
within 2e-7 (MPS floating point). After the v2 additions the default flags still reproduce the v1 files exactly (rule
v1 labels, rerank CSVs and top-30 arrays, LLM-free Hit@1, search), and the v2 question embeddings and search arrays
differ from v1 only in the 380 fallback rows.

## License

The code is released under the MIT License (`LICENSE`). The files in `artifacts/` and `results/` are derived
from MultiTQ, ICEWS05-15, ICEWS18 and TimelineKGQA and remain subject to the terms of those datasets; ICEWS data are
licensed for research and education only.
