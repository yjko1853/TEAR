# Paper-side derivations

Short scripts that compute the paper's remaining numbers (cells marked "derived" in the top-level README) from the
pipeline's `cache/` and `results/`, together with their outputs as used in the paper. Run them from the repository root
with the retrieval environment, e.g. `OMP_NUM_THREADS=2 .venv/bin/python -I paper/derived/derive_ground.py`; outputs are
written next to the scripts. No model is loaded and nothing is written to `cache/` or `results/`.

Files with the suffix `_ground` belong to the paper's main setting (original questions with anchor grounding, dataset
`multitq_orig`, tag `rule_v2_ground_exact_anydate`); `_v2` and unsuffixed files belong to the earlier configurations on
TimeR4's rewritten questions (`multitq`, tags `rule_v2_exact_anydate` and `rule_exact_anydate`).

| script | outputs | inputs besides `cache/<ds>/` questions, labels, facts and rerank arrays |
|---|---|---|
| `derive_stats.py` | `group_structure.csv`, `top15_redundancy.csv`, `rule_firing_shares.csv` | MultiTQ, ICEWS-Actor and Cron caches; `results/*_constraint_recognition_*.csv` |
| `derive_ground.py` | `top15_redundancy_ground.csv`, `rule_firing_shares_ground.csv`, `llmfree_hit1_sample6000_ground.csv`, `labels_ground.csv`, `rule_check_accuracy_ground.csv`, `figure1_example_ground.txt` | `cache/multitq_orig/anchors.json`, `top1000_ids.npy`; check sheets (below) |
| `derive_v2.py` | `top15_redundancy_v2.csv`, `rule_firing_shares_v2.csv`, `llmfree_hit1_{sample6000,tune120}_v2.csv`, `query_and_dates_v2.csv`, `rule_labeller_timing_v2.csv`, `figure1_example_v2.txt` | `cache/multitq/questions_v2.json`, `topic_ner_all_v2.json` |
| `derive_sample_stats.py` | `llmfree_hit1_sample6000.csv`, `rule_labeller_timing.csv` | `cache/multitq/topic_ner_all.json` |
| `derive_tune_and_rewrites.py` | `llmfree_hit1_tune120.csv`, `rewrite_defects.csv`, `before_after_date_parsing.csv`, `timer4_prompt_bench_check.csv` | `cache/multitq/tune_120.json`, `results/qwen/*p-timer4__bench.jsonl` |
| `derive_reader_ground.py` | `reader_rules_ground.csv`, `reader_vs_llmfree_ground.csv`, `reader_leaderboard_ground.csv` | `results/qwen_ground/*.jsonl`, `results/qwen_ground_final_per_question.csv` |
| `derive_reader_icews18.py` | `reader_icews18_summary.csv`, `reader_icews18_significance.csv` | `results/qwen_icews18/*.jsonl`; checks `results/qwen_icews18_final_summary.csv` |
| `error_analysis_ground.py` / `error_analysis_v2.py` | `error_breakdown_*.csv`, `error_propagation_*.csv`, `case_candidates_*.txt` | — |
| `qualitative_cases_ground.py` | `qualitative_cases_ground.{csv,txt}`, `reader_errors_ground.csv` | `results/qwen_ground/*.jsonl`, `results/qwen_ground_final_per_question.csv` |
| `figure1_example.py`, `figure1_ground.py` | standard output, saved as `figure1_example.txt`, `figure1_flags_ground.txt` | `top1000_{ids,scores}.npy` |
| `paraphrase_miss_forms.py` | `paraphrase_miss_forms.csv` | `cache/multitq_para_src/paraphrases.json`, `results/paraphrase_robustness_per_question.csv` |
| `rescore_timer4_released.py` | `timer4_released_rescored.csv`, `timer4_released_rescored_sample6000.csv` | TimeR4's released predictions (`results.zip` of the TimeR4 repository, Git LFS) unpacked to `external/TimeR4_results/unzipped/` |
| `rule_check_accuracy.py` | `rule_check_accuracy.csv` | check sheets (below) |
| `llmfree_significance.sh` | `llmfree_significance_vs_{none,timer4,distance_penalty_v2,distance_penalty_ground}.csv`, `llmfree_hit1_substring_atc.csv` | `scripts/06_llmfree_eval.py` with another baseline or tag, into a temporary folder |

The question files, labels, anchors, reader samples and top-30 arrays of the main setting are in `artifacts/`
(`artifacts/artifacts.py unpack`); facts, vocabularies, the top-1000 arrays and the arrays of the earlier configurations
come from running the pipeline (top-level README).

**Check sheets.** The blind check of the answer-slot rules (Supplementary S9.1) was judged by the first author in
private working sheets that are not released. `rule_check_accuracy.py` and the check part of `derive_ground.py` read
them from `TEAR_CHECK_DIR` (file, sheet and column names as in the original sheets); without them,
`rule_check_accuracy.py` stops and `derive_ground.py` skips that part, and the released CSVs stand as they are.

**Verification (2026-10-31).** Every script, and each `scripts/06` call of `llmfree_significance.sh`, was run from a
copy of this repository against the pipeline's `cache/` and the released `results/`, and every output was compared
with the file used for the paper. All are byte-identical except `rule_labeller_timing*.csv`, whose `seconds` columns
are wall-clock times. `derive_sample_stats.py` now passes `version="v1"` to the labeller, because v2 became the
default after that file was written.
