# Released artefacts

Compact copies of intermediate files under `cache/` that the paper's results depend on. They are byte-identical to the
pipeline's files after unpacking (`SHA256SUMS` lists the packed files):

```bash
.venv/bin/python -I artifacts/artifacts.py unpack            # -> cache/<ds>/...; existing files are kept (--force)
```

`artifacts.py pack --cache <cache>` rebuilds this folder from a full `cache/` (numpy only). Each `.npz` holds one
reranking run with tag `rule_v2_ground_exact_anydate` (rule-v2 labels, date parser v2, exact-match ATC, anchor
grounding with the equal window, lambda1 = lambda2 = 0.8):

| key | content |
|---|---|
| `ids` | int32, questions x 9 settings x 30: top-30 fact ids of each reranker, best first; fact *i* is line *i* of `cache/<ds>/sentences.txt` |
| `settings` | reranker names along axis 1: `none, timer4, timer4_sym, chrono, hardwindow, atc, tr, tear, tear_window` |
| `question_ids` | row order (= `rerank/question_order.json`) |
| `tag`, `rerank_dir` | run tag and the `cache/<ds>/` sub-folder that `unpack` writes `<setting>_<tag>_top30_ids.npy` to |

```python
import numpy as np
z = np.load("artifacts/multitq_orig/rerank_rule_v2_ground_exact_anydate_top30.npz")
top15 = z["ids"][:, list(z["settings"]).index("tear_window"), :15]
```

## Files

| file | content | written by |
|---|---|---|
| `multitq_orig/rerank_rule_v2_ground_exact_anydate_top30.npz` | `ids` 54,584 x 9 x 30: main setting (original MultiTQ test questions, all-mpnet-base-v2 top-1000 pool) | `scripts/05_rerank.py` in `logs/build_anchor.sh` |
| `multitq_orig/slots_rule_v2.json.gz` | {quid: answer_slot (subject / object / timestamp), topic_entity, topic_source}, 54,584 questions | `scripts/04_label_slots.py --method rule --rule_version v2` (`logs/build_orig_v2.sh`; NER topic fallback from `--method ner`) |
| `multitq_orig/anchors.json.gz` | {quid: anchor type, phrase, status, anchor entity, relation, date, n_dates, qtype, timer4_date}, the 18,863 anchored questions (18,629 resolved) | `scripts/14_ground_anchors.py --dataset multitq_orig` |
| `multitq_orig/sample_6000.json.gz` | reader evaluation sample: 1,000 test quids per question type (seed 13); same quids as `cache/multitq/sample_6000.json` | `scripts/07_read_with_llm.py --make_sample` |
| `multitq_orig/sample_6000_dateslot.json.gz` | the 1,299 sample questions whose rule-v2 answer slot is timestamp (prompt v3g runs; the selection rule is stored in the file) | selected from `sample_6000.json` and `slots_rule_v2.json` for `logs/run_bge_and_granularity.sh` part B |
| `multitq_bge/rerank_rule_v2_ground_exact_anydate_top30.npz` | `ids` 54,584 x 9 x 30: the main setting with BAAI/bge-base-en-v1.5 as the retriever | `logs/run_bge_and_granularity.sh` part A |
| `multitq_bge/anchors.json.gz` | anchor grounding with the bge retriever's fact scores (same format) | `scripts/14_ground_anchors.py --dataset multitq_bge` |
| `icews18/questions.json.gz` | 6,000 generated MultiTQ-style test questions over ICEWS18 (1,000 per type, seed 2026), MultiTQ question format plus a `gen` record (answer slot, topic, anchor and date, gold fact ids, template) | `scripts/17_generate_questions.py generate --dataset icews18 --n_per 1000 --seed 2026` |
| `icews18/questions.meta.json` | seed, generator config, rejection counts and SHA-256 of templates, config, code and `questions.json` | same |
| `icews18/questions_dev.json.gz`, `questions_dev.meta.json` | 1,200 dev questions (200 per type, seed 2027, no text shared with test) | `... generate --n_per 200 --seed 2027 --out questions_dev.json --exclude questions.json --quid_start 5100000` |
| `icews18/rerank_rule_v2_ground_exact_anydate_top30.npz` | `ids` 6,000 x 9 x 30 | `logs/run_icews18.sh` (R6) |
| `icews18/rerank_dev_rule_v2_ground_exact_anydate_top30.npz` | `ids` 1,200 x 9 x 30 (dev; `results/icews18_dev/`) | `logs/run_icews18.sh` (R6) |
| `icews18/slots_rule_v2.json.gz`, `slots_rule_v2_dev.json.gz` | rule-v2 labels, 6,000 / 1,200 questions | `logs/run_icews18.sh` (R4) |
| `icews18/anchors.json.gz`, `anchors_dev.json.gz` | anchor grounding, 3,340 / 657 anchored questions (3,320 / 649 resolved) | `logs/run_icews18.sh` (R5) |
| `icews18/sample_6000.json.gz` | reader order source (all 6,000 test questions; the reader read the first 2,000 of its fixed shuffle) | `scripts/07_read_with_llm.py --dataset icews18 --make_sample` |
| `multitq_para_src/sample.json.gz` | paraphrase-robustness sample: 600 multi-constraint test quids (200 per type, seed 2026) | `scripts/21_paraphrase_questions.py --select` |
| `multitq_para_src/paraphrases.json.gz` | {quid: paraphrase, raw output, valid, reasons, lost entities, constraints of original and paraphrase}; 439 of 600 valid | `scripts/21_paraphrase_questions.py --generate` (local Qwen2.5-7B-Instruct-4bit, greedy); `valid` is the pre-registered automatic filter (`--recheck` recomputes it) |

## Not included

- MultiTQ questions and facts (`cache/multitq*/questions.json`, `facts.npz`, `sentences.txt`, `vocab.json`): rebuilt from
  the TimeR4 copy of MultiTQ by `scripts/01_prepare_data.py`. ICEWS18 facts: `scripts/17_generate_questions.py prepare`
  from the RE-NET release.
- Full top-1000 candidate pools (`top1000_ids.npy`, `top1000_scores.npy`; 437 MB for MultiTQ) and embeddings
  (`fact_emb.npy`, 1.4 GB per retriever): `scripts/02_encode_facts.py` and `scripts/03_search.py`.
- Other tags (v1 rules, rewrite runs, runs without grounding, substring ATC) and the TimelineKGQA rerank arrays: rerun
  `scripts/05_rerank.py` / `scripts/16_timeline_grounding.py` after search (CPU).
- Reader predictions are not here but in `results/qwen*/` (one JSON line per question).
