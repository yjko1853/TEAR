# Audit of TimeR4's reported Hits@1 (2026-10-17)

Source: TimeR4's own released predictions (`results.zip` from github.com/qianxinying/TimeR4, Git LFS, 54,478,447 bytes;
unpacked to `external/TimeR4_results/`). The README of that repository states that the fine-tuned LLaMA2 results in
`results/` are the ones reported in the paper. Predictions are in test.json order (54,584/54,584 ground truths match).

## Result

| dataset | rule | overall |
|---|---|---|
| MultiTQ (n=54,584) | any-match over the generated list, gold substring (TimeR4 repo `eval_hit`) | **0.7278** (paper: 72.8) |
| MultiTQ | first listed answer contains gold (Hit@1, lenient) | 0.6495 |
| MultiTQ | first listed answer equals gold (Hit@1, exact) | **0.6457** |
| TimeQuestions (n=3,237) | any-match over the list (repo `eval_hit`) | **0.7800** (paper: 78.1) |
| TimeQuestions | Hit@1 lenient | 0.7275 |
| TimeQuestions | Hit@1 exact | **0.6923** |

MultiTQ by question type:

| rule | after_first | before_after | before_last | equal | equal_multi | first_last |
|---|---|---|---|---|---|---|
| any-match over list | 0.3359 | 0.8540 | 0.2989 | 0.8995 | 0.4047 | 0.8996 |
| Hit@1 lenient | 0.3342 | 0.5961 | 0.2965 | 0.8329 | 0.3308 | 0.8996 |
| Hit@1 exact | 0.3318 | 0.5848 | 0.2941 | 0.8307 | 0.3271 | 0.8993 |

## Findings

1. The reported 72.8 (MultiTQ) and 78.1 (TimeQuestions) are reproduced exactly by TimeR4's released predictions under
   the repository's `eval_hit`: a question counts as correct if **any** gold answer string occurs **anywhere** in the
   whole generated answer list (`' '.join(prediction)`, no top-1 cut). This is a list-level hit rate, not Hits@1.
2. Under Hits@1 (first listed answer), TimeR4 is 64.6 (exact) / 65.0 (lenient) on MultiTQ and 69.2 / 72.8 on
   TimeQuestions. The gap is concentrated in multi-answer question types: before_after 85.4 -> 58.5, equal 90.0 -> 83.1,
   equal_multi 40.5 -> 32.7.
3. The released `detailed_eval_result.jsonl` files carry `hit = 1` for all 54,584 MultiTQ questions (mean hit 1.0000),
   including wrong answers (e.g. prediction 'Barack Obama', gold 'Jack Straw'). This is reproduced exactly by iterating
   over the characters of the unparsed ground-truth string (`for a in "['Jack Straw']"`), i.e. a bug in an earlier
   scorer version; the `acc`/`recall` fields in those files are not meaningful either. The paper numbers do not come
   from that field (they match the list-level rule above).
4. The original TEAR manuscript's reproduction of TimeR4 (62.6 on MultiTQ, own fine-tuned reader) is close to TimeR4's
   own strict Hits@1 (64.6), i.e. the earlier reproduction was not far off; the 10-point gap to 72.8 is mostly the
   metric.
5. Later papers copy 72.8 into tables whose own methods are scored with exact-match Hits@1 (e.g. AT2QA "Hits@1 (Exact
   Match)", PoK format-strict). Comparisons against TimeR4's 72.8 therefore mix two metrics.

Reproduce: score `external/TimeR4_results/unzipped/results/MultiTQ/finetuned_llama2/predictions.jsonl` (and the
TimeQuestions file) with `tear.evaluation.hits.score_reader_record` (`tear.text.match` / `exact_match`).
