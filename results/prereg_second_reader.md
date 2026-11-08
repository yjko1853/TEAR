# Pre-registration: a second, larger local reader (Qwen2.5-14B-Instruct-4bit)

Registered 2026-10-31 09:55, before any output of this reader existed. The result is reported in the paper whatever
it shows (supplementary material, with one sentence in the main text where the reader is introduced and in the
limitations). Later changes are listed as deviations with their reason.

## Question

Every reader number in the paper comes from one 7B model (Qwen2.5-7B-Instruct, 4-bit, MLX). Does the reader gain of
TEAR+window over no reranking, and the share of it that the hard window with grounding gives, hold with a larger
reader of the same family (Qwen2.5-14B-Instruct, 4-bit, mlx-community, 8.3 GB)? A larger reader may need the
reranking less (it can find the right fact among the 15 itself) or more (it follows the list order more faithfully).

## Setting (nothing new is designed or tuned)

- Model: `mlx-community/Qwen2.5-14B-Instruct-4bit`, greedy decoding, max 48 tokens, prompt v3 unchanged (frozen on
  the 7B tuning set; not re-tuned for 14B), k = 15, the same code (scripts/07_read_with_llm.py) and the same shuffled
  order (seed 2026). Chosen as the largest instruction-tuned model of the same family that fits the 16 GB machine.
- Data: MultiTQ 6,000-question sample, first 1,000 questions of the reader order (paired, roughly stratified: 187
  first_last, 176 equal, 166 before_last, 161 before_after, 160 after_first, 150 equal_multi).
- Settings, in this order: `none`, `tear_window`, then `hardwindow` if the first two finish (each about 1,000
  questions; the time per question is measured on a 30-question speed check first). Lossless reuse across settings
  where the top-15 lists are identical (scripts/14). Predictions: `results/qwen14b_ground/`.
- If the 14B model does not run on this machine (memory) or exceeds 25 s per question, the fallback is
  `mlx-community/Meta-Llama-3.1-8B-Instruct-4bit` (a different family of the same size) with the same protocol;
  the change is recorded here as a deviation.

## Measures

- Exact Hit@1 (first answer), reader+date, per question type and entity/time; paired McNemar and bootstrap
  (10,000 draws, seed 0) for tear_window vs none and, if read, hardwindow vs none and tear_window vs hardwindow;
  the same three comparisons for the 7B reader on the same 1,000 questions, side by side.

## What is written whatever the result

- A supplementary table with the 7B and 14B readers on the same 1,000 questions, and the sentence in the main text
  that names the second reader. If the TEAR+window gain holds (interval above 0), the limitation "one reader" is
  softened to "two readers of one family". If it shrinks or vanishes, the paper says the gain depends on the reader's
  size and reports it as a limitation of the plug-in. The decomposition (window share vs ATC+TR share) is reported
  for the 14B reader only if `hardwindow` is read.
DEVIATION 2026-10-31 18:46:44: fallback to mlx-community/Meta-Llama-3.1-8B-Instruct-4bit (14B bench: finished 30 questions in 1274s (42.46 s/question) -> results/qwen14b_ground/none_rule_v2_ground_exact_anydate__Qwen2.5-14B-Instruct-4bit__k15__p-v3__bench.jsonl)

## Outcome of the registered protocol and a registered deviation (written 2026-11-07 23:55, before any run with the change below)

- Runs: `none` 10-31 19:05 .. 11-03 22:48 (272 s/question of wall time: the laptop slept most of the time), `tear_window`
  11-03 22:48 .. 11-07 22:44 with a reboot on 11-05 19:40 and a resume on 11-07 22:14 (`logs/run_second_reader_resume.sh`,
  no bench repeated), `hardwindow` 11-07 22:44 .. 23:41; 6.6 s/question while awake. Predictions `results/qwen_llama8b_ground/`.
- Result under the registered protocol (prompt v3 frozen for Qwen, 48 tokens, `scripts/29_second_reader_report.py
  --pred_dir results/qwen_llama8b_ground --model Meta-Llama-3.1-8B-Instruct-4bit`): exact Hit@1 0.085 / 0.092 / 0.078 and
  reader+date 0.102 / 0.109 / 0.095 for none / TEAR+window / hard window on the 1,000 questions (7B: 0.320 / 0.388 / 0.360);
  no difference between settings is significant. The cause is format, not reading: the Llama reader reasons in prose
  before answering ("To find ..., we need to look ...") and the 48-token budget cuts it off before any answer list in most
  questions (see the counts in results/second_reader_format.txt), so the parser reads sentence fragments as answers. The
  7B reader starts its output with the list, so the budget never affects its first answer.
- Deviation registered now: the same model, prompt, decoding, questions, settings and order, with the token budget raised
  from 48 to 256 so that the answer list the Llama reader writes after its reasoning fits (`--max_tokens 256`, output
  directory `results/qwen_llama8b_ground_t256/`, driver `logs/run_second_reader_t256.sh`). Nothing else changes; the prompt
  is not edited. A 30-question check decides only whether the run is feasible on this machine (under 20 s/question); its
  outputs are not read for anything else. Both outcomes are reported: the registered 48-token result as the result of the
  registered protocol, and the 256-token run as a deviation marked as such, in the supplementary table and paragraph. The
  decision rule of "What is written whatever the result" is applied to the 256-token run if it is completed, and the main
  text says that the second reader needed a larger budget.

## Outcome of the registered deviation (256 tokens; written 2026-11-08 09:20 after the run)

- Run: `logs/run_second_reader_t256.sh` 2026-11-07 23:55 .. 2026-11-08 08:54 (laptop awake throughout): `none` 1,000 in
  12,726 s (12.7 s/question), `tear_window` 909 read + 91 reused in 12,229 s (13.5), `hardwindow` 514 read + 486 reused in
  7,370 s (14.3); 13.3 s/question over the 2,423 questions read. Predictions `results/qwen_llama8b_ground_t256/`; report
  `scripts/29_second_reader_report.py --pred_dir results/qwen_llama8b_ground_t256 --model Meta-Llama-3.1-8B-Instruct-4bit
  --label 8B-t256 --out_name second_reader_t256` -> `results/second_reader_t256{.md,_summary.csv,_significance.csv}`;
  format counts appended to `results/second_reader_format.txt`.
- Format: a bracketed list occurs in 599 / 573 / 606 of 1,000 outputs for none / tear_window / hardwindow (0 / 1 / 0 start
  with it; the 7B reader starts with the list on 1,000 / 1,000 in every setting). The other outputs still end inside the
  reasoning and count as misses; the prompt was not changed.
- Result (reader+date, exact first answer, n = 1,000): none 0.230, hardwindow 0.274, tear_window 0.279 (7B on the same
  questions: 0.320 / 0.360 / 0.388; exact without the date cut: 0.210 / 0.254 / 0.259). Paired differences: tear_window vs
  none +0.049 [0.019, 0.078], McNemar p = 0.0012, 135 gains / 86 losses; hardwindow vs none +0.044 [0.025, 0.064],
  p < 0.001, 71 / 27; tear_window vs hardwindow +0.005 [-0.025, 0.034], p = 0.79, 116 / 111. For the 7B reader on the same
  1,000: +0.068 [0.041, 0.095]; +0.040 [0.024, 0.056]; +0.028 [0.001, 0.054], p = 0.044. The second reader is below the 7B
  reader in every setting (-0.090 / -0.086 / -0.109, all p < 0.001). LLM-free on the same lists: 0.229 / 0.261 / 0.434.
- Decision rule: the TEAR+window gain holds (interval above 0), so the limitation "one reader" is softened to "two readers
  of different families" (the registered wording said "of one family" because the registered model was Qwen 14B; the
  fallback is a Llama model). The decomposition, reported because `hardwindow` was read, differs from the 7B reader's: the
  gain of the window alone holds (+0.044), the gain of ATC and TR within the window does not (+0.005, p = 0.79, against
  +0.028 for the 7B reader). The paper reports this as a limitation of ATC and TR (main text, Limitations "Readers and
  retrievers") rather than claiming the within-window gain for the second reader.
- Written into the manuscript 2026-11-08: supplementary Section S10 paragraph "A second reader" + Table S17 (Qwen 7B /
  Llama 8B 48 tokens / Llama 8B 256 tokens x None / Hard window / TEAR+window, with the list share), main text Section 4
  (reader paragraph, one sentence naming the second reader and the larger budget), Limitations (second reader and the
  reader-dependence of the within-window gain), supplementary "The reader is a bottleneck" and "Scope of the evidence",
  Introduction ("one main reader"). The 48-token result stays in the table as the result of the registered protocol.
