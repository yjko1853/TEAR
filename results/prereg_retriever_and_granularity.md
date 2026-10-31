# Pre-registration: second retriever and answer granularity (MultiTQ)

Registered 2026-10-20, before the second encoder was downloaded and before any v3g reader output existed. Both
experiments are reported in the paper whatever they show (user decision, 2026-10-20). Later changes are listed as
deviations with their reason.

## A. Second retriever

Question: do the gains of the rules over the retriever order hold when the retriever changes?

- Retriever: BAAI/bge-base-en-v1.5 through sentence-transformers, no query instruction, L2-normalised embeddings,
  exact inner-product top-1000 (FAISS). Facts are the same 461,329 sentences.
- Everything else is the main setting, unchanged: original questions, rule-v2 labels and NER topics (the same files),
  anchor grounding (scripts/14; the all-mpnet-base-v2 name encoder and tau = 0.5 of the design; the reference fact is
  chosen by the new retriever's score), reranking with --date_granularity any --date_parser v2 --atc_match exact
  --anchors --equal_window, lambda1 = lambda2 = 0.8. Nothing is tuned for the new retriever.
- Data: cache/multitq_bge (scripts/01 --dataset multitq_bge), full test set (54,584 questions).
- Measures: coverage@15 and LLM-free Hit@1 of every reranker, McNemar and paired bootstrap against no reranking
  (scripts/06), shown next to the all-mpnet-base-v2 numbers. No reader run.

## B. Answer granularity (answer type = slot + granularity)

Question: the zero-shot reader often answers a date question at the wrong granularity ("2010-07-13" for "In which
year ..."). Does stating the granularity that the rules predict help, and how does it compare with projecting the
answer onto that granularity?

- Prediction (no LLM, no gold label): the rule-v2 answer slot says whether the answer is a date; the unit is year if
  the question says year, month if it says month, otherwise day (tear/labeling/granularity.py). Its agreement with
  the gold answers is measured on all 15,884 test time questions.
- Reader: Qwen2.5-7B-Instruct-4bit, k = 15, greedy, max 48 tokens, on the 6,000-question reader sample and the
  rerankers none and tear_window (tag rule_v2_ground_exact_anydate).
- Prompt v3g = v3 plus one line for predicted date answers: "This question asks for a <unit>: answer with the <unit>
  only, written as <format>." Questions without a predicted date get exactly the v3 prompt, so their v3 outputs are
  reused (greedy decoding gives the same output for the same input). The line is fixed here and not tuned.
- Variants, exact Hit@1: (A) v3 as read; (B) v3 + projection onto the predicted unit; (C) v3g as read; (D) v3g +
  projection. (B) uses the rule's prediction, not the gold answer type; the paper's current "reader+date" uses the
  gold answer type to choose the questions to cut, which changes 2 to 3 of the 6,000 answers (checked before this
  registration on the existing outputs).
- Measures: on the gold time questions and on all 6,000; the share of time questions answered at the wrong
  granularity only (right after projection, wrong before); paired McNemar and bootstrap for C vs A and D vs B.
- Reporting: all four variants are reported. If C is not better than A, the paper says that stating the granularity
  does not help a 7B reader and that the deterministic projection is what closes the gap.
