#!/bin/bash
# Rule v2 label-dependent artefacts only (steps 4-6 of logs/build_v2.sh); search outputs are unaffected by rule changes.
cd "$(dirname "$0")/.." || exit 1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 NUMEXPR_NUM_THREADS=2 HF_HUB_OFFLINE=1
P=".venv/bin/python -I"
step() { echo "[$(date '+%H:%M:%S')] $1"; }
step "4 slots_rule_v2.json"
$P scripts/04_label_slots.py --dataset multitq --method rule --rule_version v2 --questions questions_v2.json \
   --ner_topics cache/multitq/topic_ner_all_v2.json --out cache/multitq/slots_rule_v2.json || exit 1
step "5 rerank rule_v2_exact_anydate (sweep)"
$P scripts/05_rerank.py --dataset multitq --questions questions_v2.json --suffix _v2 --slots cache/multitq/slots_rule_v2.json \
   --atc_match exact --date_granularity any --date_parser v2 --sweep --tag rule_v2_exact || exit 1
step "6 LLM-free Hit@1 v2"
$P scripts/06_llmfree_eval.py --questions questions_v2.json --tag rule_v2_exact_anydate \
   --slots cache/multitq/slots_rule_v2.json --baseline none || exit 1
step "done"
