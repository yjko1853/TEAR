#!/bin/bash
# No-rewrite ablation with the v2 configuration (rule v2 + date parser v2) on the original questions (multitq_orig).
# Question encoding, search and NER topics are unchanged by v2 and reused from the v1 ablation run.
cd "$(dirname "$0")/.." || exit 1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 NUMEXPR_NUM_THREADS=2 HF_HUB_OFFLINE=1
P=".venv/bin/python -I"
step() { echo "[$(date '+%H:%M:%S')] $1"; }
step "4 multitq_orig slots_rule_v2.json"
$P scripts/04_label_slots.py --dataset multitq_orig --method rule --rule_version v2 \
   --out cache/multitq_orig/slots_rule_v2.json || exit 1
step "5 rerank multitq_orig rule_v2_exact_anydate (sweep)"
$P scripts/05_rerank.py --dataset multitq_orig --slots cache/multitq_orig/slots_rule_v2.json \
   --atc_match exact --date_granularity any --date_parser v2 --sweep --tag rule_v2_exact || exit 1
step "6 LLM-free Hit@1 multitq_orig v2"
$P scripts/06_llmfree_eval.py --dataset multitq_orig --tag rule_v2_exact_anydate \
   --slots cache/multitq_orig/slots_rule_v2.json --baseline none || exit 1
step "11 compare with the v2 rewrite run"
$P scripts/11_compare_rewrite.py --tag rule_v2_exact_anydate --base_questions questions_v2.json \
   --base_slots slots_rule_v2.json --variant_slots slots_rule_v2.json --base_top top1000_v2_ids.npy || exit 1
step "done"
