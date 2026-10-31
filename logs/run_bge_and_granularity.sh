#!/bin/bash
# Pre-registered (results/prereg_retriever_and_granularity.md):
#   A. the main setting with a second retriever (BAAI/bge-base-en-v1.5), LLM-free, full MultiTQ test set
#   B. answer granularity: prompt v3g for the predicted date answers of the 6,000-question reader sample
# Run sequentially (both use the GPU):  caffeinate -i bash logs/run_bge_and_granularity.sh > logs/bge_granularity.log 2>&1
cd "$(dirname "$0")/.." || exit 1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 NUMEXPR_NUM_THREADS=2 HF_HUB_OFFLINE=1
P=".venv/bin/python -I"; M=".venv-mlx/bin/python -I"
ENC=BAAI/bge-base-en-v1.5; DS=multitq_bge; D=cache/$DS
COMMON="--date_granularity any --date_parser v2 --atc_match exact --slots $D/slots_rule_v2.json"
step() { echo "[$(date '+%H:%M:%S')] $1"; }
if [ "${SKIP_A:-0}" != 1 ]; then
  step "A1 variant";          $P scripts/01_prepare_data.py --dataset $DS || exit 1
  step "A2 encode facts";     $P scripts/02_encode_facts.py --dataset $DS --model $ENC --device mps || exit 1
  step "A3 search";           $P scripts/03_search.py --dataset $DS --model $ENC --device mps --threads 2 || exit 1
  step "A4 grounding";        $P scripts/14_ground_anchors.py --dataset $DS || exit 1
  step "A5 rerank";           $P scripts/05_rerank.py --dataset $DS $COMMON --anchors $D/anchors.json --equal_window --tag rule_v2_ground_exact || exit 1
  step "A6 LLM-free";         $P scripts/06_llmfree_eval.py --dataset $DS --tag rule_v2_ground_exact_anydate --slots $D/slots_rule_v2.json --baseline none || exit 1
  step "A7 report";           $P scripts/23_retriever_report.py || exit 1
fi
for s in none tear_window; do
  step "B reader v3g $s"
  $M scripts/07_read_with_llm.py --dataset multitq_orig --setting $s --tag rule_v2_ground_exact_anydate \
     --sample sample_6000_dateslot.json --prompt v3g --k 15 --pred_dir results/qwen_ground || exit 1
done
step "B report";              $P scripts/24_granularity_report.py || exit 1
step "done"
