#!/bin/bash
# Registered deviation of results/prereg_second_reader.md (2026-11-07 23:55): the fallback second reader
# (Meta-Llama-3.1-8B-Instruct-4bit) with the token budget raised from 48 to 256, everything else as registered
# (prompt v3, greedy, k = 15, first 1,000 questions of the reader order, settings none -> tear_window -> hardwindow,
# lossless reuse where the top-15 lists are identical). Predictions: results/qwen_llama8b_ground_t256/.
# Usage: caffeinate -dims bash logs/run_second_reader_t256.sh      (resumable: scripts/07 skips finished questions)
cd "$(dirname "$0")/.." || exit 1
export OMP_NUM_THREADS=2 HF_HUB_OFFLINE=1
TAG=rule_v2_ground_exact_anydate
PY=".venv-mlx/bin/python -I"
PYC=".venv/bin/python -I"
LOG=logs/second_reader_t256.log
PRED=results/qwen_llama8b_ground_t256
MODEL=mlx-community/Meta-Llama-3.1-8B-Instruct-4bit
MNAME=$(basename $MODEL)
log() { echo "[$(date '+%F %T')] $1" | tee -a $LOG; }
mkdir -p $PRED
log "start; model $MODEL; max_tokens 256"
read_set() {  # setting sources
  [ -n "$2" ] && $PYC scripts/14_reuse_across_settings.py --dataset multitq_orig --tag $TAG --pred_dir $PRED --target "$1" \
    --sources "$2" --start 0 --end 1000 --prompt v3 --model $MNAME | tee -a $LOG
  log "multitq_orig 0-1000 $1 ($MNAME, 256 tokens)"
  $PY scripts/07_read_with_llm.py --dataset multitq_orig --setting "$1" --tag $TAG --questions questions.json \
    --sample sample_6000.json --prompt v3 --k 15 --batch 1 --start 0 --end 1000 --max_tokens 256 --pred_dir $PRED \
    --model $MODEL 2>&1 | grep -v -i warning | tail -1 | tee -a $LOG
}
read_set none ""
read_set tear_window none
read_set hardwindow none,tear_window
log "all done"
