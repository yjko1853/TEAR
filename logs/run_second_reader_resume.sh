#!/bin/bash
# Resume of logs/run_second_reader.sh after the reboot of 2026-11-05 19:40 (the Mac also slept most of 10-31..11-05, so the
# first setting took 272 s/question of wall time). Same registered fallback model (Meta-Llama-3.1-8B-Instruct-4bit, see the
# DEVIATION line in results/prereg_second_reader.md), same prompt v3, k = 15, greedy, first 1,000 questions of the reader
# order: finishes tear_window (727/1000 done; scripts/07 skips questions already in the output file) and runs hardwindow,
# reusing none / tear_window predictions where the lists are identical. No bench is repeated.
# Usage: caffeinate -dims bash logs/run_second_reader_resume.sh
cd "$(dirname "$0")/.." || exit 1
export OMP_NUM_THREADS=2 HF_HUB_OFFLINE=1
TAG=rule_v2_ground_exact_anydate
PY=".venv-mlx/bin/python -I"
PYC=".venv/bin/python -I"
LOG=logs/second_reader.log
PRED=results/qwen_llama8b_ground
MODEL=mlx-community/Meta-Llama-3.1-8B-Instruct-4bit
MNAME=$(basename $MODEL)
log() { echo "[$(date '+%F %T')] $1" | tee -a $LOG; }
log "resume after the reboot of 2026-11-05; model $MODEL"
read_set() {  # setting sources
  [ -n "$2" ] && $PYC scripts/14_reuse_across_settings.py --dataset multitq_orig --tag $TAG --pred_dir $PRED --target "$1" \
    --sources "$2" --start 0 --end 1000 --prompt v3 --model $MNAME | tee -a $LOG
  log "multitq_orig 0-1000 $1 ($MNAME)"
  $PY scripts/07_read_with_llm.py --dataset multitq_orig --setting "$1" --tag $TAG --questions questions.json \
    --sample sample_6000.json --prompt v3 --k 15 --batch 1 --start 0 --end 1000 --pred_dir $PRED --model $MODEL \
    2>&1 | grep -v -i warning | tail -1 | tee -a $LOG
}
read_set tear_window none
read_set hardwindow none,tear_window
log "all done"
