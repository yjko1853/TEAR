#!/bin/bash
# Reader for the TR-promote variant (results/prereg_tr_promote.md): Qwen2.5-7B-Instruct-4bit, prompt v3, k = 15.
#   MultiTQ  tear_window_promote on the 3,000 sample questions with a first/last cue; tr_promote on the 1,000 first_last
#   ICEWS18  tear_window_promote on the 354 first_last questions among the first 2,000 of the reader order
# Before each run, predictions are copied from the settings already read whose top-15 list is identical (scripts/14,
# greedy decoding; sources in results/qwen_ground / results/qwen_icews18). Resumable.
# Usage: caffeinate -i bash logs/run_tr_promote_reader.sh
cd "$(dirname "$0")/.." || exit 1
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2
TAG=rule_v2_ground_exact_anydate
PY=".venv-mlx/bin/python -I"
PYC=".venv/bin/python -I"
LOG=logs/tr_promote_reader.log

read_set() {  # dataset pred_dir source_dir sources sample setting
  $PYC scripts/14_reuse_across_settings.py --dataset "$1" --tag $TAG --pred_dir "$2" --source_dir "$3" --target "$6" \
    --sources "$4" --sample "$5" --prompt v3 | tee -a $LOG
  echo "[$(date '+%F %T')] $1 $5 $6" | tee -a $LOG
  $PY scripts/07_read_with_llm.py --dataset "$1" --setting "$6" --tag $TAG --questions questions.json \
    --sample "$5" --prompt v3 --k 15 --batch 1 --pred_dir "$2" \
    2>&1 | grep -v -i warning | tail -1 | tee -a $LOG
}

if pgrep -f "07_read_with_llm.py" >/dev/null; then echo "another reader is running; not starting" | tee -a $LOG; exit 2; fi
mkdir -p results/qwen_ground_promote results/qwen_icews18_promote
read_set multitq_orig results/qwen_ground_promote results/qwen_ground tear_window,tear,tr,hardwindow,none,atc,timer4 \
  sample_firstlast_cue_3000.json tear_window_promote
read_set multitq_orig results/qwen_ground_promote results/qwen_ground tr,tear,tear_window,none,hardwindow,atc,timer4 \
  sample_first_last_1000.json tr_promote
read_set icews18 results/qwen_icews18_promote results/qwen_icews18 tear_window,tear,hardwindow,none,timer4 \
  sample_reader2000_first_last.json tear_window_promote
echo "[$(date '+%F %T')] all done" | tee -a $LOG
