#!/bin/bash
# Reader with the label-free hard window + anchor grounding (results/prereg_hardwindow_reader.md):
# MultiTQ 6,000-question sample (6 chunks) and the first 2,000 ICEWS18 questions, Qwen2.5-7B-Instruct-4bit, prompt v3,
# k = 15. Before each slice, predictions are copied from the settings already read where the top-15 list is identical
# (scripts/14; greedy decoding). Resumable. Usage: caffeinate -i bash logs/run_hardwindow_reader.sh
cd "$(dirname "$0")/.." || exit 1
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2
TAG=rule_v2_ground_exact_anydate
PY=".venv-mlx/bin/python -I"
PYC=".venv/bin/python -I"
LOG=logs/hardwindow_reader.log

read_slice() {  # dataset pred_dir sources lo hi
  $PYC scripts/14_reuse_across_settings.py --dataset "$1" --tag $TAG --pred_dir "$2" --target hardwindow \
    --sources "$3" --start "$4" --end "$5" --prompt v3 | tee -a $LOG
  echo "[$(date '+%F %T')] $1 $4-$5 hardwindow" | tee -a $LOG
  $PY scripts/07_read_with_llm.py --dataset "$1" --setting hardwindow --tag $TAG --questions questions.json \
    --sample sample_6000.json --prompt v3 --k 15 --batch 1 --start "$4" --end "$5" --pred_dir "$2" \
    2>&1 | grep -v -i warning | tail -1 | tee -a $LOG
}

if pgrep -f "07_read_with_llm.py" >/dev/null; then echo "another reader is running; not starting" | tee -a $LOG; exit 2; fi
for lo in $(seq 0 1000 5999); do
  read_slice multitq_orig results/qwen_ground none,tear_window,timer4,tear,atc,tr $lo $((lo + 1000))
done
read_slice icews18 results/qwen_icews18 none,timer4,tear,tear_window 0 2000
echo "[$(date '+%F %T')] all done" | tee -a $LOG
