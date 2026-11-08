#!/bin/bash
# Second, larger reader (results/prereg_second_reader.md): mlx-community/Qwen2.5-14B-Instruct-4bit on the first 1,000
# questions of the MultiTQ reader order, settings none, tear_window, hardwindow (in this order), prompt v3, k = 15.
# Waits until no other reader runs and the model download has finished; a 30-question speed check decides between the
# 14B model and the fallback (Meta-Llama-3.1-8B-Instruct-4bit) when the 14B exceeds 25 s/question or fails.
# Usage: caffeinate -i bash logs/run_second_reader.sh
cd "$(dirname "$0")/.." || exit 1
export OMP_NUM_THREADS=2
TAG=rule_v2_ground_exact_anydate
PY=".venv-mlx/bin/python -I"
PYC=".venv/bin/python -I"
LOG=logs/second_reader.log
PRED=results/qwen14b_ground
MODEL=mlx-community/Qwen2.5-14B-Instruct-4bit
FALLBACK=mlx-community/Meta-Llama-3.1-8B-Instruct-4bit
log() { echo "[$(date '+%F %T')] $1" | tee -a $LOG; }

while pgrep -f "07_read_with_llm.py" >/dev/null || pgrep -f "run_tr_promote_reader.sh" >/dev/null; do sleep 300; done
while ! grep -q "^DONE" logs/download_qwen14b.log 2>/dev/null; do sleep 120; done
log "start; model $MODEL"
mkdir -p $PRED
export HF_HUB_OFFLINE=1
bench=$($PY scripts/07_read_with_llm.py --dataset multitq_orig --setting none --tag $TAG --questions questions.json \
  --sample sample_6000.json --prompt v3 --k 15 --batch 1 --limit 30 --pred_dir $PRED --model $MODEL 2>&1 | grep -v -i warning | tail -1)
log "bench: $bench"
spq=$(echo "$bench" | sed -n 's/.*(\([0-9.]*\) s\/question).*/\1/p')
if [ -z "$spq" ] || [ "$(echo "$spq > 25" | bc -l)" = "1" ]; then
  log "14B too slow or failed (s/question='$spq'); falling back to $FALLBACK"
  unset HF_HUB_OFFLINE
  $PY -c "from huggingface_hub import snapshot_download; print(snapshot_download('$FALLBACK'))" 2>&1 | tail -1 | tee -a $LOG
  export HF_HUB_OFFLINE=1
  MODEL=$FALLBACK; PRED=results/qwen_llama8b_ground; mkdir -p $PRED
  echo "DEVIATION $(date '+%F %T'): fallback to $FALLBACK (14B bench: $bench)" >> results/prereg_second_reader.md
  bench=$($PY scripts/07_read_with_llm.py --dataset multitq_orig --setting none --tag $TAG --questions questions.json \
    --sample sample_6000.json --prompt v3 --k 15 --batch 1 --limit 30 --pred_dir $PRED --model $MODEL 2>&1 | grep -v -i warning | tail -1)
  log "fallback bench: $bench"
fi
MNAME=$(basename $MODEL)
read_set() {  # setting sources
  [ -n "$2" ] && $PYC scripts/14_reuse_across_settings.py --dataset multitq_orig --tag $TAG --pred_dir $PRED --target "$1" \
    --sources "$2" --start 0 --end 1000 --prompt v3 --model $MNAME | tee -a $LOG
  log "multitq_orig 0-1000 $1 ($MNAME)"
  $PY scripts/07_read_with_llm.py --dataset multitq_orig --setting "$1" --tag $TAG --questions questions.json \
    --sample sample_6000.json --prompt v3 --k 15 --batch 1 --start 0 --end 1000 --pred_dir $PRED --model $MODEL \
    2>&1 | grep -v -i warning | tail -1 | tee -a $LOG
}
read_set none ""
read_set tear_window none
read_set hardwindow none,tear_window
log "all done"
