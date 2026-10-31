#!/bin/bash
# Grounded original-question run with cross-setting reuse (lossless): before a setting runs on a chunk, predictions of
# settings already run in that chunk are copied where the top-15 fact list is identical (scripts/14; greedy decoding).
# Resumable. Usage: bash logs/run_qwen_chunks_ground_dedup.sh v3
cd "$(dirname "$0")/.." || exit 1
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2
PROMPT=${1:-v3}
CHUNK=${CHUNK:-1000}
SETTINGS=${SETTINGS:-"none tear_window timer4 timer4_public tear atc tr"}
TAG=rule_v2_ground_exact_anydate
PY=".venv-mlx/bin/python -I"
PYC=".venv/bin/python -I"
mkdir -p results/qwen_ground_scores
for lo in $(seq 0 $CHUNK 5999); do
  hi=$((lo + CHUNK)); done_settings=""
  for s in $SETTINGS; do
    if [ -n "$done_settings" ] && [ "$s" != "timer4_public" ]; then
      $PYC scripts/14_reuse_across_settings.py --dataset multitq_orig --tag $TAG --pred_dir results/qwen_ground \
        --target "$s" --sources "${done_settings#,}" --start $lo --end $hi --prompt "$PROMPT"
    fi
    echo "[$(date '+%F %T')] chunk $lo-$hi setting $s prompt $PROMPT"
    $PY scripts/07_read_with_llm.py --dataset multitq_orig --setting "$s" --tag $TAG --questions questions.json \
      --sample sample_6000.json --prompt "$PROMPT" --k 15 --batch 1 --start $lo --end $hi --pred_dir results/qwen_ground \
      2>&1 | grep -v -i warning | tail -1
    [ "$s" != "timer4_public" ] && done_settings="$done_settings,$s"
  done
  $PY scripts/08_score_reader.py --pred_dir results/qwen_ground --results_dir results/qwen_ground_scores \
    --baseline none_${TAG} > logs/qwen_ground_score_latest.log 2>&1
  echo "[$(date '+%F %T')] chunk $lo-$hi done; see results/qwen_ground_scores/"
done
echo "[$(date '+%F %T')] all chunks done"
