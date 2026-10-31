#!/bin/bash
# Round-robin driver, main setting: original questions + anchor grounding (dataset multitq_orig, tag
# rule_v2_ground_exact_anydate). The reader prompt shows the original question (no rewrite, no grounded date inserted).
# Every setting on questions [lo, lo+CHUNK) of the fixed shuffled 6,000-question sample (same order as the v1/v2 runs),
# then the next chunk. Resumable (skips quids already in results/qwen_ground/).
# Needs patches/reader_dataset.patch (scripts/07 --dataset). Usage: bash logs/run_qwen_chunks_ground.sh <prompt>
cd "$(dirname "$0")/.." || exit 1
export HF_HUB_OFFLINE=1
PROMPT=${1:-v3}
CHUNK=${CHUNK:-1000}
SETTINGS=${SETTINGS:-"none tear_window timer4 timer4_public tear atc tr"}
TAG=rule_v2_ground_exact_anydate
PY=".venv-mlx/bin/python -I"
grep -q -- '--dataset' scripts/07_read_with_llm.py || { echo "apply patches/reader_dataset.patch first"; exit 1; }
mkdir -p results/qwen_ground_scores
for lo in $(seq 0 $CHUNK 5999); do
  hi=$((lo + CHUNK))
  for s in $SETTINGS; do
    echo "[$(date '+%F %T')] chunk $lo-$hi setting $s prompt $PROMPT"
    $PY scripts/07_read_with_llm.py --dataset multitq_orig --setting "$s" --tag $TAG --questions questions.json \
      --sample sample_6000.json --prompt "$PROMPT" --k 15 --batch 1 --start $lo --end $hi --pred_dir results/qwen_ground \
      2>&1 | grep -v -i warning | tail -1
  done
  $PY scripts/08_score_reader.py --pred_dir results/qwen_ground --results_dir results/qwen_ground_scores \
    --baseline none_$TAG > logs/qwen_ground_score_latest.log 2>&1
  echo "[$(date '+%F %T')] chunk $lo-$hi done; see results/qwen_ground_scores/"
done
echo "[$(date '+%F %T')] all chunks done"
