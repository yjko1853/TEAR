#!/bin/bash
# Round-robin driver, rule v2 + query fallback + date parser v2 (tag rule_v2_exact_anydate).
# Every setting on questions [lo, lo+CHUNK) of the fixed shuffled 6,000-question sample, then the next chunk.
# Resumable (skips quids already in results/qwen_v2/). Usage: bash logs/run_qwen_chunks_v2.sh <prompt>
cd "$(dirname "$0")/.." || exit 1
export HF_HUB_OFFLINE=1
PROMPT=${1:-v3}
CHUNK=${CHUNK:-1000}
SETTINGS=${SETTINGS:-"none tear_window timer4 timer4_public tear atc tr"}
PY=".venv-mlx/bin/python -I"
mkdir -p results/qwen_v2_scores
for lo in $(seq 0 $CHUNK 5999); do
  hi=$((lo + CHUNK))
  for s in $SETTINGS; do
    echo "[$(date '+%F %T')] chunk $lo-$hi setting $s prompt $PROMPT"
    $PY scripts/07_read_with_llm.py --setting "$s" --tag rule_v2_exact_anydate --questions questions_v2.json \
      --sample sample_6000.json --prompt "$PROMPT" --batch 1 --start $lo --end $hi --pred_dir results/qwen_v2 \
      2>&1 | grep -v -i warning | tail -1
  done
  $PY scripts/08_score_reader.py --pred_dir results/qwen_v2 --results_dir results/qwen_v2_scores \
    --baseline none_rule_v2_exact_anydate > logs/qwen_v2_score_latest.log 2>&1
  echo "[$(date '+%F %T')] chunk $lo-$hi done; see results/qwen_v2_scores/"
done
echo "[$(date '+%F %T')] all chunks done"
