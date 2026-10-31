#!/bin/bash
# Round-robin driver (release code): every setting on questions [lo, lo+CHUNK) of the fixed shuffled 6,000-question
# sample, then the next chunk, so each finished chunk is a complete paired comparison. Resumable.
# Usage: bash logs/run_qwen_chunks.sh <prompt>     (env: CHUNK=1000, SETTINGS="...")
cd "$(dirname "$0")/.." || exit 1
export HF_HUB_OFFLINE=1
PROMPT=${1:-v2}
CHUNK=${CHUNK:-1000}
SETTINGS=${SETTINGS:-"none tear_window timer4 timer4_public tear atc tr"}
PY=".venv-mlx/bin/python -I"
for lo in $(seq 0 $CHUNK 5999); do
  hi=$((lo + CHUNK))
  for s in $SETTINGS; do
    echo "[$(date '+%F %T')] chunk $lo-$hi setting $s prompt $PROMPT"
    $PY scripts/07_read_with_llm.py --setting "$s" --tag rule_exact_anydate --sample sample_6000.json \
      --prompt "$PROMPT" --batch 1 --start $lo --end $hi 2>&1 | grep -v -i warning | tail -1
  done
  $PY scripts/08_score_reader.py --baseline none_rule_exact_anydate > logs/qwen_score_latest.log 2>&1
  echo "[$(date '+%F %T')] chunk $lo-$hi done; see results/multitq_qwen_hit1.csv and multitq_qwen_significance.csv"
done
echo "[$(date '+%F %T')] all chunks done"
