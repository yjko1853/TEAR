#!/bin/bash
# TimelineKGQA reader experiment: does anchor grounding carry through to the final answer?
# Data: scripts/19_timeline_reader_data.py (cache/{icews,cron}_rd, sample_1000.json: 1,000 medium/complex questions with
# a subject/object answer per dataset; cache/{icews,cron}_rd_tune: 60 validation questions each).
# Labels: rule v2 (question grammar, scripts/16 --make_rule_labels), tag rule_ground (scripts/16 --tag rule_ground).
# Settings: none (similarity order), tear_window (TEAR + window + grounding), hardwindow (hard window + grounding).
# Reader: Qwen2.5-7B-Instruct-4bit (MLX), k = 15, batch 1, greedy; prompt v3, or v3i (interval wording) if it beats v3
# with the 'none' reranker on the tuning sets (needs patches/reader_prompt_v3i.patch; the choice is frozen in
# logs/timeline_prompt_choice.json).
# Order: chunks of 250 questions of the fixed shuffled sample; within a chunk ICEWS then Cron, settings in the order
# above, so every finished chunk is a paired comparison. Before a setting runs, scripts/14 copies predictions whose
# top-15 list is identical in an earlier setting. Scores after each chunk: results/qwen_timeline_<ds>_scores/.
# Resumable. Exits at once (no waiting) if another reader is running.
#   caffeinate -i bash logs/run_timeline_reader.sh > logs/timeline_reader.log 2>&1 &
#   SAMPLE=sample_1000_noleak.json caffeinate -i bash logs/run_timeline_reader.sh ...   # questions without the answer
#       in their text (predictions then go to results/qwen_timeline_<ds>_noleak/)
cd "$(dirname "$0")/.." || exit 1
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2
M=".venv-mlx/bin/python -I"
P=".venv/bin/python -I"
TAG=rule_ground
CHUNK=${CHUNK:-250}
SETTINGS="none tear_window hardwindow"
SAMPLE=${SAMPLE:-sample_1000.json}
SFX=""; [ "$SAMPLE" = "sample_1000.json" ] || SFX="_${SAMPLE#sample_1000_}"; SFX=${SFX%.json}
CHOICE=logs/timeline_prompt_choice.json
step() { echo "[$(date '+%F %T')] $*"; }

if pgrep -f "07_read_with_llm.py" >/dev/null || pgrep -f "run_icews18.sh" >/dev/null || pgrep -f "run_qwen_chunks" >/dev/null; then
  step "another reader is running; not starting"; pgrep -fl "07_read_with_llm.py|run_icews18.sh|run_qwen_chunks"; exit 2
fi
for ds in icews cron; do
  for f in cache/${ds}_rd/$SAMPLE cache/${ds}_rd_tune/tune_60.json cache/${ds}_rd/rerank/tear_window_${TAG}_top30_ids.npy; do
    [ -e "$f" ] || { step "missing $f (run scripts/16 --tag $TAG and scripts/19 first)"; exit 1; }
  done
done
grep -q -- '--dataset' scripts/07_read_with_llm.py || { step "scripts/07 lacks --dataset (patches/reader_dataset.patch)"; exit 1; }

# ---- 1. prompt choice on the validation tuning sets, 'none' reranker only (cannot favour grounding)
if [ -f "$CHOICE" ]; then
  PROMPT=$($P -c "import json; print(json.load(open('$CHOICE'))['prompt'])")
  step "prompt frozen earlier: $PROMPT"
else
  CANDS="v3"; grep -q '"v3i"' tear/reader/prompt.py && CANDS="v3 v3i"
  for ds in icews cron; do
    for p in $CANDS; do
      step "tune $ds prompt $p"
      $M scripts/07_read_with_llm.py --dataset ${ds}_rd_tune --setting none --tag $TAG --questions questions.json \
        --sample tune_60.json --prompt $p --k 15 --batch 1 --pred_dir results/qwen_timeline_${ds} 2>&1 | grep -v -i warning | tail -1
    done
    $M scripts/08_score_reader.py --pred_dir results/qwen_timeline_${ds} --results_dir results/qwen_timeline_${ds}_scores \
      --tune > logs/qwen_timeline_${ds}_tune_score.log 2>&1
  done
  PROMPT=$($P - "$CHOICE" "$CANDS" <<'EOF'
import json, sys
import pandas as pd
out, cands = sys.argv[1], sys.argv[2].split()
df = pd.concat([pd.read_csv(f"results/qwen_timeline_{ds}_scores/multitq_qwen_per_question_tune.csv").assign(dataset=ds)
                for ds in ("icews", "cron")])
df = df[(df.setting == "none_rule_ground") & df.prompt.isin(cands)]
score = {p: (g.hit1_exact.mean(), g.hit1_lenient.mean(), len(g)) for p, g in df.groupby("prompt")}
# rule fixed before the run: v3i only if it is strictly better on exact Hit@1, ties broken by lenient Hit@1, then v3
best = "v3"
if "v3i" in score and (score["v3i"][0], score["v3i"][1]) > (score["v3"][0], score["v3"][1]):
    best = "v3i"
json.dump({"prompt": best, "rule": "exact Hit@1, then lenient, ties -> v3; 'none' reranker; 60+60 validation questions",
           "scores": {p: {"exact": s[0], "lenient": s[1], "n": s[2]} for p, s in score.items()}}, open(out, "w"), indent=1)
print(best)
EOF
)
  step "prompt chosen: $PROMPT ($(cat $CHOICE | tr -d '\n'))"
fi

# ---- 2. main run: 2 x 1,000 questions x {none, tear_window, hardwindow}
for lo in $(seq 0 $CHUNK 999); do
  hi=$((lo + CHUNK))
  for ds in icews cron; do
    PRED=results/qwen_timeline_${ds}${SFX}
    for s in $SETTINGS; do
      case $s in tear_window) SRC=none ;; hardwindow) SRC=tear_window,none ;; *) SRC= ;; esac
      if [ -n "$SRC" ]; then
        $P scripts/14_reuse_across_settings.py --dataset ${ds}_rd --tag $TAG --pred_dir $PRED --target $s --sources $SRC \
          --sample $SAMPLE --start $lo --end $hi --prompt $PROMPT
      fi
      step "$ds chunk $lo-$hi setting $s prompt $PROMPT"
      $M scripts/07_read_with_llm.py --dataset ${ds}_rd --setting $s --tag $TAG --questions questions.json \
        --sample $SAMPLE --prompt $PROMPT --k 15 --batch 1 --start $lo --end $hi --pred_dir $PRED \
        2>&1 | grep -v -i warning | tail -1
    done
    $M scripts/08_score_reader.py --pred_dir $PRED --results_dir ${PRED}_scores --baseline none_$TAG \
      > logs/qwen_timeline_${ds}_score_latest.log 2>&1
    step "$ds chunk $lo-$hi scored: ${PRED}_scores/multitq_qwen_hit1.csv, multitq_qwen_significance.csv"
  done
done
step "all done"
