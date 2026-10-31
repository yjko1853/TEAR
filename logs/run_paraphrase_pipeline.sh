#!/bin/bash
# Paraphrase robustness of anchor grounding (pre-registered: results/prereg_paraphrase_robustness.md).
# After scripts/21_paraphrase_questions.py --generate: build cache/multitq_para and run the main-setting pipeline on it
# with unchanged code and settings, then compare with the original wording (scripts/22_paraphrase_report.py).
#   caffeinate -i bash logs/run_paraphrase_pipeline.sh > logs/paraphrase_pipeline.log 2>&1
cd "$(dirname "$0")/.." || exit 1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 NUMEXPR_NUM_THREADS=2 HF_HUB_OFFLINE=1
P=".venv/bin/python -I"
DS=multitq_para
D=cache/$DS
COMMON="--date_granularity any --date_parser v2 --atc_match exact --slots $D/slots_rule_v2.json"
step() { echo "[$(date '+%H:%M:%S')] $1"; }
step "build $D";            $P scripts/21_paraphrase_questions.py --build || exit 1
step "search";              $P scripts/03_search.py --dataset $DS --device cpu --threads 2 || exit 1
step "NER topic fallback";  $P scripts/04_label_slots.py --dataset $DS --method ner --labels all --device cpu --verify_sample 0 || exit 1
step "rule v2 labels";      $P scripts/04_label_slots.py --dataset $DS --method rule --rule_version v2 --out $D/slots_rule_v2.json || exit 1
step "anchor grounding";    $P scripts/14_ground_anchors.py --dataset $DS || exit 1
step "rerank, grounding";   $P scripts/05_rerank.py --dataset $DS $COMMON --anchors $D/anchors.json --equal_window --tag rule_v2_ground_exact || exit 1
$P scripts/06_llmfree_eval.py --dataset $DS --tag rule_v2_ground_exact_anydate --slots $D/slots_rule_v2.json --baseline none || exit 1
step "rerank, no grounding"; $P scripts/05_rerank.py --dataset $DS $COMMON --tag rule_v2_exact || exit 1
$P scripts/06_llmfree_eval.py --dataset $DS --tag rule_v2_exact_anydate --slots $D/slots_rule_v2.json --baseline none || exit 1
step "meaning-check sheet"; $P scripts/21_paraphrase_questions.py --meaning || exit 1
step "report";              $P scripts/22_paraphrase_report.py || exit 1
step "done"
