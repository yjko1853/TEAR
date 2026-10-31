#!/bin/bash
# Anchor grounding (tear/labeling/anchor.py): original questions with event anchors resolved in the KG, compared with
# TimeR4's rewrites. Rule v2 labels, date parser v2, exact ATC. CPU only, 2 threads, one step at a time.
# Rewrite-side runs write their top-k arrays to cache/multitq_orig/rerank_on_rewrite (cache/multitq stays untouched).
cd "$(dirname "$0")/.." || exit 1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 NUMEXPR_NUM_THREADS=2 HF_HUB_OFFLINE=1
P=".venv/bin/python -I"
step() { echo "[$(date '+%H:%M:%S')] $1"; }
COMMON="--date_granularity any --date_parser v2 --atc_match exact"
RW="--dataset multitq --questions questions_v2.json --suffix _v2 --slots cache/multitq/slots_rule_v2.json"
RWDIR=cache/multitq_orig/rerank_on_rewrite
if [ ! -f cache/multitq_orig/anchors.json ]; then
  step "anchors"; $P scripts/14_ground_anchors.py --dataset multitq_orig || exit 1
fi
step "B  original + grounding (anchors + equal window), sweep"
$P scripts/05_rerank.py --dataset multitq_orig --slots cache/multitq_orig/slots_rule_v2.json $COMMON \
   --anchors cache/multitq_orig/anchors.json --equal_window --sweep --tag rule_v2_ground_exact || exit 1
$P scripts/06_llmfree_eval.py --dataset multitq_orig --tag rule_v2_ground_exact_anydate \
   --slots cache/multitq_orig/slots_rule_v2.json --baseline none || exit 1
step "B1 original + anchors only"
$P scripts/05_rerank.py --dataset multitq_orig --slots cache/multitq_orig/slots_rule_v2.json $COMMON \
   --anchors cache/multitq_orig/anchors.json --tag rule_v2_anchor_exact || exit 1
$P scripts/06_llmfree_eval.py --dataset multitq_orig --tag rule_v2_anchor_exact_anydate \
   --slots cache/multitq_orig/slots_rule_v2.json || exit 1
step "C' rewrite + equal window"
$P scripts/05_rerank.py $RW $COMMON --equal_window --tag rule_v2_eqwin_exact --rerank_dir $RWDIR || exit 1
$P scripts/06_llmfree_eval.py --dataset multitq --questions questions_v2.json --tag rule_v2_eqwin_exact_anydate \
   --slots cache/multitq/slots_rule_v2.json --rerank_dir $RWDIR || exit 1
step "D  rewrite + grounding (anchors where the rewrite left an event anchor) + equal window"
$P scripts/05_rerank.py $RW $COMMON --anchors cache/multitq_orig/anchors.json --equal_window \
   --tag rule_v2_ground_exact --rerank_dir $RWDIR || exit 1
$P scripts/06_llmfree_eval.py --dataset multitq --questions questions_v2.json --tag rule_v2_ground_exact_anydate \
   --slots cache/multitq/slots_rule_v2.json --rerank_dir $RWDIR || exit 1
step "done"
