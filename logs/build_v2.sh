#!/bin/bash
# Rule v2 + query fallback artefacts (new file names only; v1 files are read, never written). CPU only, 2 threads.
cd "$(dirname "$0")/.." || exit 1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export HF_HUB_OFFLINE=1 PYTORCH_ENABLE_MPS_FALLBACK=0
P=".venv/bin/python -I"
S="${SCRATCH:-/tmp}"
step() { echo "[$(date '+%H:%M:%S')] $1"; }
step "1 questions_v2.json";  $P scripts/01_prepare_data.py --dataset multitq --make_query || exit 1
step "2 q_emb_v2 + top1000_v2 (changed rows only, CPU)"
$P scripts/03_search.py --dataset multitq --questions questions_v2.json --suffix _v2 --incremental \
   --device cpu --torch_threads 2 --threads 2 || exit 1
step "3 NER spans for changed questions (CPU)"
$P scripts/04_label_slots.py --dataset multitq --method ner --labels all --questions questions_v2.json --suffix _v2 \
   --device cpu --torch_threads 2 --verify_sample 0 --report_dir "$S/ner_v2_diag" || exit 1
step "4 slots_rule_v2.json"
$P scripts/04_label_slots.py --dataset multitq --method rule --rule_version v2 --questions questions_v2.json \
   --ner_topics cache/multitq/topic_ner_all_v2.json --out cache/multitq/slots_rule_v2.json || exit 1
step "5 rerank rule_v2_exact_anydate (sweep)"
$P scripts/05_rerank.py --dataset multitq --questions questions_v2.json --suffix _v2 --slots cache/multitq/slots_rule_v2.json \
   --atc_match exact --date_granularity any --date_parser v2 --sweep --tag rule_v2_exact || exit 1
step "6 LLM-free Hit@1 v2"
$P scripts/06_llmfree_eval.py --questions questions_v2.json --tag rule_v2_exact_anydate \
   --slots cache/multitq/slots_rule_v2.json --baseline none || exit 1
step "done"
