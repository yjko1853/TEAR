#!/bin/bash
# LLM-free Hit@1 significance against other baselines and the substring-ATC run (paper/derived/README.md):
# scripts/06_llmfree_eval.py with another --baseline or --tag, written to a temporary folder and copied here.
# Run from the repository root after the pipeline stages 01-05 (cache/multitq, cache/multitq_orig). CPU only.
cd "$(dirname "$0")/../.." || exit 1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 HF_HUB_OFFLINE=1
P=".venv/bin/python -I"; OUT=paper/derived; TMP=$(mktemp -d) || exit 1
run() {  # <name> <scripts/06 arguments>
  $P scripts/06_llmfree_eval.py --results_dir "$TMP/$1" "${@:2}" || exit 1
}
# main setting (original questions + anchor grounding) against the distance penalty (TimeR4 Eq. 9-10)
run g_t4 --dataset multitq_orig --tag rule_v2_ground_exact_anydate --slots cache/multitq_orig/slots_rule_v2.json --baseline timer4
cp "$TMP/g_t4/multitq_orig_llmfree_significance_rule_v2_ground_exact_anydate.csv" $OUT/llmfree_significance_vs_distance_penalty_ground.csv
# rewritten questions, rule v2 + query fallback
run v2_t4 --dataset multitq --questions questions_v2.json --tag rule_v2_exact_anydate --slots cache/multitq/slots_rule_v2.json --baseline timer4
cp "$TMP/v2_t4/multitq_llmfree_significance_rule_v2_exact_anydate.csv" $OUT/llmfree_significance_vs_distance_penalty_v2.csv
# rewritten questions, rule v1
run v1_none --dataset multitq --tag rule_exact_anydate --slots cache/multitq/slots_rule.json --baseline none
cp "$TMP/v1_none/multitq_llmfree_significance_rule_exact_anydate.csv" $OUT/llmfree_significance_vs_none.csv
run v1_t4 --dataset multitq --tag rule_exact_anydate --slots cache/multitq/slots_rule.json --baseline timer4
cp "$TMP/v1_t4/multitq_llmfree_significance_rule_exact_anydate.csv" $OUT/llmfree_significance_vs_timer4.csv
# rule v1 with substring ATC (tag rule_anydate)
run v1_sub --dataset multitq --tag rule_anydate --slots cache/multitq/slots_rule.json
cp "$TMP/v1_sub/multitq_llmfree_hit1_rule_anydate.csv" $OUT/llmfree_hit1_substring_atc.csv
rm -r "$TMP"
