# Answer granularity (MultiTQ reader sample, Qwen2.5-7B-Instruct-4bit, k = 15)

Pre-registered: results/prereg_retriever_and_granularity.md (B). A = v3, B = v3 + projection, C = v3g, D = v3g + projection; the unit is predicted by the rules (no gold label).

- predicted unit matches the gold answers on 15876/15884 test time questions (0.9995)
- rule-v2 slot is timestamp for 15820/15884 test time questions (0.9960)

## Exact Hit@1

| setting     | subset              | n    | A     | B     | C     | D     | wrong_unit_only_v3   | wrong_unit_only_v3g   |
|:------------|:--------------------|:-----|:------|:------|:------|:------|:---------------------|:----------------------|
| none        | all 6,000           | 6000 | 0.235 | 0.308 | 0.315 | 0.315 | 0.073                | 0.000                 |
| none        | gold time questions | 1305 | 0.407 | 0.743 | 0.772 | 0.774 | 0.337                | 0.002                 |
| tear_window | all 6,000           | 6000 | 0.322 | 0.385 | 0.396 | 0.396 | 0.064                | 0.001                 |
| tear_window | gold time questions | 1305 | 0.405 | 0.697 | 0.745 | 0.747 | 0.293                | 0.002                 |

## Paired tests (bootstrap 10,000, seed 2026)

| setting     | subset              | system   | baseline   | n    | diff   | ci_low   | ci_high   | mcnemar_p   | gained   | lost   |
|:------------|:--------------------|:---------|:-----------|:-----|:-------|:---------|:----------|:------------|:---------|:-------|
| none        | all 6,000           | C        | A          | 6000 | 0.080  | 0.072    | 0.086     | 1.5e-101    | 486      | 9      |
| none        | all 6,000           | D        | B          | 6000 | 0.007  | 0.004    | 0.009     | 4.3e-06     | 56       | 16     |
| none        | all 6,000           | B        | A          | 6000 | 0.073  | 0.067    | 0.080     | 1.3e-96     | 440      | 1      |
| none        | gold time questions | C        | A          | 1305 | 0.366  | 0.339    | 0.392     | 1.5e-101    | 486      | 9      |
| none        | gold time questions | D        | B          | 1305 | 0.031  | 0.018    | 0.044     | 4.3e-06     | 56       | 16     |
| none        | gold time questions | B        | A          | 1305 | 0.336  | 0.311    | 0.362     | 1.3e-96     | 440      | 1      |
| tear_window | all 6,000           | C        | A          | 6000 | 0.074  | 0.067    | 0.081     | 5.6e-94     | 454      | 10     |
| tear_window | all 6,000           | D        | B          | 6000 | 0.011  | 0.007    | 0.014     | 2.9e-10     | 84       | 19     |
| tear_window | all 6,000           | B        | A          | 6000 | 0.064  | 0.057    | 0.070     | 3.4e-84     | 383      | 1      |
| tear_window | gold time questions | C        | A          | 1305 | 0.340  | 0.313    | 0.367     | 5.6e-94     | 454      | 10     |
| tear_window | gold time questions | D        | B          | 1305 | 0.050  | 0.035    | 0.065     | 2.9e-10     | 84       | 19     |
| tear_window | gold time questions | B        | A          | 1305 | 0.293  | 0.268    | 0.317     | 3.4e-84     | 383      | 1      |
