# Paraphrase robustness of anchor grounding (MultiTQ, pre-registered)

Pre-registration: results/prereg_paraphrase_robustness.md (with Amendments 1 and 2). Sample: 600 multi-constraint test questions without a date or month name, 200 per type, seed 2026. Paraphrases: local Qwen2.5-7B-Instruct-4bit, greedy, one per question.

- valid paraphrases (automatic filter): 439/600
- LLM judge: same meaning 355/600; valid and same 322/439
- meaning check not filled yet

## Anchor detection, resolution, date agreement, slot agreement

| subset                | qtype       | n   | detected_orig   | detected_para   | resolved_orig   | resolved_para   | both_resolved   | same_date_of_both   | slot_agree   |
|:----------------------|:------------|:----|:----------------|:----------------|:----------------|:----------------|:----------------|:--------------------|:-------------|
| valid                 | all         | 439 | 0.954           | 0.882           | 0.948           | 0.863           | 377             | 0.769               | 0.793        |
| valid                 | after_first | 141 | 1.000           | 0.986           | 0.993           | 0.972           | 137             | 0.701               | 0.738        |
| valid                 | before_last | 143 | 1.000           | 0.916           | 0.986           | 0.881           | 125             | 0.792               | 0.839        |
| valid                 | equal_multi | 155 | 0.871           | 0.755           | 0.871           | 0.748           | 115             | 0.826               | 0.800        |
| all                   | all         | 600 | 0.950           | 0.810           | 0.945           | 0.788           | 471             | 0.722               | 0.782        |
| all                   | after_first | 200 | 1.000           | 0.955           | 0.995           | 0.940           | 188             | 0.665               | 0.760        |
| all                   | before_last | 200 | 1.000           | 0.855           | 0.990           | 0.815           | 162             | 0.735               | 0.845        |
| all                   | equal_multi | 200 | 0.850           | 0.620           | 0.850           | 0.610           | 121             | 0.793               | 0.740        |
| judged same meaning   | all         | 355 | 0.969           | 0.913           | 0.961           | 0.893           | 315             | 0.781               | 0.820        |
| judged same meaning   | after_first | 129 | 1.000           | 0.984           | 0.992           | 0.961           | 124             | 0.718               | 0.736        |
| judged same meaning   | before_last | 113 | 1.000           | 0.912           | 0.982           | 0.876           | 98              | 0.857               | 0.876        |
| judged same meaning   | equal_multi | 113 | 0.903           | 0.832           | 0.903           | 0.832           | 93              | 0.785               | 0.858        |
| valid and judged same | all         | 322 | 0.969           | 0.935           | 0.960           | 0.916           | 293             | 0.788               | 0.823        |
| valid and judged same | after_first | 122 | 1.000           | 0.984           | 0.992           | 0.967           | 118             | 0.703               | 0.738        |
| valid and judged same | before_last | 97  | 1.000           | 0.918           | 0.979           | 0.876           | 84              | 0.893               | 0.897        |
| valid and judged same | equal_multi | 103 | 0.903           | 0.893           | 0.903           | 0.893           | 91              | 0.802               | 0.854        |

## LLM-free Hit@1 and coverage@15 (main setting: TEAR+window, lambda 0.8; ground = with grounding)

| subset                | qtype       | n   | orig_none_ground_hit1   | orig_tear_window_ground_hit1   | orig_tear_window_plain_hit1   | orig_none_ground_cov15   | orig_tear_window_ground_cov15   | orig_tear_window_plain_cov15   | para_none_ground_hit1   | para_tear_window_ground_hit1   | para_tear_window_plain_hit1   | para_none_ground_cov15   | para_tear_window_ground_cov15   | para_tear_window_plain_cov15   |
|:----------------------|:------------|:----|:------------------------|:-------------------------------|:------------------------------|:-------------------------|:--------------------------------|:-------------------------------|:------------------------|:-------------------------------|:------------------------------|:-------------------------|:--------------------------------|:-------------------------------|
| valid                 | all         | 439 | 0.002                   | 0.257                          | 0.032                         | 0.228                    | 0.601                           | 0.289                          | 0.007                   | 0.185                          | 0.018                         | 0.226                    | 0.519                           | 0.292                          |
| valid                 | after_first | 141 | 0.000                   | 0.248                          | 0.050                         | 0.206                    | 0.525                           | 0.248                          | 0.000                   | 0.177                          | 0.035                         | 0.199                    | 0.496                           | 0.291                          |
| valid                 | before_last | 143 | 0.000                   | 0.210                          | 0.028                         | 0.154                    | 0.629                           | 0.259                          | 0.007                   | 0.168                          | 0.007                         | 0.168                    | 0.434                           | 0.224                          |
| valid                 | equal_multi | 155 | 0.006                   | 0.310                          | 0.019                         | 0.316                    | 0.645                           | 0.355                          | 0.013                   | 0.206                          | 0.013                         | 0.303                    | 0.619                           | 0.355                          |
| all                   | all         | 600 | 0.008                   | 0.268                          | 0.037                         | 0.233                    | 0.602                           | 0.303                          | 0.010                   | 0.163                          | 0.022                         | 0.243                    | 0.478                           | 0.300                          |
| all                   | after_first | 200 | 0.005                   | 0.250                          | 0.055                         | 0.200                    | 0.530                           | 0.240                          | 0.005                   | 0.160                          | 0.030                         | 0.200                    | 0.430                           | 0.270                          |
| all                   | before_last | 200 | 0.005                   | 0.230                          | 0.030                         | 0.170                    | 0.605                           | 0.290                          | 0.005                   | 0.150                          | 0.010                         | 0.195                    | 0.415                           | 0.250                          |
| all                   | equal_multi | 200 | 0.015                   | 0.325                          | 0.025                         | 0.330                    | 0.670                           | 0.380                          | 0.020                   | 0.180                          | 0.025                         | 0.335                    | 0.590                           | 0.380                          |
| judged same meaning   | all         | 355 | 0.006                   | 0.256                          | 0.034                         | 0.239                    | 0.600                           | 0.287                          | 0.006                   | 0.200                          | 0.020                         | 0.251                    | 0.541                           | 0.307                          |
| judged same meaning   | after_first | 129 | 0.000                   | 0.256                          | 0.054                         | 0.217                    | 0.535                           | 0.225                          | 0.000                   | 0.186                          | 0.031                         | 0.225                    | 0.504                           | 0.302                          |
| judged same meaning   | before_last | 113 | 0.009                   | 0.239                          | 0.035                         | 0.221                    | 0.619                           | 0.327                          | 0.009                   | 0.221                          | 0.018                         | 0.221                    | 0.504                           | 0.283                          |
| judged same meaning   | equal_multi | 113 | 0.009                   | 0.274                          | 0.009                         | 0.283                    | 0.655                           | 0.319                          | 0.009                   | 0.195                          | 0.009                         | 0.310                    | 0.619                           | 0.336                          |
| valid and judged same | all         | 322 | 0.003                   | 0.248                          | 0.034                         | 0.227                    | 0.596                           | 0.280                          | 0.006                   | 0.202                          | 0.019                         | 0.230                    | 0.540                           | 0.295                          |
| valid and judged same | after_first | 122 | 0.000                   | 0.254                          | 0.057                         | 0.213                    | 0.525                           | 0.238                          | 0.000                   | 0.180                          | 0.033                         | 0.213                    | 0.492                           | 0.303                          |
| valid and judged same | before_last | 97  | 0.000                   | 0.216                          | 0.031                         | 0.196                    | 0.629                           | 0.299                          | 0.010                   | 0.216                          | 0.010                         | 0.175                    | 0.495                           | 0.247                          |
| valid and judged same | equal_multi | 103 | 0.010                   | 0.272                          | 0.010                         | 0.272                    | 0.650                           | 0.311                          | 0.010                   | 0.214                          | 0.010                         | 0.301                    | 0.641                           | 0.330                          |

## Paired tests on exact LLM-free Hit@1 (bootstrap 10,000, seed 2026)

| subset                | system                       | baseline                     | n   | diff   | ci_low   | ci_high   | bootstrap_p   | mcnemar_p   | gained   | lost   |
|:----------------------|:-----------------------------|:-----------------------------|:----|:-------|:---------|:----------|:--------------|:------------|:---------|:-------|
| valid                 | orig:tear_window_ground_hit1 | orig:none_ground_hit1        | 439 | 0.255  | 0.214    | 0.296     | 0.0e+00       | 9.8e-26     | 112      | 0      |
| valid                 | para:tear_window_ground_hit1 | para:none_ground_hit1        | 439 | 0.178  | 0.144    | 0.214     | 0.0e+00       | 2.8e-18     | 78       | 0      |
| valid                 | para:tear_window_ground_hit1 | para:tear_window_plain_hit1  | 439 | 0.166  | 0.132    | 0.203     | 0.0e+00       | 2.3e-16     | 75       | 2      |
| valid                 | para:tear_window_ground_hit1 | orig:tear_window_ground_hit1 | 439 | -0.073 | -0.109   | -0.039    | 0.0e+00       | 1.4e-04     | 17       | 49     |
| valid                 | para:none_ground_hit1        | orig:none_ground_hit1        | 439 | 0.005  | 0.000    | 0.011     | 2.8e-01       | 5.0e-01     | 2        | 0      |
| all                   | orig:tear_window_ground_hit1 | orig:none_ground_hit1        | 600 | 0.260  | 0.225    | 0.295     | 0.0e+00       | 6.2e-35     | 157      | 1      |
| all                   | para:tear_window_ground_hit1 | para:none_ground_hit1        | 600 | 0.153  | 0.125    | 0.182     | 0.0e+00       | 2.4e-21     | 92       | 0      |
| all                   | para:tear_window_ground_hit1 | para:tear_window_plain_hit1  | 600 | 0.142  | 0.113    | 0.170     | 0.0e+00       | 1.3e-18     | 88       | 3      |
| all                   | para:tear_window_ground_hit1 | orig:tear_window_ground_hit1 | 600 | -0.105 | -0.138   | -0.072    | 0.0e+00       | 2.1e-09     | 22       | 85     |
| all                   | para:none_ground_hit1        | orig:none_ground_hit1        | 600 | 0.002  | -0.007   | 0.010     | 8.7e-01       | 1.0e+00     | 4        | 3      |
| judged same meaning   | orig:tear_window_ground_hit1 | orig:none_ground_hit1        | 355 | 0.251  | 0.206    | 0.296     | 0.0e+00       | 2.8e-20     | 90       | 1      |
| judged same meaning   | para:tear_window_ground_hit1 | para:none_ground_hit1        | 355 | 0.194  | 0.155    | 0.237     | 0.0e+00       | 2.7e-16     | 69       | 0      |
| judged same meaning   | para:tear_window_ground_hit1 | para:tear_window_plain_hit1  | 355 | 0.180  | 0.141    | 0.223     | 0.0e+00       | 2.2e-14     | 66       | 2      |
| judged same meaning   | para:tear_window_ground_hit1 | orig:tear_window_ground_hit1 | 355 | -0.056 | -0.099   | -0.017    | 7.6e-03       | 9.7e-03     | 17       | 37     |
| judged same meaning   | para:none_ground_hit1        | orig:none_ground_hit1        | 355 | 0.000  | -0.008   | 0.008     | 1.0e+00       | 1.0e+00     | 1        | 1      |
| valid and judged same | orig:tear_window_ground_hit1 | orig:none_ground_hit1        | 322 | 0.245  | 0.199    | 0.292     | 0.0e+00       | 1.7e-18     | 79       | 0      |
| valid and judged same | para:tear_window_ground_hit1 | para:none_ground_hit1        | 322 | 0.196  | 0.152    | 0.242     | 0.0e+00       | 5.7e-15     | 63       | 0      |
| valid and judged same | para:tear_window_ground_hit1 | para:tear_window_plain_hit1  | 322 | 0.183  | 0.140    | 0.227     | 0.0e+00       | 2.7e-13     | 61       | 2      |
| valid and judged same | para:tear_window_ground_hit1 | orig:tear_window_ground_hit1 | 322 | -0.047 | -0.087   | -0.006    | 3.4e-02       | 4.1e-02     | 16       | 31     |
| valid and judged same | para:none_ground_hit1        | orig:none_ground_hit1        | 322 | 0.003  | 0.000    | 0.009     | 7.2e-01       | 1.0e+00     | 1        | 0      |

## Where the paraphrases lose hits (TEAR+window with grounding, exact Hit@1)

| subset                | cause                        | n   | orig_hit1   | para_hit1   | lost_hits   |
|:----------------------|:-----------------------------|:----|:------------|:------------|:------------|
| valid                 | 0 unresolved on the original | 23  | 0.043       | 0.000       | 1           |
| valid                 | 1 anchor phrase not found    | 33  | 0.424       | 0.000       | 14          |
| valid                 | 2 found, not resolved        | 6   | 0.167       | 0.000       | 1           |
| valid                 | 3 slot label changed         | 70  | 0.271       | 0.157       | 8           |
| valid                 | 4 same slot, other date      | 39  | 0.154       | 0.051       | 4           |
| valid                 | 5 same slot and date         | 268 | 0.269       | 0.254       | 4           |
| valid and judged same | 0 unresolved on the original | 13  | 0.077       | 0.000       | 1           |
| valid and judged same | 1 anchor phrase not found    | 12  | 0.500       | 0.000       | 6           |
| valid and judged same | 2 found, not resolved        | 4   | 0.250       | 0.000       | 1           |
| valid and judged same | 3 slot label changed         | 51  | 0.216       | 0.137       | 4           |
| valid and judged same | 4 same slot, other date      | 27  | 0.185       | 0.074       | 3           |
| valid and judged same | 5 same slot and date         | 215 | 0.260       | 0.260       | 0           |
