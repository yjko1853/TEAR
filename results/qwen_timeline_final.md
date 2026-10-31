# TimelineKGQA reader, final (Qwen2.5-7B-Instruct-4bit, prompt v3i, k = 15, no-leak sample)

Settings: none = retriever order; tear_window = TEAR + window + interval grounding (rule labels);
hardwindow = window + interval grounding, no ATC / TR (needs only entity vs time from the rule label).
Hit@1 exact = first answer equals a gold answer after normalisation.

- icews: 1000 questions; grounded 1000, constraint resolved 1000, rule label timestamp (window not applied) 0
- cron: 1000 questions; grounded 996, constraint resolved 996, rule label timestamp (window not applied) 0

## Hit@1 by subset

| dataset   | subset         | n    | none_exact   | none_lenient   | tear_window_exact   | tear_window_lenient   | hardwindow_exact   | hardwindow_lenient   |
|:----------|:---------------|:-----|:-------------|:---------------|:--------------------|:----------------------|:-------------------|:---------------------|
| icews     | all            | 1000 | 0.074        | 0.077          | 0.125               | 0.129                 | 0.129              | 0.133                |
| icews     | level=complex  | 234  | 0.120        | 0.124          | 0.214               | 0.222                 | 0.231              | 0.239                |
| icews     | level=medium   | 766  | 0.060        | 0.063          | 0.098               | 0.101                 | 0.098              | 0.101                |
| icews     | answer=object  | 610  | 0.079        | 0.084          | 0.128               | 0.134                 | 0.141              | 0.148                |
| icews     | answer=subject | 390  | 0.067        | 0.067          | 0.121               | 0.121                 | 0.110              | 0.110                |
| cron      | all            | 1000 | 0.075        | 0.077          | 0.116               | 0.123                 | 0.159              | 0.167                |
| cron      | level=complex  | 389  | 0.131        | 0.136          | 0.183               | 0.190                 | 0.275              | 0.283                |
| cron      | level=medium   | 611  | 0.039        | 0.039          | 0.074               | 0.080                 | 0.085              | 0.093                |
| cron      | answer=object  | 704  | 0.097        | 0.099          | 0.131               | 0.141                 | 0.189              | 0.199                |
| cron      | answer=subject | 296  | 0.024        | 0.024          | 0.081               | 0.081                 | 0.088              | 0.091                |

## Paired tests (exact Hit@1, bootstrap 10,000, seed 2026)

| dataset   | subset        | system      | baseline    | n    | diff   | ci_low   | ci_high   | bootstrap_p   | mcnemar_p   | gained   | lost   |
|:----------|:--------------|:------------|:------------|:-----|:-------|:---------|:----------|:--------------|:------------|:---------|:-------|
| icews     | all           | hardwindow  | none        | 1000 | 0.055  | 0.037    | 0.073     | 0.0e+00       | 1.0e-08     | 72       | 17     |
| icews     | all           | tear_window | none        | 1000 | 0.051  | 0.032    | 0.071     | 0.0e+00       | 8.4e-07     | 77       | 26     |
| icews     | all           | hardwindow  | tear_window | 1000 | 0.004  | -0.006   | 0.014     | 5.2e-01       | 5.7e-01     | 16       | 12     |
| icews     | level=complex | hardwindow  | none        | 234  | 0.111  | 0.060    | 0.162     | 0.0e+00       | 5.0e-05     | 32       | 6      |
| icews     | level=complex | tear_window | none        | 234  | 0.094  | 0.043    | 0.150     | 1.4e-03       | 1.5e-03     | 33       | 11     |
| icews     | level=complex | hardwindow  | tear_window | 234  | 0.017  | -0.013   | 0.047     | 3.5e-01       | 4.2e-01     | 9        | 5      |
| icews     | level=medium  | hardwindow  | none        | 766  | 0.038  | 0.020    | 0.056     | 0.0e+00       | 8.8e-05     | 40       | 11     |
| icews     | level=medium  | tear_window | none        | 766  | 0.038  | 0.018    | 0.057     | 0.0e+00       | 2.7e-04     | 44       | 15     |
| icews     | level=medium  | hardwindow  | tear_window | 766  | 0.000  | -0.009   | 0.009     | 1.0e+00       | 1.0e+00     | 7        | 7      |
| cron      | all           | hardwindow  | none        | 1000 | 0.084  | 0.066    | 0.104     | 0.0e+00       | 1.0e-16     | 92       | 8      |
| cron      | all           | tear_window | none        | 1000 | 0.041  | 0.021    | 0.061     | 0.0e+00       | 8.1e-05     | 72       | 31     |
| cron      | all           | hardwindow  | tear_window | 1000 | 0.043  | 0.026    | 0.061     | 0.0e+00       | 3.1e-06     | 62       | 19     |
| cron      | level=complex | hardwindow  | none        | 389  | 0.144  | 0.105    | 0.183     | 0.0e+00       | 6.2e-12     | 60       | 4      |
| cron      | level=complex | tear_window | none        | 389  | 0.051  | 0.010    | 0.093     | 1.8e-02       | 2.1e-02     | 44       | 24     |
| cron      | level=complex | hardwindow  | tear_window | 389  | 0.093  | 0.051    | 0.134     | 0.0e+00       | 2.2e-05     | 52       | 16     |
| cron      | level=medium  | hardwindow  | none        | 611  | 0.046  | 0.028    | 0.065     | 0.0e+00       | 6.8e-06     | 32       | 4      |
| cron      | level=medium  | tear_window | none        | 611  | 0.034  | 0.016    | 0.054     | 0.0e+00       | 7.2e-04     | 28       | 7      |
| cron      | level=medium  | hardwindow  | tear_window | 611  | 0.011  | 0.000    | 0.023     | 5.6e-02       | 9.2e-02     | 10       | 3      |
