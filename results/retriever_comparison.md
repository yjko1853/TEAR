# Second retriever (MultiTQ test, original questions + grounding, LLM-free)

Pre-registered: results/prereg_retriever_and_granularity.md (A). Same labels, grounding, lambdas and rules; only the retriever differs.

| retriever         | reranker    | coverage15   | hit1   | hit1_entity   | hit1_time   | diff_vs_none   | ci_low   | ci_high   | mcnemar_p   |
|:------------------|:------------|:-------------|:-------|:--------------|:------------|:---------------|:---------|:----------|:------------|
| all-mpnet-base-v2 | none        | 0.608        | 0.276  | 0.086         | 0.739       | 0.000          |          |           |             |
| all-mpnet-base-v2 | timer4      | 0.656        | 0.302  | 0.122         | 0.739       | 0.026          | 0.024    | 0.027     | 5.0e-195    |
| all-mpnet-base-v2 | hardwindow  | 0.670        | 0.319  | 0.146         | 0.739       | 0.043          | 0.041    | 0.045     | 0.0e+00     |
| all-mpnet-base-v2 | atc         | 0.644        | 0.366  | 0.213         | 0.739       | 0.090          | 0.088    | 0.093     | 0.0e+00     |
| all-mpnet-base-v2 | tr          | 0.683        | 0.328  | 0.109         | 0.861       | 0.051          | 0.050    | 0.053     | 0.0e+00     |
| all-mpnet-base-v2 | tear        | 0.725        | 0.447  | 0.277         | 0.861       | 0.171          | 0.168    | 0.174     | 0.0e+00     |
| all-mpnet-base-v2 | tear_window | 0.756        | 0.502  | 0.354         | 0.861       | 0.225          | 0.222    | 0.229     | 0.0e+00     |
| bge-base-en-v1.5  | none        | 0.729        | 0.330  | 0.149         | 0.772       | 0.000          |          |           |             |
| bge-base-en-v1.5  | timer4      | 0.765        | 0.350  | 0.176         | 0.772       | 0.020          | 0.018    | 0.021     | 6.6e-117    |
| bge-base-en-v1.5  | hardwindow  | 0.780        | 0.367  | 0.200         | 0.772       | 0.036          | 0.035    | 0.038     | 1.2e-275    |
| bge-base-en-v1.5  | atc         | 0.778        | 0.487  | 0.371         | 0.772       | 0.157          | 0.154    | 0.161     | 0.0e+00     |
| bge-base-en-v1.5  | tr          | 0.790        | 0.384  | 0.174         | 0.894       | 0.054          | 0.052    | 0.056     | 0.0e+00     |
| bge-base-en-v1.5  | tear        | 0.845        | 0.563  | 0.428         | 0.894       | 0.233          | 0.230    | 0.237     | 0.0e+00     |
| bge-base-en-v1.5  | tear_window | 0.865        | 0.616  | 0.501         | 0.894       | 0.285          | 0.282    | 0.289     | 0.0e+00     |
