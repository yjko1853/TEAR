"""TEAR: training-free Temporal-constraint and Answer-type aware Reranking for temporal KGQA.

Package layout (lower layers never import higher ones):

    paths, utils, text, temporal          basic helpers (no numpy models)
    data                                  dataset preparation and the shared candidate pool
    retrieval                             sentence encoder + exact top-N search
    labeling                              answer-slot / topic-entity labels (rule, NER port, KG oracle)
    rerank                                ATC penalty, temporal reranking (TR), baselines, reranker registry
    reader                                LLM reader (prompt, output parsing, local MLX model)
    evaluation                            coverage@k, Hit@1, significance tests
    analysis, reporting                   motivation analyses and the markdown report

Command-line entry points live in scripts/ (numbered by pipeline stage).
"""
__version__ = "2.0.0"
