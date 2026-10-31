"""Training-free rerankers applied to a shared top-N candidate pool.

    atc        answer-type (answer-slot) penalty: candidates whose predicted answer slot holds the topic entity
    tr         temporal reranking: one representative fact per group for first/last cues, before/after validity
    baselines  TimeR4-style before reranking, chronological sort, hard time window
    features   per-question feature arrays built from the pure functions above
    core       reranker registry: rerank(name, sims, features) -> permutation of the candidate pool
"""
from .core import RERANKERS, rerank  # noqa: F401
