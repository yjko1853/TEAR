"""Answer-slot and topic-entity labels consumed by the ATC penalty.

Label format (JSON): {question_id: {"answer_slot": "subject|object|relation|timestamp", "topic_entity": str, ...}}

    rules   rule-based classifier (default; training-free, API-free, deterministic)
    ner     port of the original TEAR topic-entity extractor (dslim/bert-base-NER + country list + fuzzy match)
    oracle  KG-derived labels; diagnostic only (see oracle.py: direction errors on reciprocal ICEWS events)
"""
