"""Off-the-shelf sentence encoder and exact top-N search (no training).

macOS note: loading torch (sentence-transformers) and faiss in the same process hangs at index.search on Apple
silicon (two OpenMP runtimes). Question encoding therefore runs in a child process (scripts/03_search.py) and FAISS
runs in the parent; search.numpy_topn is an exact fallback that needs neither.
"""
MODEL = "sentence-transformers/all-mpnet-base-v2"
