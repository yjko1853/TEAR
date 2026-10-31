"""Parse an LLM answer into a list of strings ("['a', 'b']", "1. a\n2. b", "a, b", ...)."""
from __future__ import annotations

import ast
import re


def parse_list(pred: str):
    s = pred.strip()
    m = re.search(r"\[.*?\]", s, re.S)
    if m:
        try:
            v = ast.literal_eval(m.group(0))
            if isinstance(v, (list, tuple)):
                return [str(x) for x in v]
        except Exception:
            pass
        inner = m.group(0)[1:-1]
        return [x.strip(" '\"") for x in inner.split(",") if x.strip(" '\"")]
    items = [re.sub(r"^\d+[.)]\s*", "", x.strip(" -*'\"")) for x in re.split(r"[\n,]", s)]
    return [x.strip(" '\"") for x in items if x.strip(" '\"")]
