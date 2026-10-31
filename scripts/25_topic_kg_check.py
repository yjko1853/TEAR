"""Automatic KG check of the rule-v2 topic entities (main setting: original MultiTQ questions), in place of a manual
topic check. For each test question: is the topic entity a KG entity, and does it share a fact with a gold answer
(entity answers: a fact between the topic and the answer in either direction; time answers: a fact of the topic dated
inside the gold date)? A topic without such a fact is not necessarily wrong, but a right topic almost always has one.

Output: results/topic_kg_check.csv, printed table.
"""
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tear.paths import CACHE, RESULTS  # noqa: E402
from tear.utils.io import load_json  # noqa: E402


def main():
    d = CACHE / "multitq_orig"
    qs, slots = load_json(d / "questions.json"), load_json(d / "slots_rule_v2.json")
    voc, F = load_json(d / "vocab.json"), np.load(d / "facts.npz")
    eid = {e: i for i, e in enumerate(voc["entities"])}
    S, O, T = F["s_id"], F["o_id"], F["t_id"]
    ts = voc["timestamps"]
    partners, dates = defaultdict(set), defaultdict(set)
    for s, o, t in zip(S.tolist(), O.tolist(), T.tolist()):
        partners[s].add(o); partners[o].add(s)
        dates[s].add(ts[t]); dates[o].add(ts[t])
    rows = []
    for q in qs:
        lab = slots.get(str(q["quid"]), {})
        topic = lab.get("topic_entity")
        t = eid.get(topic)
        linked = False
        if t is not None:
            if q["answer_type"] == "time":
                linked = any(any(x.startswith(a) for x in dates[t]) for a in q["answers"])
            else:
                linked = any(eid.get(a) in partners[t] for a in q["answers"])
        rows.append({"qtype": q["qtype"], "answer_type": q["answer_type"], "has_topic": topic is not None,
                     "in_kg": t is not None, "linked_to_gold": linked})
    df = pd.DataFrame(rows)
    out = []
    for name, g in [("all", df)] + list(df.groupby("qtype")) + [(f"answer={k}", v) for k, v in df.groupby("answer_type")]:
        out.append({"group": name, "n": len(g), "topic_found": g.has_topic.mean(), "topic_in_kg": g.in_kg.mean(),
                    "linked_to_gold": g.linked_to_gold.mean(), "in_kg_not_linked": (g.in_kg & ~g.linked_to_gold).mean()})
    tab = pd.DataFrame(out)
    tab.to_csv(RESULTS / "topic_kg_check.csv", index=False)
    print(tab.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
