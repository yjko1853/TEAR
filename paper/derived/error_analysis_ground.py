"""Error breakdown, error propagation and candidate qualitative cases (main setting: original questions + KG anchor grounding, dataset multitq_orig, tag rule_v2_ground_exact_anydate).

Run from the repository root:  .venv/bin/python -I paper/derived/error_analysis_ground.py
Outputs (paper/derived/):
  error_breakdown_ground.csv     per reranker (none, distance penalty = timer4, tear, tear_window) and per qtype: share of
                             questions whose LLM-free top-1 answer is wrong, split into
                             A no gold answer in any element of the top-15 facts,
                             B gold answer in the top-15 but not in the top-1 fact,
                             C gold answer in the top-1 fact but read from another slot or at another granularity
  error_propagation_ground.csv   LLM-free Hit@1 (none, tear_window) for groups defined by rule inputs:
                             timestamp decision right/wrong (gold answer_type), parsed date yes/no (before_after),
                             first/last cue present/absent (after_first, before_last), topic entity present/absent
  case_candidates_ground.txt     per qtype two success cases (none wrong, tear_window right) and two failure cases
                             (tear_window wrong) with the top-3 facts of both rankings, for the qualitative section
"""
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root (tear/)
from tear.data.multitq import stage_query  # noqa: E402
from tear.evaluation.hits import read_answer_from_fact  # noqa: E402
from tear.temporal import extract_temporal_cue, target_days  # noqa: E402
from tear.text import exact_match, normalize  # noqa: E402

OUT = Path(__file__).resolve().parent
TAG = "rule_v2_ground_exact_anydate"
d = Path("cache/multitq_orig")
qs = json.load(open(d / "questions.json"))
order = [str(k) for k in json.load(open(d / "rerank" / "question_order.json"))["question_ids"]]
qby = {str(q["quid"]): q for q in qs}
slots = json.load(open(d / "slots_rule_v2.json"))
anchors = json.load(open(d / "anchors.json"))   # resolved reference events (tear/labeling/anchor.py)
f = np.load(d / "facts.npz"); voc = json.load(open(d / "vocab.json"))
E, R, T = voc["entities"], voc["relations"], voc["timestamps"]
En = [normalize(x) for x in E]; Rn = [normalize(x) for x in R]; Tn = [normalize(x) for x in T]
s_id, r_id, o_id, t_id = f["s_id"], f["r_id"], f["o_id"], f["t_id"]
sent = (d / "sentences.txt").read_text().split("\n")
RK = {"none": "none", "distance_penalty": "timer4", "tear": "tear", "tear_window": "tear_window"}
top = {name: np.load(d / "rerank" / f"{rk}_{TAG}_top30_ids.npy", mmap_mode="r") for name, rk in RK.items()}


def contains(fid, golds):
    els = (En[s_id[fid]], Rn[r_id[fid]], En[o_id[fid]], Tn[t_id[fid]])
    return any(g and g in e for g in golds for e in els)


rows = []
for i, k in enumerate(order):
    q = qby[k]; golds = [normalize(a) for a in q["answers"]]; sl = slots[k]["answer_slot"]; text = stage_query(q)
    rec = {"quid": k, "qtype": q["qtype"], "answer_type": q["answer_type"], "slot": sl,
           "topic": bool(slots[k]["topic_entity"]),
           "explicit_date": bool(target_days(text, "any", "v2")),
           "anchored": k in anchors, "anchor_resolved": bool(anchors.get(k, {}).get("date")),
           "parsed_date": bool(target_days(text, "any", "v2")) or (anchors.get(k, {}).get("type") in ("before", "after") and bool(anchors.get(k, {}).get("date"))),
           "cues": ",".join(extract_temporal_cue(text))}
    for name, arr in top.items():
        ids = [int(x) for x in arr[i, :15]]
        ok = exact_match(read_answer_from_fact(sl, ids[0], f, E, T, text), q["answers"])
        rec[f"{name}_ok"] = ok
        if not ok:
            rec[f"{name}_err"] = "C" if contains(ids[0], golds) else ("B" if any(contains(x, golds) for x in ids) else "A")
        else:
            rec[f"{name}_err"] = ""
    rows.append(rec)
df = pd.DataFrame(rows)

# ---- error breakdown
out = []
for name in RK:
    for gname, g in [("ALL", df)] + list(df.groupby("qtype")):
        n = len(g); e = g[f"{name}_err"]
        out.append({"reranker": name, "qtype": gname, "n": n, "hit1": g[f"{name}_ok"].mean(),
                    "err_A_not_in_top15": (e == "A").mean(), "err_B_in_top15_not_top1": (e == "B").mean(),
                    "err_C_in_top1_wrong_slot_or_format": (e == "C").mean()})
eb = pd.DataFrame(out); eb.to_csv(OUT / "error_breakdown_ground.csv", index=False)
print(eb[eb.qtype == "ALL"].round(4).to_string(index=False))

# ---- error propagation
prop = []
def add(group, mask):
    g = df[mask]
    prop.append({"group": group, "n": len(g), "none_hit1": g.none_ok.mean(), "tear_window_hit1": g.tear_window_ok.mean(),
                 "gain": g.tear_window_ok.mean() - g.none_ok.mean()})
ts_right = ((df.slot == "timestamp") == (df.answer_type == "time"))
add("timestamp decision right", ts_right); add("timestamp decision wrong", ~ts_right)
ba = df.qtype == "before_after"
add("before_after, explicit date", ba & df.explicit_date)
add("before_after, grounded anchor date", ba & ~df.explicit_date & df.parsed_date)
add("before_after, no date", ba & ~df.parsed_date)
add("anchored, resolved", df.anchored & df.anchor_resolved); add("anchored, not resolved", df.anchored & ~df.anchor_resolved)
for qt, cue in (("after_first", "first"), ("before_last", "last")):
    m = df.qtype == qt; has = df.cues.str.contains(cue)
    add(f"{qt}, cue present", m & has); add(f"{qt}, cue absent", m & ~has)
ent = df.answer_type == "entity"
add("entity question, topic found", ent & df.topic); add("entity question, no topic", ent & ~df.topic)
pr = pd.DataFrame(prop); pr.to_csv(OUT / "error_propagation_ground.csv", index=False); print(pr.round(4).to_string(index=False))

# ---- qualitative case candidates
rng = random.Random(23)
with open(OUT / "case_candidates_ground.txt", "w") as fo:
    for qt in sorted(df.qtype.unique()):
        g = df[df.qtype == qt]
        for kind, m in (("SUCCESS (none wrong, TEAR+window right)", (~g.none_ok) & g.tear_window_ok),
                        ("FAILURE (TEAR+window wrong)", ~g.tear_window_ok)):
            keys = list(g[m].quid); rng.shuffle(keys)
            for k in keys[:2]:
                q = qby[k]; i = order.index(k); row = df[df.quid == k].iloc[0]
                fo.write(f"== {qt} | {kind} | quid {k} | error class {row.tear_window_err or '-'}\n")
                fo.write(f"   original : {q['question']}\n   query    : {stage_query(q)}\n   answers  : {q['answers']}\n")
                fo.write(f"   slot {slots[k]['answer_slot']} | topic {slots[k]['topic_entity']} | cues {row.cues} | date {row.parsed_date}\n")
                for name in ("none", "tear_window"):
                    fo.write(f"   {name:12s}: " + " || ".join(sent[int(x)] for x in top[name][i, :3]) + "\n")
print(open(OUT / "case_candidates_ground.txt").read()[:3000])
