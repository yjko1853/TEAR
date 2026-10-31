"""Figure 1 example (question 3000150): top candidates under none, ATC, TR and TEAR with rule flags.

Run from the repository root:  .venv/bin/python -I paper/derived/figure1_example.py > paper/derived/figure1_example.txt
"""
import json, numpy as np, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root (tear/)
from tear.temporal import date_to_days
d = "cache/multitq"; k = "3000150"
order = [str(x) for x in json.load(open(f"{d}/rerank/question_order.json"))["question_ids"]]; i = order.index(k)
ids = np.load(f"{d}/top1000_ids.npy", mmap_mode="r")[i]; sc = np.load(f"{d}/top1000_scores.npy", mmap_mode="r")[i]
f = np.load(f"{d}/facts.npz"); voc = json.load(open(f"{d}/vocab.json")); E = voc["entities"]
sent = open(f"{d}/sentences.txt").read().split("\n")
td = date_to_days("2009-07-03")
s, r, o, t = f["s_id"][ids], f["r_id"][ids], f["o_id"][ids], f["t_days"][ids]
atc = np.array([E[x] == "Kuwait" for x in s]); valid = t < td
rep = np.zeros(len(ids), bool); best = {}
for j in range(len(ids)):
    if not valid[j]: continue
    key = (int(r[j]), int(o[j]))
    if key not in best or t[j] > t[best[key]]: best[key] = j
for j in best.values(): rep[j] = True
tear = sc * np.where(atc, .8, 1) * np.where(rep, 1, .8)
print("sim-order top5:"); [print(" ", round(float(sc[j]),4), sent[ids[j]], "ATC" if atc[j] else "", "valid" if valid[j] else "INVALID", "rep" if rep[j] else "nonrep") for j in range(5)]
for name, score in (("atc", sc*np.where(atc,.8,1)), ("tr", sc*np.where(rep,1,.8)), ("tear", tear)):
    o_ = np.argsort(-score, kind="stable")[:4]
    print(name); [print(" ", round(float(score[j]),4), round(float(sc[j]),4), sent[ids[j]], "ATC" if atc[j] else "", "valid" if valid[j] else "INVALID", "rep" if rep[j] else "nonrep") for j in o_]
gold = [j for j in range(len(ids)) if "France Make a visit Kuwait" in sent[ids[j]]]
print("gold rank in sim order:", [(j+1, sent[ids[j]]) for j in gold][:5])
