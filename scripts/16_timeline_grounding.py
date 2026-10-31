"""TimelineKGQA (ICEWS-Actor, CronQuestions-KG): event-anchored interval constraints resolved in the KG
(tear/labeling/interval_anchor.py) as a TEAR window. CPU only; reads cache/<ds>/ and writes only there and results/.

  --dev_design   validation split, medium / complex subject / object questions: connective lexicon coverage, anchor
                 accuracy against the gold events, and whether the gold answer event passes the window, for the two
                 ways of choosing among several intervals of one anchor (best-scored vs any)
                 -> results/<ds>_grounding_dev.csv
  --make_rule_labels  answer slot / topic entity from the question text with the MultiTQ rules (v2) instead of the
                 dataset's answer_type -> cache/<ds>/slots_rule_v2.json, results/<ds>_rule_label_accuracy.csv
  (default)      test split, shared top-1000 pool: none / hardwindow / atc / tear / tear_window, and hardwindow / tear /
                 tear_window with grounding (window from the anchors; --anchor_atc also penalises anchor entities)
                 -> results/<ds>_grounding_{coverage,llmfree,significance,constraints,event_recall}[_<tag>].csv
                    (no suffix for the first, oracle-label run),
                    cache/<ds>/rerank/<reranker>_<tag>_top30_ids.npy, cache/<ds>/interval_anchors.json
"""
import argparse
import collections
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tear.data.candidates import CandidatePool  # noqa: E402
from tear.data.timelinekgqa import DEFAULT_DATADIR, FILES, parse_events  # noqa: E402
from tear.evaluation.coverage import answer_hit_features, coverage_rows  # noqa: E402
from tear.evaluation.significance import mcnemar, paired_bootstrap  # noqa: E402
from tear.labeling.interval_anchor import IntervalAnchorResolver, allen_of  # noqa: E402
from tear.paths import CACHE, RESULTS  # noqa: E402
from tear.rerank.core import rerank  # noqa: E402
from tear.rerank.features import build_features  # noqa: E402
from tear.text import normalize  # noqa: E402
from tear.utils.io import load_json, save_json  # noqa: E402
from tear.utils.timing import timed  # noqa: E402

END_CUE = re.compile(r"\b(end|ended|ending|finish|finished|stop|stopped|conclude|concluded|terminate|terminated)\b",
                     re.I)        # 'When did X end / finish ...' -> end date (validation split)
TARGET = ("subject", "object")
SLOT_TYPES = ("subject", "object", "timestamp_start", "timestamp_end")
BASE = ["none", "hardwindow", "atc", "tear", "tear_window"]
GROUNDED = ["hardwindow", "tear", "tear_window"]


def load_split(ds: str, split: str) -> list:
    """Questions of a split with gold fact ids (as tear.data.timelinekgqa.prepare; unparseable events -> no gold)."""
    kg_file, q_file = FILES[ds]
    kg = pd.read_csv(DEFAULT_DATADIR / kg_file, dtype=str, keep_default_na=False)
    key2idx = {}
    for i, r in enumerate(kg.itertuples(index=False)):
        key2idx.setdefault((r.subject, r.predicate, r.object, r.start_time, r.end_time), i)
    preds = set(load_json(CACHE / ds / "vocab.json")["predicates"])
    q = pd.read_csv(DEFAULT_DATADIR / q_file, dtype=str, keep_default_na=False)
    out = []
    for r in q[q["split"] == split].itertuples(index=False):
        try:
            evs = parse_events(r.events, preds)
        except ValueError:
            evs = None
        out.append({"qid": r.id, "question": r.question, "answer": r.answer, "answer_type": r.answer_type,
                    "level": r.question_level, "qtype": r.question_type, "temporal_relation": r.temporal_relation,
                    "gold_fact_ids": [key2idx.get(e, -1) for e in evs] if evs else []})
    return out


def encode(texts, threads: int):
    import torch
    torch.set_num_threads(threads)
    from tear.retrieval.encoder import load_encoder
    m = load_encoder("cpu")[0]
    e = m.encode(list(texts), batch_size=64, show_progress_bar=False, convert_to_numpy=True)
    return (e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-12)).astype(np.float32)


def run_dev(args, d):
    qs = [q for q in load_split(args.dataset, "validation") if q["level"] != "simple" and q["answer_type"] in TARGET
          and len(q["gold_fact_ids"]) >= 2 and min(q["gold_fact_ids"]) >= 0]
    voc = load_json(d / "vocab.json"); facts = np.load(d / "facts.npz")
    res = IntervalAnchorResolver(facts, voc["entities"], voc["predicates"], np.load(d / "fact_emb.npy", mmap_mode="r"))
    qv = encode([q["question"] for q in qs], args.threads)
    spo = lambda f: (int(res.S[f]), int(res.P[f]), int(res.O[f]))
    c = collections.Counter()
    for q, v in zip(qs, qv):
        x, ys = q["gold_fact_ids"][0], q["gold_fact_ids"][1:]
        labels = q["temporal_relation"].split("&")
        for choose in ("best", "any"):
            cons = res.resolve(q["question"], v, choose)
            for y, lab in zip(ys, labels):
                c[(choose, "gold_anchors")] += 1
                hit = [k for k in cons if k["anchors"] and spo(k["anchors"][0]) == spo(y)]
                c[(choose, "anchor_spo_found")] += bool(hit)
                c[(choose, "anchor_interval_found")] += any(y in k["anchors"] for k in hit)
                if hit and lab.startswith("X "):
                    k = hit[0]
                    c[(choose, "family_has_gold_relation")] += bool(k["relations"] and lab.split()[1] in k["relations"])
                    c[(choose, "family_checked")] += 1
            c[(choose, "questions")] += 1
            c[(choose, "constraints_parsed")] += len(cons) > 0
            w = res.window(cons, res.ST[[x]], res.EN[[x]])
            c[(choose, "window_built")] += w is not None
            c[(choose, "gold_answer_event_in_window")] += bool(w is not None and w[0])
            c[(choose, "gold_answer_relation_check")] += all(
                allen_of(int(res.ST[x]), int(res.EN[x]), int(res.ST[y]), int(res.EN[y])) == lab.split()[1]
                for y, lab in zip(ys, labels) if lab.startswith("X "))
    rows = [{"choose": k[0], "measure": k[1], "count": v} for k, v in sorted(c.items())]
    tab = pd.DataFrame(rows)
    base = {"anchor_spo_found": "gold_anchors", "anchor_interval_found": "gold_anchors",
            "family_has_gold_relation": "family_checked", "constraints_parsed": "questions",
            "window_built": "questions", "gold_answer_event_in_window": "questions", "gold_answer_relation_check": "questions"}
    tab["share"] = [r["count"] / c[(r["choose"], base[r["measure"]])] if r["measure"] in base else np.nan
                    for _, r in tab.iterrows()]
    tab.to_csv(RESULTS / f"{args.dataset}_grounding_dev.csv", index=False)
    print(f"{len(qs)} validation questions"); print(tab.round(4).to_string(index=False))


def slot_to_read(q, lab, read_slot):
    """Which value of the top-1 fact is the answer: the dataset's answer_type (oracle) or the labels' slot."""
    if read_slot == "dataset":
        return q["answer_type"]
    s = (lab or {}).get("answer_slot")
    if s in TARGET:
        return s
    if s == "timestamp":
        return "timestamp_end" if lab.get("time_end") else "timestamp_start"
    return None


def llmfree(perm, pool, F_ids, qs, slots=None, read_slot="dataset"):
    """Evaluated on the questions whose dataset answer is a slot value (subject / object / start / end date)."""
    facts = {"s_id": pool.s, "o_id": pool.o, "start_days": pool.start, "end_days": pool.end}
    from tear.labeling.interval_anchor import read_answer
    out = np.zeros(len(qs), bool)
    for i, q in enumerate(qs):
        if q["answer_type"] in SLOT_TYPES:
            at = slot_to_read(q, (slots or {}).get(pool.key(q)), read_slot)
            pred = read_answer(at, int(F_ids[i, perm[i, 0]]), facts, pool.ents) if at else None
            out[i] = pred is not None and normalize(pred) == normalize(q["answer"])
    return out


def make_rule_labels(args, d):
    """Rule v2 (tear/labeling/rules.py, written for MultiTQ) on the TimelineKGQA question text; accuracy against the
    dataset's answer_type and the oracle topic entity."""
    from tear.labeling import rules
    qs = load_json(d / "questions.json"); ents = load_json(d / "vocab.json")["entities"]
    mentions, sector = rules.build_mentioner(ents), rules.build_sector_matcher(ents)
    ner = load_json(d / "topic_ner_all.json") if (d / "topic_ner_all.json").exists() else {}
    oracle = load_json(d / "slots_oracle.json")
    labels, rows = {}, []
    with timed("S4_rule_slots", args.dataset, "v2"):
        for q in qs:
            k, text = str(q["qid"]), q["question"]
            slot = rules.rule_slot(text, "v2")
            topic, src = rules.rule_topic(text, mentions, ner.get(k, {}).get("topic_entity"), sector, "v2")
            lab = {"answer_slot": slot, "topic_entity": topic, "topic_source": src}
            if slot == "timestamp":
                lab["time_end"] = bool(END_CUE.search(text))
            labels[k] = lab
            if q["answer_type"] in SLOT_TYPES:
                gold = q["answer_type"] if q["answer_type"] in TARGET else "timestamp"
                rows.append({"level": q["level"], "answer_type": q["answer_type"], "slot_ok": slot == gold,
                             "read_ok": slot_to_read(q, lab, "labels") == q["answer_type"],
                             "topic_ok": topic == oracle.get(k, {}).get("topic_entity"), "pred": slot})
    save_json(labels, d / "slots_rule_v2.json")
    df = pd.DataFrame(rows)
    tab = df.groupby(["level", "answer_type"]).agg(n=("slot_ok", "size"), slot_acc=("slot_ok", "mean"),
                                                    slot_and_date_end_acc=("read_ok", "mean"),
                                                    topic_agree_oracle=("topic_ok", "mean")).reset_index()
    pred = df.groupby(["level", "answer_type"]).pred.value_counts().unstack(fill_value=0).reset_index()
    tab = tab.merge(pred, on=["level", "answer_type"])
    allrow = {"level": "all", "answer_type": "slot types", "n": len(df), "slot_acc": df.slot_ok.mean(),
              "slot_and_date_end_acc": df.read_ok.mean(), "topic_agree_oracle": df.topic_ok.mean()}
    tab = pd.concat([tab, pd.DataFrame([allrow])], ignore_index=True)
    tab.to_csv(RESULTS / f"{args.dataset}_rule_label_accuracy.csv", index=False)
    print(tab.round(3).to_string(index=False))


def run_test(args, d):
    pool = CandidatePool(args.dataset, 1000)
    slots = load_json(args.slots)
    sfx = "" if args.tag == "oracle_ground" else f"_{args.tag}"
    with timed("S5_features", args.dataset, args.tag):
        F = build_features(pool, slots, None, "full", args.atc_match)
        F.update(answer_hit_features(pool))
    res = IntervalAnchorResolver(np.load(d / "facts.npz"), pool.ents, pool.rels, np.load(d / "fact_emb.npy", mmap_mode="r"))
    qemb = np.load(d / "q_emb.npy", mmap_mode="r")
    Fg = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in F.items()}
    Fk = {"window": F["window"].copy(), "has_window": F["has_window"].copy()}   # variant: anchor facts kept
    anchor_ids = {}
    anchors, crow = {}, []
    with timed("S5_interval_grounding", args.dataset, args.tag):
        for qi, q in enumerate(pool.qs):
            cons = res.resolve(q["question"], np.asarray(qemb[qi]), args.choose)
            cand = pool.ids[qi]
            w = res.window(cons, pool.start[cand], pool.end[cand])
            slot = slots.get(pool.key(q), {}).get("answer_slot")
            n_anc = 0
            if w is not None and slot not in TARGET:
                w = None            # union / intersection / range questions: both events belong to the answer
            if w is not None:
                Fg["window"][qi] = F["window"][qi] & w if F["has_window"][qi] else w
                Fg["has_window"][qi] = True
                anchor_ids[qi] = sorted({f for c in cons for f in c.get("anchors", [])})
                Fk["window"][qi] = Fg["window"][qi] | np.isin(cand, anchor_ids[qi]); Fk["has_window"][qi] = True
                if args.anchor_atc and slot in TARGET:   # off by default: unlike MultiTQ, the answer is an anchor
                                                          # entity in 3.5% / 12.4% of validation questions
                    m = np.isin(pool.s[cand] if slot == "subject" else pool.o[cand], list(res.anchor_entities(cons)))
                    Fg["atc"][qi] |= m; n_anc = int(m.any())
            anchors[pool.key(q)] = [{k: (sorted(v) if isinstance(v, set) else v) for k, v in c.items()} for c in cons]
            crow.append({"level": q["level"], "qtype": q["qtype"], "answer_type": q["answer_type"],
                         "n_constraints": len(cons), "resolved": sum(bool(c["anchors"]) for c in cons),
                         "window": w is not None, "window_keeps": float(w.mean()) if w is not None else np.nan,
                         "anchor_atc": n_anc > 0})
    save_json(anchors, d / "interval_anchors.json")
    cr = pd.DataFrame(crow)
    cr["target"] = cr.level.ne("simple") & cr.answer_type.isin(TARGET)
    ctab = cr.groupby(["level", "answer_type"]).agg(n=("window", "size"), parsed=("n_constraints", lambda s: (s > 0).mean()),
                                                     window=("window", "mean"), keeps=("window_keeps", "mean"),
                                                     anchor_atc=("anchor_atc", "mean")).reset_index()
    ctab.to_csv(RESULTS / f"{args.dataset}_grounding_constraints{sfx}.csv", index=False)
    print(ctab.round(3).to_string(index=False))

    qdf = pd.DataFrame(pool.qs)
    grounded = cr.window.values
    groups = {"overall": pd.Series(["all"] * len(qdf)), "answer_type": qdf["answer_type"], "level": qdf["level"],
              "qtype": qdf["qtype"], "grounded": pd.Series(np.where(grounded, "grounded", "not_grounded"))}
    rows, hits, perms = [], {}, {}
    rdir = d / "rerank"
    for name in BASE:
        perm = rerank(name, pool.sims, F, args.lam1, args.lam2)
        perms[name] = perm
        if name == "none":           # the reader's baseline input under the same tag
            np.save(rdir / f"none_{args.tag}_top30_ids.npy", np.take_along_axis(pool.ids, perm[:, :30], axis=1).astype(np.int32))
        rows += coverage_rows(F, perm, name, groups, ks=[15])
    for name in GROUNDED:
        perm = rerank(name, pool.sims, Fg, args.lam1, args.lam2)
        perms[name + "+grounding"] = perm
        rows += coverage_rows(Fg, perm, name + "+grounding", groups, ks=[15])
        np.save(rdir / f"{name}_{args.tag}_top30_ids.npy", np.take_along_axis(pool.ids, perm[:, :30], axis=1).astype(np.int32))
    Fkeep = dict(Fg, **Fk)
    perm = rerank("tear_window", pool.sims, Fkeep, args.lam1, args.lam2)
    perms["tear_window+grounding_keep_anchors"] = perm
    rows += coverage_rows(Fkeep, perm, "tear_window+grounding_keep_anchors", groups, ks=[15])
    cov = pd.DataFrame(rows)
    # answer event (first gold event) vs anchor events (the others) in the top-15, grounded questions
    erows = []
    gq = [qi for qi in anchor_ids if pool.qs[qi]["gold_fact_ids"] and min(pool.qs[qi]["gold_fact_ids"]) >= 0]
    for name, perm in perms.items():
        top = np.take_along_axis(pool.ids, perm[:, :15], axis=1)
        ans = [pool.qs[qi]["gold_fact_ids"][0] in top[qi] for qi in gq]
        anc = [all(g in top[qi] for g in pool.qs[qi]["gold_fact_ids"][1:]) for qi in gq]
        erows.append({"reranker": name, "n_grounded": len(gq), "answer_event_recall15": float(np.mean(ans)),
                      "all_anchor_events_recall15": float(np.mean(anc))})
    pd.DataFrame(erows).to_csv(RESULTS / f"{args.dataset}_grounding_event_recall{sfx}.csv", index=False)
    print(pd.DataFrame(erows).round(4).to_string(index=False))
    cov.to_csv(RESULTS / f"{args.dataset}_grounding_coverage{sfx}.csv", index=False)
    slot_q = qdf.answer_type.isin(SLOT_TYPES).values
    hrows = []
    for name, perm in perms.items():
        h = llmfree(perm, pool, pool.ids, pool.qs, slots, args.read_slot); hits[name] = h
        for gtype, gser in (("overall", pd.Series(["all"] * len(qdf))), ("answer_type", qdf["answer_type"]),
                            ("level", qdf["level"]), ("grounded", groups["grounded"])):
            for gval in sorted(gser.unique()):
                m = (gser == gval).values & slot_q
                if m.any():
                    hrows.append({"reranker": name, "group_type": gtype, "group": gval, "n": int(m.sum()),
                                  "hit1": float(h[m].mean())})
    pd.DataFrame(hrows).to_csv(RESULTS / f"{args.dataset}_grounding_llmfree{sfx}.csv", index=False)
    sig = []
    cov15 = {name: np.take_along_axis((Fg if "+grounding" in name else F)["hit"], perms[name][:, :15], axis=1).any(1)
             for name in perms}
    for a, b in [("tear_window+grounding", "tear_window"), ("tear_window+grounding", "none"),
                 ("hardwindow+grounding", "hardwindow"), ("tear+grounding", "tear"), ("tear_window", "none"),
                 ("tear_window+grounding_keep_anchors", "tear_window")]:
        for metric, vals, mask in (("llmfree_hit1", hits, slot_q), ("coverage15", cov15, F["cov_defined"])):
            for subset, sm in (("all", mask), ("grounded", mask & grounded)):
                x, y = vals[b][sm], vals[a][sm]
                mc = mcnemar(x, y); bs = paired_bootstrap(x, y)
                sig.append({"run": a, "vs": b, "metric": metric, "subset": subset, "n": int(sm.sum()), "diff": bs["diff"],
                            "ci_low": bs["ci_low"], "ci_high": bs["ci_high"], "mcnemar_p": mc["p_value"],
                            "vs_wrong_run_right": mc["n01"], "vs_right_run_wrong": mc["n10"]})
    sig = pd.DataFrame(sig)
    sig.to_csv(RESULTS / f"{args.dataset}_grounding_significance{sfx}.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    piv = cov[(cov.group_type.isin(["overall", "level", "grounded"]))].pivot_table(index="reranker", columns="group",
                                                                                  values="coverage")
    print(piv.round(4).to_string())
    print(pd.DataFrame(hrows).pivot_table(index="reranker", columns="group", values="hit1").round(4).to_string())
    print(sig.round(4).to_string(index=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="icews", choices=["icews", "cron"])
    ap.add_argument("--slots", default=None, help="default cache/<ds>/slots_oracle.json (answer slot from answer_type)")
    ap.add_argument("--atc_match", choices=["substring", "exact"], default="exact")
    ap.add_argument("--choose", choices=["best", "any"], default="any",
                    help="several intervals of one anchor: any (validation choice) or the best-scored one")
    ap.add_argument("--tag", default="oracle_ground")
    ap.add_argument("--lam1", type=float, default=0.8)
    ap.add_argument("--lam2", type=float, default=0.8)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--anchor_atc", action="store_true", help="also penalise anchor entities in the answer slot")
    ap.add_argument("--read_slot", choices=["labels", "dataset"], default="labels",
                    help="LLM-free Hit@1 reads the labels' slot, or the dataset's answer_type (oracle; first run)")
    ap.add_argument("--make_rule_labels", action="store_true")
    ap.add_argument("--dev_design", action="store_true")
    args = ap.parse_args()
    d = CACHE / args.dataset
    args.slots = args.slots or str(d / "slots_oracle.json")
    if args.make_rule_labels:
        make_rule_labels(args, d)
    elif args.dev_design:
        run_dev(args, d)
    else:
        run_test(args, d)


if __name__ == "__main__":
    main()
