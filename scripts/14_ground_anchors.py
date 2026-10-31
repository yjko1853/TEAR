"""Stage 3b: event-anchored constraints -> reference dates looked up in the KG (tear/labeling/anchor.py). CPU only.

  --dataset multitq_orig --slots cache/multitq_orig/slots_rule_v2.json
      -> cache/<ds>/anchors.json ({quid: anchor type, phrase, status, anchor entity, relation, date, ...})
      -> results/<ds>_anchor_resolution.csv: per qtype and anchor type, anchored questions, resolved share and, when the
         questions carry TimeR4's rewrite ('rewrite_timer4'), agreement of the date with the date TimeR4 wrote in
         (day / month / year; empty or truncated rewrites excluded)
  --dev_design
      reproduces the design check on the MultiTQ dev split (600 questions per multi-constraint type, seed 13, rule-v2
      labels): share of anchored questions whose resolved constraint yields a gold answer in the KG, for the settings
      that were compared -> results/anchor_dev_design.csv
Encodes the entity names (cached as cache/<ds>/entity_emb.npy) and the anchor phrases with all-mpnet-base-v2 on CPU;
questions are scored with the stage's own question embeddings (cache/<ds>/q_emb.npy).
"""
import argparse
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tear.data.multitq import DEFAULT_TEST, rewrite_is_valid, stage_query  # noqa: E402
from tear.labeling import rules  # noqa: E402
from tear.labeling.anchor import AnchorResolver, anchor_phrase  # noqa: E402
from tear.paths import CACHE, RESULTS, ensure_dirs  # noqa: E402
from tear.temporal import date_to_days, extract_temporal_cue  # noqa: E402
from tear.utils.io import load_json, save_json  # noqa: E402
from tear.utils.timing import timed  # noqa: E402

DATE = re.compile(r"\b(\d{4}-\d{2}-\d{2}|\d{4}-\d{2}|\d{4})\b")
MULTI = ("after_first", "before_last", "equal_multi")


def encode(model, texts, batch=64):
    e = model.encode(list(texts), batch_size=batch, show_progress_bar=False, convert_to_numpy=True)
    return (e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-12)).astype(np.float32)


def load_model(device, threads):
    import torch
    torch.set_num_threads(threads)
    from tear.retrieval.encoder import load_encoder
    return load_encoder(device)[0]


def entity_embeddings(path: Path, entities, model):
    if path.exists():
        return np.load(path)
    emb = encode(model, entities, 128)
    np.save(path, emb)
    return emb


def timer4_date(q: dict) -> str | None:
    """The date TimeR4's rewrite adds to the question (first date in the rewrite that the original does not contain)."""
    rw = q.get("rewrite_timer4")
    if not rw or not rewrite_is_valid(rw, q["question"]):
        return None
    orig = set(DATE.findall(q["question"]))
    new = [d for d in DATE.findall(rw) if d not in orig]
    return new[0] if new else None


def agree(ours: str | None, ref: str | None, n: int) -> bool:
    return bool(ours and ref and len(ref) >= n and ours[:n] == ref[:n])


def run_test(args, d, model):
    qs = load_json(d / args.questions)
    slots = load_json(args.slots)
    voc = load_json(d / "vocab.json")
    facts = np.load(d / "facts.npz")
    qemb = np.load(d / args.qemb, mmap_mode="r")
    assert qemb.shape[0] == len(qs), (qemb.shape, len(qs))
    ent_emb = entity_embeddings(d / "entity_emb.npy", voc["entities"], model)
    res = AnchorResolver(facts, voc["entities"], voc["relations"], ent_emb, np.load(d / "fact_emb.npy", mmap_mode="r"),
                         tau=args.tau)
    items = [(i, q, anchor_phrase(stage_query(q))) for i, q in enumerate(qs)]
    items = [(i, q, p) for i, q, p in items if p]
    with timed("S4b_anchor_phrases", args.dataset, f"n={len(items)}"):
        xemb = encode(model, [p[1] for _, _, p in items])
    out = {}
    with timed("S4b_anchor_resolve", args.dataset, f"n={len(items)} tau={args.tau}"):
        for (i, q, (typ, phrase)), xv in zip(items, xemb):
            lab = slots.get(str(q["quid"]), {})
            r = res.resolve(typ, phrase, lab.get("answer_slot"), lab.get("topic_entity"), np.asarray(qemb[i]), xv)
            r.update(qtype=q["qtype"], timer4_date=timer4_date(q))
            out[str(q["quid"])] = r
    save_json(out, Path(args.out) if args.out else d / "anchors.json")
    rows = []
    df = pd.DataFrame(out.values())
    n_type = pd.Series([q["qtype"] for q in qs]).value_counts()
    for (qt, typ), g in df.groupby(["qtype", "type"]):
        ok = g[g.status == "resolved"]
        ref = g[g.timer4_date.notna()]
        both = ok[ok.timer4_date.notna()]
        rows.append({"qtype": qt, "anchor_type": typ, "n_qtype": int(n_type[qt]), "n_anchored": len(g),
                     "resolved": len(ok) / len(g), "by_mention": (ok.anchor_source == "mention").mean() if len(ok) else 0,
                     "n_with_timer4_date": len(ref), "resolved_with_timer4_date": len(both),
                     **{f"agree_{u}": (both.apply(lambda r: agree(r.date, r.timer4_date, n), axis=1).mean()
                                       if len(both) else np.nan) for u, n in (("day", 10), ("month", 7), ("year", 4))},
                     **{f"status_{s}": int((g.status == s).sum()) for s in
                        ("resolved", "no_anchor_entity", "anchor_is_topic", "no_topic")}})
    tab = pd.DataFrame(rows)
    tab.to_csv(RESULTS / f"{args.dataset}_anchor_resolution.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(tab.round(3).to_string(index=False))


def run_dev_design(args, d, model):
    dev = load_json(Path(DEFAULT_TEST).with_name("dev.json"))
    rng = random.Random(13)
    qs = []
    for qt in MULTI:
        qs += rng.sample([q for q in dev if q["qtype"] == qt], 600)
    for q in qs:
        q["rewrite"] = q["question"]
    voc = load_json(d / "vocab.json")
    labels = rules.label_multitq(qs, voc["entities"], None, "v2")
    facts = np.load(d / "facts.npz")
    fact_emb = np.load(d / "fact_emb.npy", mmap_mode="r")
    ent_emb = entity_embeddings(d / "entity_emb.npy", voc["entities"], model)
    qemb = encode(model, [q["question"] for q in qs])
    phr = [anchor_phrase(q["question"]) for q in qs]
    xemb = encode(model, [p[1] if p else "" for p in phr])
    # final setting first; then one change at a time
    settings = [("final: tau=0.5, widest, no guard", 0.5, "widest", False), ("topic guard", 0.5, "widest", True)] + \
               [(f"tau={t}", t, "widest", False) for t in (1.0, 0.7, 0.6, 0.4)] + \
               [(f"date={r}", 0.5, r, False) for r in ("earliest", "latest", "best")]
    rows = []
    for name, tau, rule, guard in settings:
        res = AnchorResolver(facts, voc["entities"], voc["relations"], ent_emb, fact_emb, tau, rule, guard)
        cnt = {qt: [0, 0, 0] for qt in MULTI}
        for q, p, qv, xv in zip(qs, phr, qemb, xemb):
            if not p:
                continue
            lab = labels[str(q["quid"])]
            r = res.resolve(p[0], p[1], lab["answer_slot"], lab["topic_entity"], qv, xv)
            c = cnt[q["qtype"]]; c[0] += 1
            if r["status"] == "resolved":
                c[1] += 1
                pred = res.implied_answers(lab["answer_slot"], lab["topic_entity"], r["relation"], p[0],
                                           date_to_days(r["date"]), extract_temporal_cue(q["question"]))
                c[2] += bool(pred & set(q["answers"]))
        for qt, (n, ok, cons) in cnt.items():
            rows.append({"setting": name, "qtype": qt, "n_sample": 600, "n_anchored": n, "resolved": ok,
                         "gold_consistent": cons, "gold_consistent_share": cons / n if n else np.nan})
    tab = pd.DataFrame(rows)
    tab.to_csv(RESULTS / "anchor_dev_design.csv", index=False)
    print(tab.pivot(index="setting", columns="qtype", values="gold_consistent").to_string())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="multitq_orig")
    ap.add_argument("--questions", default="questions.json")
    ap.add_argument("--slots", default=None, help="rule labels (default cache/<ds>/slots_rule_v2.json)")
    ap.add_argument("--qemb", default="q_emb.npy", help="question embeddings in cache/<ds>/ (rows = questions)")
    ap.add_argument("--tau", type=float, default=0.5, help="name-embedding threshold for described anchors")
    ap.add_argument("--out", default=None, help="default cache/<ds>/anchors.json")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--dev_design", action="store_true")
    args = ap.parse_args()
    ensure_dirs()
    d = CACHE / args.dataset
    args.slots = args.slots or str(d / "slots_rule_v2.json")
    model = load_model(args.device, args.threads)
    run_dev_design(args, d, model) if args.dev_design else run_test(args, d, model)


if __name__ == "__main__":
    main()
