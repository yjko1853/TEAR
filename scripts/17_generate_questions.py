"""Stage Q: MultiTQ-style questions over another event KG (second benchmark: ICEWS18). CPU only, no model.

  extract    MultiTQ train + dev questions aligned to ICEWS05-15 -> frozen templates
             (resources/qgen/multitq_templates_v1.json.gz + .sha256; refuses to overwrite the frozen file)
  calibrate  generator v2 config from MultiTQ dev statistics (resources/qgen/generator_v2.json + .sha256)
  prepare    RE-NET ICEWS18 quadruples -> cache/icews18/{facts.npz, vocab.json, sentences.txt, kg_info.json}
  generate   questions from the frozen templates -> cache/<ds>/<out> in the MultiTQ question format of
             scripts/01 (rewrite = question, empty TimeR4 top-15) plus a 'gen' record (answer slot, topic, anchor and
             its date, date expression, gold fact ids, template)
  align-test MultiTQ *test* questions aligned with the frozen models (fidelity comparison only; run after extract)
  fidelity   question-form and KG-side statistics: MultiTQ test vs regenerated ICEWS05-15 vs ICEWS18
  rulecheck  TEAR's answer-slot / topic rules, date parser and anchor phrases against the generator's record
  difficulty (after logs/run_icews18.sh) coverage@15 / LLM-free Hit@1: MultiTQ vs regenerated ICEWS05-15 vs ICEWS18

  P=".venv/bin/python -I"
  $P scripts/17_generate_questions.py extract
  $P scripts/17_generate_questions.py calibrate        # generator v2 rules + time seed alpha (MultiTQ dev targets)
  $P scripts/17_generate_questions.py prepare
  $P scripts/17_generate_questions.py generate --dataset icews0515_regen --n_per 1000 --seed 2025
  $P scripts/17_generate_questions.py generate --dataset icews18 --n_per 1000 --seed 2026 --out questions.json
  $P scripts/17_generate_questions.py generate --dataset icews18 --n_per 200 --seed 2027 --out questions_dev.json \
      --exclude questions.json --quid_start 5100000
  $P scripts/17_generate_questions.py align-test
  $P scripts/17_generate_questions.py fidelity
  $P scripts/17_generate_questions.py rulecheck --dataset icews18
"""
import argparse
import collections
import datetime as dt
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tear.data.qgen import freeze  # noqa: E402
from tear.data.qgen.aliases import split_sector  # noqa: E402
from tear.data.qgen.generate import DEFAULT_CONFIG, Generator  # noqa: E402
from tear.data.qgen.kg import TemporalKG, prepare_renet  # noqa: E402
from tear.paths import CACHE, EXTERNAL, RESULTS, ROOT, ensure_dirs  # noqa: E402
from tear.utils.io import load_json, save_json  # noqa: E402
from tear.utils.timing import timed  # noqa: E402

FROZEN = ROOT / "resources/qgen/multitq_templates_v1.json.gz"
GEN_CONFIG = ROOT / "resources/qgen/generator_v2.json"
FREEZE_LOG = ROOT / "resources/qgen/FREEZE_LOG.md"
MTQ = EXTERNAL / "MultiTQ_hf"
MTQ_TEST = EXTERNAL / "TimeR4/datasets/MultiTQ/questions/test.json"
SHARED = ("facts.npz", "vocab.json", "sentences.txt", "fact_emb.npy")
QTYPES = ("after_first", "before_after", "before_last", "equal", "equal_multi", "first_last")
QUID_START = {"icews18": 5000000, "icews0515_regen": 7000000}


def log(msg):
    print(f"[{dt.datetime.now():%H:%M:%S}] {msg}", flush=True)


def rel(p) -> str:
    p = Path(p).resolve()
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


# ---------------------------------------------------------------------- extract
def cmd_extract(args):
    out = Path(args.out)
    if out.exists() and not args.force:
        raise SystemExit(f"{out} exists (frozen); refusing to overwrite (use --force only for a new version)")
    train, dev = load_json(Path(args.train)), load_json(Path(args.dev))
    kg_dir = CACHE / "multitq"
    kg = TemporalKG.from_cache(kg_dir)
    sources = {"train": {"path": rel(args.train), "sha256": freeze.sha256(args.train), "n": len(train)},
               "dev": {"path": rel(args.dev), "sha256": freeze.sha256(args.dev), "n": len(dev)},
               "kg_facts": {"path": "cache/multitq/facts.npz", "sha256": freeze.sha256(kg_dir / "facts.npz")},
               "kg_vocab": {"path": "cache/multitq/vocab.json", "sha256": freeze.sha256(kg_dir / "vocab.json")},
               "code": {p.name: freeze.sha256(p) for p in sorted((ROOT / "tear/data/qgen").glob("*.py"))}}
    with timed("SQ_extract", "multitq", f"train={len(train)} dev={len(dev)}"):
        frozen, rows = freeze.build_frozen(train, dev, kg, sources, log)
    digest = freeze.save_frozen(frozen, out)
    rows_path = (CACHE / "qgen" / "multitq_traindev_alignment.json" if out.resolve() == FROZEN.resolve()
                 else out.parent / "traindev_alignment.json")
    save_json(rows, rows_path)
    n_t = len(frozen["templates"]); n_u = sum(t["usable"] for t in frozen["templates"])
    entry = (f"\n## {frozen['version']} — {frozen['created']}\n\n"
             f"- file: `{rel(out)}`\n- SHA-256: `{digest}`\n"
             f"- sources: MultiTQ train ({len(train)} questions, sha256 {sources['train']['sha256'][:16]}...), "
             f"dev ({len(dev)}, {sources['dev']['sha256'][:16]}...); KG cache/multitq/facts.npz "
             f"({sources['kg_facts']['sha256'][:16]}...). The MultiTQ test split was not read.\n"
             f"- templates: {n_t} ({n_u} usable), relations: {len(frozen['relations'])}, "
             f"demonyms: {len(frozen['demonyms'])}\n"
             f"- frozen before any ICEWS18 question was generated and before any retrieval, reranking or reader "
             f"result on generated questions existed.\n")
    log_path = FREEZE_LOG if out.resolve() == FROZEN.resolve() else out.parent / "FREEZE_LOG.md"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if not log_path.exists():
        log_path.write_text("# Frozen MultiTQ templates\n\nEach entry is written by `scripts/17_generate_questions.py "
                              "extract`; `load_frozen` checks the SHA-256 before generation.\n")
    with open(log_path, "a") as f:
        f.write(entry)
    # summaries
    st = pd.DataFrame(rows)
    tab = st.groupby(["split", "qtype", "answer_type", "status"]).size().unstack(fill_value=0)
    rd = RESULTS if out.resolve() == FROZEN.resolve() else out.parent
    tab.to_csv(rd / "qgen_alignment_traindev.csv")
    pd.DataFrame([{"relation": r, **v} for r, v in frozen["relation_selection"]["dev_alignment"].items()]).to_csv(
        rd / "qgen_relation_selection.csv", index=False)
    pd.DataFrame([{"relation": r, "phrases": " | ".join(p)} for r, p in frozen["relation_phrases"].items()]).to_csv(
        rd / "qgen_relation_phrases.csv", index=False)
    log(f"wrote {out} sha256={digest}; templates {n_t} (usable {n_u})")
    print(tab.to_string())


# ---------------------------------------------------------------------- prepare
def cmd_prepare(args):
    with timed("SQ_prepare", "icews18"):
        info = prepare_renet(Path(args.raw), CACHE / args.dataset, dt.date.fromisoformat(args.start))
    print(info)


# ---------------------------------------------------------------------- calibrate (generator v2)
def dev_targets(rows, dev) -> dict:
    """MultiTQ dev regularities behind generator v2 (aligned dev questions for KG-side properties, all dev
    questions for answer counts)."""
    rows = [r for r in rows if r["split"] == "dev"]
    anch = [r for r in rows if r.get("anchor_n_days") is not None]
    fl = [r for r in rows if r["qtype"] == "first_last" and r["answer_type"] == "time" and r.get("n_periods")]
    by_lvl = collections.defaultdict(list)
    for r in fl:
        by_lvl[r["time_level"]].append(r["n_periods"] == 1)
    uniq = collections.defaultdict(list)
    for q in dev:
        uniq[f"{q['qtype']}|{freeze.text_variant(q)}|{q['answer_type']}"].append(len(q["answers"]) == 1)
    sec = lambda e: bool(split_sector(e)) if e else False  # noqa: E731
    sig = collections.defaultdict(collections.Counter)
    for r in rows:
        if not r.get("relation"):
            continue
        if r["side"] == "timestamp":
            sig["time"]["TF"[not sec(r["subject"])] + "TF"[not sec(r["object"])]] += 1
        else:
            k = "entity|anchor" if r.get("anchor") else "entity|plain"
            sig[k]["TF"[not sec(r["topic"])] + ("TF"[not sec(r["anchor"])] if r.get("anchor") else "")] += 1
    role_sig = {k: {s_: n / sum(c.values()) for s_, n in sorted(c.items())} for k, c in sorted(sig.items())}
    return {"role_signatures": role_sig, "role_signature_counts": {k: dict(c) for k, c in sig.items()},
            "anchor_single_day_share": sum(r["anchor_n_days"] == 1 for r in anch) / len(anch), "n_anchored": len(anch),
            "first_last_time_single_period_share": sum(r["n_periods"] == 1 for r in fl) / len(fl),
            "first_last_time_single_period_by_level": {k: sum(v) / len(v) for k, v in sorted(by_lvl.items())},
            "n_first_last_time": len(fl),
            "single_answer_share": {k: round(sum(v) / len(v), 4) for k, v in sorted(uniq.items())}}


def cmd_calibrate(args):
    if GEN_CONFIG.exists() and not args.force:
        raise SystemExit(f"{GEN_CONFIG} exists (frozen); use --force only for a new version")
    frozen = freeze.load_frozen(args.templates)
    rows = load_json(CACHE / "qgen" / "multitq_traindev_alignment.json")
    dev = load_json(MTQ / "dev.json")
    tg = dev_targets(rows, dev)
    kg = TemporalKG.from_cache(CACHE / "multitq")
    grid = []
    for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
        g = Generator(kg, frozen, seed=args.seed, config={"time_seed_alpha": alpha,
                                                          "role_signatures": tg["role_signatures"]})
        qs = g.generate({"first_last": args.n}, 0)
        tq = [q for q in qs if q["answer_type"] == "time"]
        share = sum(q["gen"]["n_periods"] == 1 for q in tq) / len(tq)
        by = collections.defaultdict(list)
        for q in tq:
            by[q["time_level"]].append(q["gen"]["n_periods"] == 1)
        grid.append({"alpha": alpha, "single_period_share": share, "n": len(tq),
                     "by_level": {k: sum(v) / len(v) for k, v in sorted(by.items())}})
        log(f"alpha={alpha}: single-period share {share:.3f} (target {tg['first_last_time_single_period_share']:.3f})")
    best = min(grid, key=lambda x: abs(x["single_period_share"] - tg["first_last_time_single_period_share"]))
    cfg = dict(DEFAULT_CONFIG, time_seed_alpha=best["alpha"], role_signatures=tg["role_signatures"])
    out = {"config": cfg, "dev_targets": tg, "alpha_grid": grid,
           "note": "Calibrated on ICEWS05-15 against MultiTQ dev only (the test split is used afterwards for the "
                   "fidelity comparison); fixed before any retrieval, reranking or reader run on generated questions.",
           "templates_sha256": freeze.sha256(args.templates),
           "generator_code_sha256": freeze.sha256(ROOT / "tear/data/qgen/generate.py"),
           "created": dt.datetime.now().isoformat(timespec="seconds")}
    save_json(out, GEN_CONFIG)
    digest = freeze.sha256(GEN_CONFIG)
    Path(str(GEN_CONFIG) + ".sha256").write_text(f"{digest}  {GEN_CONFIG.name}\n")
    with open(FREEZE_LOG, "a") as f:
        f.write(f"\n## generator v2 config — {out['created']}\n\n- file: `{rel(GEN_CONFIG)}`\n- SHA-256: `{digest}`\n"
                f"- rules from MultiTQ dev: single answer for {', '.join('/'.join(x) for x in cfg['unique_answer'])} "
                f"(dev shares {tg['single_answer_share']}); anchor events on one day (dev "
                f"{tg['anchor_single_day_share']:.3f} of {tg['n_anchored']} aligned anchored questions); time seed "
                f"alpha = {cfg['time_seed_alpha']} (first/last time questions in one period: dev "
                f"{tg['first_last_time_single_period_share']:.3f}, regenerated "
                f"{best['single_period_share']:.3f}); role signatures (sector / other entity per named role) "
                f"{tg['role_signatures']}.\n- fixed before any retrieval, reranking or reader run on "
                f"generated questions.\n")
    print(json.dumps(out, indent=1)[:3000])


def load_config(path):
    p = Path(path)
    if not p.exists():
        return None, None
    ref = Path(str(p) + ".sha256")
    digest = freeze.sha256(p)
    if ref.exists() and ref.read_text().split()[0] != digest:
        raise SystemExit(f"{p}: SHA-256 differs from its frozen record")
    return load_json(p)["config"], digest


# ---------------------------------------------------------------------- generate
def ensure_regen_dir(d: Path, src: Path):
    """cache/icews0515_regen: the ICEWS05-15 fact files are relative symlinks into cache/multitq (never written)."""
    d.mkdir(parents=True, exist_ok=True)
    for name in SHARED:
        link = d / name
        if (src / name).exists() and not link.is_symlink() and not link.exists():
            link.symlink_to(os.path.relpath(src / name, d.resolve()))


def to_pipeline(q: dict) -> dict:
    """MultiTQ question format of scripts/01 (rewrite = the question; no TimeR4 top-15) + the generator record."""
    return {"quid": q["quid"], "question": q["question"], "rewrite": q["question"], "answers": q["answers"],
            "answer_type": q["answer_type"], "qtype": q["qtype"], "qlabel": q["qlabel"],
            "time_level": q["time_level"], "timer4_top15_ids": [], "timer4_top15_raw": [], "gen": q["gen"]}


def cmd_generate(args):
    d = CACHE / args.dataset
    if args.dataset == "icews0515_regen":
        ensure_regen_dir(d, CACHE / "multitq")
    out = d / args.out
    if out.exists() and not args.force:
        raise SystemExit(f"{out} exists; use --force to regenerate")
    frozen = freeze.load_frozen(args.templates)
    cfg, cfg_sha = load_config(args.config)
    kg = TemporalKG.from_cache(d)
    gen = Generator(kg, frozen, seed=args.seed, config=cfg)
    exclude = []
    if args.exclude:
        exclude = [q["question"] for q in load_json(d / args.exclude)]
    n_per = {qt: args.n_per for qt in QTYPES}
    start = args.quid_start if args.quid_start is not None else QUID_START.get(args.dataset, 9000000)
    with timed("SQ_generate", args.dataset, f"n_per={args.n_per} seed={args.seed} out={args.out}"):
        qs = gen.generate(n_per, start, exclude)
    save_json([to_pipeline(q) for q in qs], out)
    meta = {"dataset": args.dataset, "out": args.out, "n_per_type": args.n_per, "seed": args.seed,
            "quid_start": start, "templates": rel(args.templates),
            "templates_sha256": freeze.sha256(args.templates), "excluded_texts_from": args.exclude,
            "generator_config": gen.config, "generator_config_file": rel(args.config) if cfg else None,
            "generator_config_sha256": cfg_sha, "generator_code_sha256": freeze.sha256(ROOT / "tear/data/qgen/generate.py"),
            "rejections": dict(gen.rejects), "adjective_form_fallbacks": dict(gen.forms.fallback),
            "degenerate_time_levels": gen.degenerate_levels, "excluded_buckets": gen.excluded_buckets,
            "bucket_shares_used": {qt: {"|".join(k): w for k, w in zip(*gen.bucket_p[qt])} for qt in gen.bucket_p},
            "created": dt.datetime.now().isoformat(timespec="seconds"), "questions_sha256": freeze.sha256(out)}
    save_json(meta, out.with_suffix(".meta.json"))
    log(f"wrote {out}: {len(qs)} questions; rejections {dict(gen.rejects)}")
    c = collections.Counter((q["qtype"], q["answer_type"], q["time_level"], q["gen"]["variant"]) for q in qs)
    for k, v in sorted(c.items()):
        print(k, v)


# ---------------------------------------------------------------------- align-test
def cmd_align_test(args):
    frozen = freeze.load_frozen(args.templates)
    kg = TemporalKG.from_cache(CACHE / "multitq")
    test = load_json(Path(args.questions))
    with timed("SQ_align_test", "multitq", f"n={len(test)}"):
        rows = freeze.align_with_frozen(test, kg, frozen, "test", log)
    out = CACHE / "qgen" / "multitq_test_alignment.json"
    save_json(rows, out)
    st = pd.DataFrame(rows).groupby(["qtype", "answer_type", "status"]).size().unstack(fill_value=0)
    st.to_csv(RESULTS / "qgen_alignment_test.csv")
    print(st.to_string())


# ---------------------------------------------------------------------- fidelity
def cmd_fidelity(args):
    from tear.analysis import qgen_fidelity as F
    frozen = freeze.load_frozen(args.templates)
    test = load_json(Path(args.questions))
    sets = {"multitq_test": test}
    for name, path in (("icews0515_regen", CACHE / "icews0515_regen/questions.json"),
                       ("icews18_test", CACHE / "icews18/questions.json"),
                       ("icews18_dev", CACHE / "icews18/questions_dev.json")):
        if path.exists():
            sets[name] = load_json(path)
    weights = collections.Counter(q["qtype"] for q in test)
    text_rows, kg_rows, extra = [], [], []
    shapes = []
    kgs = {"multitq": TemporalKG.from_cache(CACHE / "multitq")}
    if (CACHE / "icews18/facts.npz").exists():
        kgs["icews18"] = TemporalKG.from_cache(CACHE / "icews18")
    for name, qs in sets.items():
        tdf = F.text_table(qs)
        s = F.summarise_text(tdf).assign(set=name)
        text_rows.append(s)
        rw = F.reweight(s, weights)
        text_rows.append(pd.DataFrame([dict(rw, qtype="all_reweighted_to_multitq_test", set=name,
                                            n=len(qs))]))
        shapes.append(F.date_shape_share(tdf).assign(set=name))
        kg = kgs["icews18" if name.startswith("icews18") else "multitq"]
        if name == "multitq_test":
            p = CACHE / "qgen" / "multitq_test_alignment.json"
            if not p.exists():
                raise SystemExit("run align-test first")
            rows = load_json(p)
            kdf = F.kg_rows_aligned(qs, rows, kg)
            aligned = sum(r.get("relation") is not None for r in rows) / len(rows)
            extra.append({"set": name, "aligned_share": aligned})
        else:
            kdf = F.kg_rows_generated(qs, kg)
        kg_rows.append(F.summarise_kg(kdf).assign(set=name))
        tc = F.topic_concentration(kdf.topic.tolist())
        extra.append({"set": name, **{f"topic_{k}": v for k, v in tc.items() if k != "top1"},
                      "topic_top1": str(tc["top1"])})
    T = pd.concat(text_rows, ignore_index=True)
    K = pd.concat(kg_rows, ignore_index=True)
    T.to_csv(RESULTS / "qgen_fidelity_text.csv", index=False)
    K.to_csv(RESULTS / "qgen_fidelity_kg.csv", index=False)
    pd.concat(shapes, ignore_index=True).to_csv(RESULTS / "qgen_fidelity_date_shapes.csv", index=False)
    kgsum = [dict(F.kg_summary(k), kg=n, scope="all relations") for n, k in kgs.items()]
    kgsum += [dict(F.kg_summary(k, frozen["relations"]), kg=n, scope="22 MultiTQ relations") for n, k in kgs.items()]
    pd.DataFrame(kgsum).to_csv(RESULTS / "qgen_kg_summary.csv", index=False)
    pd.DataFrame(extra).groupby("set").first().to_csv(RESULTS / "qgen_fidelity_topics.csv")
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    print(T.round(3).to_string(index=False)); print(K.round(3).to_string(index=False))
    print(pd.DataFrame(kgsum).to_string(index=False)); print(pd.DataFrame(extra).to_string(index=False))


# ---------------------------------------------------------------------- rulecheck
def cmd_rulecheck(args):
    from tear.analysis import qgen_fidelity as F
    if args.dataset == "multitq_test":       # reference: rule v2 vs the aligned answer side of MultiTQ test questions
        test = load_json(MTQ_TEST)
        tab = F.rule_vs_alignment(test, load_json(CACHE / "qgen" / "multitq_test_alignment.json"),
                                  load_json(CACHE / "multitq" / "vocab.json")["entities"])
        tab.to_csv(RESULTS / "multitq_test_qgen_rulecheck_reference.csv", index=False)
        print(tab.round(4).to_string(index=False))
        return
    d = CACHE / args.dataset
    qs = load_json(d / args.questions)
    voc = load_json(d / "vocab.json")
    facts = np.load(d / "facts.npz")
    labels = load_json(args.slots) if args.slots else None
    anchors = load_json(args.anchors) if args.anchors else None
    res = F.rule_check(qs, voc, facts, labels, anchors)
    for name, df in res.items():
        df.to_csv(RESULTS / f"{args.dataset}{args.suffix}_qgen_rulecheck_{name}.csv", index=False)
        print(f"--- {name}"); print(df.round(4).to_string(index=False))


# ---------------------------------------------------------------------- difficulty (after the GPU stages)
def cmd_difficulty(args):
    """coverage@15 and LLM-free Hit@1 per qtype: original MultiTQ test (multitq_orig, original questions) vs the
    regenerated ICEWS05-15 set vs ICEWS18, for the runs with and without anchor grounding."""
    rows = []
    for ds in ("multitq_orig", "icews0515_regen", "icews18"):
        for tag in (args.base_tag, args.tag):
            cp = RESULTS / f"{ds}_coverage_main_{tag}.csv"
            hp = RESULTS / f"{ds}_llmfree_hit1_{tag}.csv"
            if not cp.exists():
                print("missing", cp); continue
            c = pd.read_csv(cp)
            c = c[(c.k == 15) & c.group_type.isin(["overall", "qtype"])]
            for _, r in c.iterrows():
                rows.append({"dataset": ds, "tag": tag, "reranker": r.reranker, "group": r.group,
                             "metric": "coverage@15", "value": r.coverage, "n": r.n})
            if hp.exists():
                h = pd.read_csv(hp)
                for _, r in h.iterrows():
                    rows.append({"dataset": ds, "tag": tag, "reranker": r.reranker, "group": "all",
                                 "metric": "llmfree_hit1", "value": r.exact_all, "n": np.nan})
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "qgen_difficulty.csv", index=False)
    if len(df):
        piv = df[df.reranker.isin(["none", "timer4", "tear", "tear_window"])].pivot_table(
            index=["metric", "tag", "group"], columns=["dataset", "reranker"], values="value")
        pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
        print(piv.round(3).to_string())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("extract")
    a.add_argument("--train", default=str(MTQ / "train.json"))
    a.add_argument("--dev", default=str(MTQ / "dev.json"))
    a.add_argument("--out", default=str(FROZEN))
    a.add_argument("--force", action="store_true")
    a = sub.add_parser("prepare")
    a.add_argument("--raw", default=str(EXTERNAL / "ICEWS18_RE-Net"))
    a.add_argument("--dataset", default="icews18")
    a.add_argument("--start", default="2018-01-01")
    a = sub.add_parser("generate")
    a.add_argument("--dataset", required=True, help="icews18 (cache/icews18 prepared) or icews0515_regen (ICEWS05-15)")
    a.add_argument("--templates", default=str(FROZEN))
    a.add_argument("--n_per", type=int, default=1000)
    a.add_argument("--seed", type=int, default=2026)
    a.add_argument("--out", default="questions.json")
    a.add_argument("--exclude", default=None, help="question file in the same directory whose texts must not recur")
    a.add_argument("--config", default=str(GEN_CONFIG), help="generator config (default: the frozen v2 file)")
    a.add_argument("--quid_start", type=int, default=None)
    a.add_argument("--force", action="store_true")
    a = sub.add_parser("calibrate")
    a.add_argument("--templates", default=str(FROZEN))
    a.add_argument("--n", type=int, default=3000, help="first_last questions per alpha")
    a.add_argument("--seed", type=int, default=11)
    a.add_argument("--force", action="store_true")
    for name in ("align-test", "fidelity"):
        a = sub.add_parser(name)
        a.add_argument("--templates", default=str(FROZEN))
        a.add_argument("--questions", default=str(MTQ_TEST))
    a = sub.add_parser("rulecheck")
    a.add_argument("--dataset", default="icews18", help="icews18 | icews0515_regen | multitq_test (reference)")
    a.add_argument("--questions", default="questions.json")
    a.add_argument("--suffix", default="", help="output name suffix (e.g. _dev)")
    a.add_argument("--slots", default=None, help="labels of scripts/04 (default: rule v2 here, no NER fallback)")
    a.add_argument("--anchors", default=None, help="anchors.json of scripts/14 (default: mention-only lookup)")
    a = sub.add_parser("difficulty")
    a.add_argument("--tag", default="rule_v2_ground_exact_anydate")
    a.add_argument("--base_tag", default="rule_v2_exact_anydate", help="tag of the runs without anchor grounding")
    args = ap.parse_args()
    ensure_dirs()
    {"extract": cmd_extract, "prepare": cmd_prepare, "generate": cmd_generate, "align-test": cmd_align_test,
     "fidelity": cmd_fidelity, "rulecheck": cmd_rulecheck, "difficulty": cmd_difficulty,
     "calibrate": cmd_calibrate}[args.cmd](args)


if __name__ == "__main__":
    main()
