"""Stage 3: answer-slot / topic-entity labels for the ATC penalty.

  --method rule    rule-based classifier (default; MultiTQ)    -> cache/<ds>/slots_rule.json
  --method ner     port of the original NER topic extractor     -> cache/<ds>/topic_ner_<labels>.json (+ agreement)
  --method oracle  KG-derived labels, diagnostic only           -> cache/<ds>/slots_oracle.json (+ derivation table)
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tear.data import is_multitq  # noqa: E402
from tear.data.multitq import stage_query  # noqa: E402
from tear.labeling import ner, oracle, rules  # noqa: E402
from tear.paths import CACHE, RESULTS, ensure_dirs  # noqa: E402
from tear.utils.device import get_device  # noqa: E402
from tear.utils.io import load_json, save_json  # noqa: E402
from tear.utils.timing import timed  # noqa: E402


def run_rule(args, d):
    if not is_multitq(args.dataset):
        raise SystemExit("the rule classifier targets MultiTQ questions; TimelineKGQA ships answer-type labels")
    ner_path = Path(args.ner_topics) if args.ner_topics else d / "topic_ner_all.json"
    ner_topics = load_json(ner_path) if ner_path.exists() else None
    with timed("S4_rule_slots", args.dataset, f"{args.rule_version} {args.questions}"):
        labels = rules.label_multitq(load_json(d / args.questions), load_json(d / "vocab.json")["entities"],
                                     ner_topics, args.rule_version)
    out = Path(args.out) if args.out else d / "slots_rule.json"
    save_json(labels, out)
    print("wrote", out, rules.summarize(labels))


def run_oracle(args, d, report_dir):
    with timed("S4_oracle_slots", args.dataset):
        if is_multitq(args.dataset):
            labels, tab = oracle.multitq_oracle(d)
        else:
            labels, tab = oracle.timeline_oracle(d)
    out = Path(args.out) if args.out else d / "slots_oracle.json"
    save_json(labels, out)
    tab.to_csv(report_dir / f"{args.dataset}_oracle_slot_derivation.csv", index=False)
    print("wrote", out)


def run_ner(args, d, report_dir):
    qs = load_json(d / args.questions)
    key = "quid" if is_multitq(args.dataset) else "qid"
    texts = [stage_query(q) for q in qs] if is_multitq(args.dataset) else [q["question"] for q in qs]
    entities = load_json(d / "vocab.json")["entities"]
    spans_path = d / f"ner_spans_{args.labels}{args.suffix}.json"    # raw spans are cached per label mode / version
    if spans_path.exists():
        spans = load_json(spans_path)
    else:
        device = get_device(args.device)
        if args.torch_threads:
            import torch
            torch.set_num_threads(args.torch_threads)
        todo = list(range(len(texts)))
        spans = [None] * len(texts)
        base_path = d / f"ner_spans_{args.labels}.json"
        if args.suffix and base_path.exists():   # reuse spans of questions whose text is unchanged
            base_qs = load_json(d / "questions.json")
            base_texts = [stage_query(q) for q in base_qs] if is_multitq(args.dataset) else [q["question"] for q in base_qs]
            base_spans = load_json(base_path)
            todo = [i for i in todo if texts[i] != base_texts[i]]
            for i in set(range(len(texts))) - set(todo):
                spans[i] = base_spans[i]
        with timed("S4_ner", args.dataset, f"labels={args.labels} device={device} n={len(todo)}"):
            new = ner.run_ner([texts[i] for i in todo], device, ner.LABEL_IDS[args.labels]) if todo else []
        for i, sp in zip(todo, new):
            spans[i] = sp
        print(f"NER run on {len(todo)} questions, {len(texts) - len(todo)} reused", flush=True)
        save_json(spans, spans_path)
    matcher = ner.Matcher(entities, args.engine)
    with timed("S4_ner_fuzzy", args.dataset, f"labels={args.labels} engine={args.engine}"):
        out = ner.topic_entities(qs, key, texts, spans, matcher)
    out_path = Path(args.out) if args.out else d / f"topic_ner_{args.labels}{args.suffix}.json"
    save_json(out, out_path)
    print(f"unique fuzzy queries: {len(matcher.cache)}", flush=True)
    oracle_path = d / "slots_oracle.json"
    if oracle_path.exists():
        tab = ner.agreement_table(qs, key, out, load_json(oracle_path))
        print(tab.to_string())
        tab.to_csv(report_dir / f"{args.dataset}_topic_entity_agreement_{args.labels}.csv", index=False)
    if args.engine == "rapidfuzz" and args.verify_sample > 0:
        msg = ner.fuzzy_engine_check(matcher, entities, args.verify_sample)
        print(msg)
        (report_dir / f"{args.dataset}_fuzzy_engine_check_{args.labels}.txt").write_text(msg + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--method", choices=["rule", "ner", "oracle"], default="rule")
    ap.add_argument("--out", default=None, help="override the output JSON path")
    ap.add_argument("--questions", default="questions.json", help="question file in cache/<ds>/ (e.g. questions_v2.json)")
    ap.add_argument("--rule_version", choices=list(rules.RULE_VERSIONS), default="v2", help="rule: classifier version")
    ap.add_argument("--suffix", default="", help="ner: suffix for ner_spans_/topic_ner_ files (e.g. _v2)")
    ap.add_argument("--torch_threads", type=int, default=0, help="ner: cap torch CPU threads (0 = default)")
    ap.add_argument("--report_dir", default=None, help="where CSV/TXT diagnostics go (default results/)")
    ap.add_argument("--ner_topics", default=None, help="rule: NER topic fallback JSON (default cache/<ds>/topic_ner_all.json)")
    ap.add_argument("--labels", choices=["faithful", "all"], default="faithful", help="ner: label ids kept")
    ap.add_argument("--engine", choices=["rapidfuzz", "fuzzywuzzy"], default="rapidfuzz", help="ner: fuzzy matcher")
    ap.add_argument("--device", default="mps", help="ner: torch device")
    ap.add_argument("--verify_sample", type=int, default=300, help="ner: rapidfuzz vs fuzzywuzzy cross-check size")
    args = ap.parse_args()
    ensure_dirs()
    d = CACHE / args.dataset
    report_dir = Path(args.report_dir) if args.report_dir else RESULTS
    report_dir.mkdir(parents=True, exist_ok=True)
    {"rule": lambda: run_rule(args, d), "oracle": lambda: run_oracle(args, d, report_dir),
     "ner": lambda: run_ner(args, d, report_dir)}[args.method]()


if __name__ == "__main__":
    main()
