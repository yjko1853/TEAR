"""TimelineKGQA preparation (ICEWS-Actor and CronQuestions-KG releases): interval facts and test questions.

Outputs (out_dir):
  vocab.json        {"entities": [...], "predicates": [...]}
  facts.npz         s_id, p_id, o_id, start_days, end_days (int32; open intervals use +-1e8), kg_id
  sentences.txt     "<s> <p> <o> from <start> to <end>" per fact
  questions.json    [{qid, question, paraphrase, answer, answer_type, level, qtype, temporal_relation,
                      gold_fact_ids, n_events, events_parsed_ok}]
  schema_notes.json column names, split counts, type counts, gold-event mapping statistics
"""
from __future__ import annotations

import ast
import collections
from pathlib import Path

import numpy as np
import pandas as pd

from ..paths import EXTERNAL
from ..temporal import NEG_INF_DAYS, POS_INF_DAYS, date_to_days
from ..utils.io import save_json

FILES = {
    "icews": ("unified_kg_icews_actor.csv", "unified_kg_icews_actor_questions_all.csv"),
    "cron": ("unified_kg_cron.csv", "unified_kg_cron_questions_all.csv"),
}
DEFAULT_DATADIR = EXTERNAL / "TimelineKGQA/Datasets"


def to_days(s: str) -> int:
    if s == "beginning of time":
        return NEG_INF_DAYS
    if s == "end of time":
        return POS_INF_DAYS
    d = date_to_days(s)
    if d is None:
        # CronQuestions-KG contains a few dates with year 0 (e.g. '0-01-01'), outside Python's date range.
        y = s.split("-")[0]
        if y.isdigit() and int(y) < 1:
            return NEG_INF_DAYS
        raise ValueError(s)
    return d


def parse_events(ev_str: str, predicates: set[str]):
    """The events column is str(list) of 's|p|o|start|end'; entity names containing ', ' were split upstream,
    so fragments without a predicate marker are re-joined to the previous fragment. Predicates are tried longest
    first: iterating a set made the result depend on the hash seed (CronQuestions-KG: 27-36 validation rows failed
    to parse under some seeds; successful parses are the same in every order)."""
    predicates = sorted(predicates, key=lambda p: (-len(p), p))
    frags = ast.literal_eval(ev_str)
    merged = []
    for fr in frags:
        if any(f"|{p}|" in fr for p in predicates) or not merged:
            merged.append(fr)
        else:
            merged[-1] = merged[-1] + ", " + fr
    out = []
    for fr in merged:
        pred = None
        for p in predicates:
            if f"|{p}|" in fr:
                pred = p
                break
        if pred is None:
            return None
        s, rest = fr.split(f"|{pred}|", 1)
        o, st, en = rest.rsplit("|", 2)
        out.append((s, pred, o, st, en))
    return out


def prepare(dataset: str, datadir, split: str, out_dir) -> None:
    kg_file, q_file = FILES[dataset]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    kg = pd.read_csv(Path(datadir) / kg_file, dtype=str, keep_default_na=False)
    ents, preds = {}, {}
    s_id, p_id, o_id, sd, ed = [], [], [], [], []
    sentences = []
    key2idx = {}
    for i, row in enumerate(kg.itertuples(index=False)):
        s, p, o, st, en = row.subject, row.predicate, row.object, row.start_time, row.end_time
        s_id.append(ents.setdefault(s, len(ents)))
        p_id.append(preds.setdefault(p, len(preds)))
        o_id.append(ents.setdefault(o, len(ents)))
        sd.append(to_days(st))
        ed.append(to_days(en))
        sentences.append(f"{s} {p} {o} from {st} to {en}")
        key2idx.setdefault((s, p, o, st, en), i)
    np.savez(out / "facts.npz", s_id=np.array(s_id, np.int32), p_id=np.array(p_id, np.int32),
             o_id=np.array(o_id, np.int32), start_days=np.array(sd, np.int32), end_days=np.array(ed, np.int32),
             kg_id=kg["id"].astype(np.int64).values)
    save_json({"entities": list(ents), "predicates": list(preds)}, out / "vocab.json")
    with open(out / "sentences.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(sentences) + "\n")
    n_open_start = int((np.array(sd) == NEG_INF_DAYS).sum())
    n_open_end = int((np.array(ed) == POS_INF_DAYS).sum())
    print(f"[{dataset}] facts={len(sentences)} entities={len(ents)} predicates={len(preds)} "
          f"open-start={n_open_start} open-end={n_open_end} dup(s,p,o,st,en)={len(sentences)-len(key2idx)}")

    q = pd.read_csv(Path(datadir) / q_file, dtype=str, keep_default_na=False)
    split_counts = q["split"].value_counts().to_dict()
    q = q[q["split"] == split].reset_index(drop=True)
    pset = set(preds)
    qs = []
    stats = collections.Counter()
    for r in q.itertuples(index=False):
        evs = parse_events(r.events, pset)
        gold = []
        ok = evs is not None
        if ok:
            for e in evs:
                j = key2idx.get(e, -1)
                if j < 0:
                    ok = False
                gold.append(j)
        stats[(r.answer_type, "ok" if ok else "fail")] += 1
        qs.append({"qid": r.id, "source_kg_id": r.source_kg_id, "question": r.question,
                   "paraphrase": r.paraphrased_question, "answer": r.answer, "answer_type": r.answer_type,
                   "level": r.question_level, "qtype": r.question_type, "temporal_relation": r.temporal_relation,
                   "gold_fact_ids": gold, "n_events": len(gold), "events_parsed_ok": ok})
    save_json(qs, out / "questions.json")
    notes = {
        "kg_columns": list(kg.columns), "question_columns": list(q.columns),
        "split_counts_all": split_counts, "n_test": len(qs),
        "test_by_level": q["question_level"].value_counts().to_dict(),
        "test_by_answer_type": q["answer_type"].value_counts().to_dict(),
        "test_by_question_type": q["question_type"].value_counts().to_dict(),
        "n_facts": len(sentences), "n_entities": len(ents), "predicates": list(preds)[:20],
        "open_start": n_open_start, "open_end": n_open_end,
        "gold_event_mapping": {f"{k[0]}|{k[1]}": v for k, v in stats.items()},
    }
    save_json(notes, out / "schema_notes.json")
    print(f"[{dataset}] test questions={len(qs)}; gold-event mapping: "
          f"ok={sum(v for k,v in stats.items() if k[1]=='ok')} fail={sum(v for k,v in stats.items() if k[1]=='fail')}")
    for k, v in sorted(stats.items()):
        print("   ", k, v)
