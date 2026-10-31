"""MultiTQ preparation from the TimeR4 release: fact table, fact sentences, questions with TimeR4's top-15.

Outputs (out_dir):
  vocab.json        {"entities": [...], "relations": [...], "timestamps": [...]}
  facts.npz         s_id, r_id, o_id, t_id (int32), t_days (int32)
  sentences.txt     one "s r o in t" sentence per fact (underscores -> spaces, as in the original TEAR code)
  questions.json    [{quid, question, rewrite, answers, answer_type, qtype, qlabel, time_level,
                      timer4_top15_ids (fact ids or -1), timer4_top15_raw}]
query_text / make_query_questions: 380 TimeR4 rewrites are empty (341) or truncated (39); the query of a question is
its rewrite when valid, otherwise the original question (questions_v2.json, field 'query'). stage_query is what every
stage reads: the 'query' field when present, otherwise 'rewrite' (so v1 files and multitq_orig behave as before).
make_original_variant builds cache/multitq_orig: the same fact files (symlinked) and questions whose 'rewrite' field
is the original question, for the no-rewrite ablation.
"""
from __future__ import annotations

import ast
import os
import re
from pathlib import Path

import numpy as np

from ..paths import EXTERNAL
from ..temporal import date_to_days
from ..utils.io import load_json, save_json

DEFAULT_KG = EXTERNAL / "TimeR4/datasets/MultiTQ/kg/full.txt"
DEFAULT_TEST = EXTERNAL / "TimeR4/datasets/MultiTQ/questions/test.json"
DEFAULT_PROMPT = EXTERNAL / "TimeR4/datasets/MultiTQ/prompt/test_prompt.json"


def prepare(kg_path, test_path, prompt_path, out_dir) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ents, rels, tss = {}, {}, {}
    s_id, r_id, o_id, t_id = [], [], [], []
    sentences = []
    with open(kg_path, encoding="utf-8") as f:
        for line in f:
            s, r, o, t = line.strip().replace("_", " ").split("\t")
            s_id.append(ents.setdefault(s, len(ents)))
            r_id.append(rels.setdefault(r, len(rels)))
            o_id.append(ents.setdefault(o, len(ents)))
            t_id.append(tss.setdefault(t, len(tss)))
            sentences.append(f"{s} {r} {o} in {t}")
    ent_list, rel_list, ts_list = list(ents), list(rels), list(tss)
    t_days = np.array([date_to_days(t) for t in ts_list], dtype=np.int32)
    assert (t_days >= 0).all()
    np.savez(out / "facts.npz", s_id=np.array(s_id, np.int32), r_id=np.array(r_id, np.int32),
             o_id=np.array(o_id, np.int32), t_id=np.array(t_id, np.int32), t_days=t_days[np.array(t_id)])
    save_json({"entities": ent_list, "relations": rel_list, "timestamps": ts_list}, out / "vocab.json")
    with open(out / "sentences.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(sentences) + "\n")
    print(f"facts={len(sentences)} entities={len(ent_list)} relations={len(rel_list)} timestamps={len(ts_list)}")

    sent2id = {}
    for i, s in enumerate(sentences):
        sent2id.setdefault(s, i)
    print(f"unique sentences={len(sent2id)} (duplicates={len(sentences)-len(sent2id)})")

    test = load_json(test_path)
    prompt = load_json(prompt_path)
    assert len(test) == len(prompt)
    qs, unmapped, total = [], 0, 0
    for a, b in zip(test, prompt):
        assert ast.literal_eval(b["answers"]) == a["answers"]
        m = re.search(r"Historical facts:(\[.*\])\nQuestion:", b["text"], re.S)
        facts = ast.literal_eval(m.group(1))
        ids = []
        for fs in facts:
            fid = sent2id.get(fs, -1)
            if fid < 0:
                unmapped += 1
            ids.append(fid)
            total += 1
        qs.append({"quid": a["quid"], "question": a["question"], "rewrite": b["question"],
                   "answers": a["answers"], "answer_type": a["answer_type"], "qtype": a["qtype"],
                   "qlabel": a["qlabel"], "time_level": a["time_level"],
                   "timer4_top15_ids": ids, "timer4_top15_raw": facts})
    save_json(qs, out / "questions.json")
    print(f"questions={len(qs)}  TimeR4 top-15 strings mapped to KG facts: {total-unmapped}/{total} "
          f"(unmapped {unmapped})")


# --------------------------------------------------------------------------- original-question variant
SHARED_FILES = ("facts.npz", "vocab.json", "sentences.txt", "fact_emb.npy")


def original_question_variant(questions: list) -> list:
    """Copies of the questions whose 'rewrite' field holds the original MultiTQ question; TimeR4's gpt-3.5 rewrite is
    kept as 'rewrite_timer4'. Every stage reads 'rewrite', so the variant runs the same pipeline on the original
    wording."""
    out = []
    for q in questions:
        v = dict(q)
        v["rewrite_timer4"] = q["rewrite"]
        v["rewrite"] = q["question"]
        out.append(v)
    return out


def make_original_variant(src_dir, out_dir) -> None:
    """out_dir (cache/multitq_orig): the fact files are relative symlinks into src_dir (never copied or written);
    questions.json is the original-question variant of src_dir/questions.json."""
    src, out = Path(src_dir).resolve(), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if out.resolve() == src:
        raise ValueError("the variant must not be written into the source directory")
    for name in SHARED_FILES:
        link = out / name
        if not link.is_symlink():
            link.symlink_to(os.path.relpath(src / name, out.resolve()))
    qs = original_question_variant(load_json(src / "questions.json"))
    save_json(qs, out / "questions.json")
    same = sum(q["rewrite"] == q["rewrite_timer4"] for q in qs)
    print(f"questions={len(qs)}  original == TimeR4 rewrite: {same}  differ: {len(qs) - same}")


ENCODER_VARIANT_SHARED = ("facts.npz", "vocab.json", "sentences.txt", "questions.json", "slots_rule_v2.json",
                          "topic_ner_all.json", "entity_emb.npy", "sample_6000.json")


def make_encoder_variant(src_dir, out_dir) -> None:
    """out_dir (e.g. cache/multitq_bge): the main setting of src_dir (cache/multitq_orig) with another retriever.
    Questions, labels, NER topics, fact sentences and the entity-name embeddings of anchor grounding are symlinks;
    fact_emb.npy, q_emb.npy, the top-N arrays, anchors.json and rerank/ are written by the stages for the new encoder."""
    src, out = Path(src_dir).resolve(), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if out.resolve() == src:
        raise ValueError("the variant must not be written into the source directory")
    for name in ENCODER_VARIANT_SHARED:
        link = out / name
        if not link.is_symlink() and (src / name).exists():
            link.symlink_to(os.path.relpath(src / name, out.resolve()))


# --------------------------------------------------------------------------- query text (v2)
TRUNCATION_RATIO = 0.6


def rewrite_is_valid(rewrite, question: str) -> bool:
    """False for an empty rewrite and for a truncated one: no question mark and fewer than 60% of the original
    question's words (e.g. 'After 2012-11-21, who was')."""
    r = (rewrite or "").strip()
    if not r:
        return False
    if not r.endswith("?") and len(r.split()) < TRUNCATION_RATIO * len(question.split()):
        return False
    return True


def query_text(q: dict) -> str:
    """The rewrite when valid, otherwise the original question."""
    return q["rewrite"] if rewrite_is_valid(q.get("rewrite"), q["question"]) else q["question"]


def stage_query(q: dict) -> str:
    """Question text read by search, labels, reranking cues, LLM-free Hit@1 and the reader: 'query' when present
    (questions_v2.json), otherwise 'rewrite' (questions.json, multitq_orig)."""
    return q["query"] if "query" in q else q["rewrite"]


def add_query_field(questions: list) -> list:
    out = []
    for q in questions:
        v = dict(q)
        v["query"] = query_text(q)
        v["query_source"] = "rewrite" if v["query"] == q["rewrite"] else "question"
        out.append(v)
    return out


def make_query_questions(d, src_name: str = "questions.json", out_name: str = "questions_v2.json") -> Path:
    """Writes d/out_name = d/src_name plus 'query' / 'query_source'; refuses to overwrite the source."""
    d = Path(d)
    if out_name == src_name:
        raise ValueError("the v2 questions must not overwrite the source file")
    qs = add_query_field(load_json(d / src_name))
    save_json(qs, d / out_name)
    n_q = sum(q["query_source"] == "question" for q in qs)
    n_empty = sum(not (q["rewrite"] or "").strip() for q in qs)
    print(f"questions={len(qs)}  query=original question for {n_q} (empty rewrite {n_empty}, truncated {n_q - n_empty})")
    return d / out_name
