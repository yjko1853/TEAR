"""Local LLM reader on Apple silicon with MLX (greedy decoding, no API), plus the question samples it reads.

Prediction files: <pred_dir>/<setting>[_<tag>]__<model>__k<k>__p-<prompt>[__bench][__tune].jsonl (resumable: questions
already present are skipped). <setting> is a reranker name, or 'timer4_public' for TimeR4's released top-15.
"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np

from ..data.multitq import stage_query
from ..utils.io import load_json
from .prompt import build_prompt

DEFAULT_MODEL = "mlx-community/Qwen2.5-7B-Instruct-4bit"
SHUFFLE_SEED = 2026   # fixed order shared by every setting: any prefix is a paired, roughly stratified subset


class MLXReader:
    def __init__(self, model: str = DEFAULT_MODEL, max_tokens: int = 48):
        from mlx_lm import generate, load
        try:
            from mlx_lm.sample_utils import make_sampler
            self.gen_kw = {"sampler": make_sampler(temp=0.0)}
        except Exception:  # older mlx_lm
            self.gen_kw = {"temp": 0.0}
        self._generate = generate
        self.model, self.tok = load(model)
        self.max_tokens = max_tokens

    def chat(self, prompt: str) -> str:
        return self.tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True,
                                            tokenize=False)

    def answer(self, prompt: str) -> str:
        return self._generate(self.model, self.tok, prompt=self.chat(prompt), max_tokens=self.max_tokens,
                              verbose=False, **self.gen_kw)

    def answer_batch(self, prompts) -> list[str]:
        from mlx_lm import batch_generate
        ids = [self.tok.encode(self.chat(p), add_special_tokens=False) for p in prompts]
        bkw = {"sampler": self.gen_kw["sampler"]} if "sampler" in self.gen_kw else {}
        return batch_generate(self.model, self.tok, ids, max_tokens=self.max_tokens, verbose=False, **bkw).texts


# --------------------------------------------------------------------------- question samples
def make_sample(questions, n_per: int = 1000, seed: int = 13) -> dict:
    """Evaluation sample: n_per questions per qtype (sorted quids within each type)."""
    by = {}
    for q in questions:
        by.setdefault(q["qtype"], []).append(str(q["quid"]))
    rnd = random.Random(seed); out = []
    for qt in sorted(by):
        out += sorted(rnd.sample(by[qt], min(n_per, len(by[qt]))))
    return {"seed": seed, "n_per_qtype": n_per, "quids": out}


def make_tune(questions, used: set, n_per: int = 20, seed: int = 7, disjoint_from: str = "sample_6000.json") -> dict:
    """Prompt-development set, disjoint from the evaluation sample."""
    by = {}
    for q in questions:
        if str(q["quid"]) not in used:
            by.setdefault(q["qtype"], []).append(str(q["quid"]))
    rnd = random.Random(seed); out = []
    for qt in sorted(by):
        out += sorted(rnd.sample(by[qt], n_per))
    return {"seed": seed, "n_per_qtype": n_per, "disjoint_from": disjoint_from, "quids": out}


def select_questions(sample: list, limit: int = 0, start: int = 0, end: int | None = None) -> list:
    """limit > 0: evenly strided speed-check subset; otherwise the fixed shuffle, sliced [start:end]."""
    if limit:
        return sample[:: max(1, len(sample) // limit)][:limit]
    sample = list(sample)
    random.Random(SHUFFLE_SEED).shuffle(sample)
    return sample[start:end]


def facts_for(d: Path, setting: str, tag: str, k: int, keys, questions_file: str = "questions.json"):
    """{quid: [top-k fact sentences]} for a reranker setting, or TimeR4's released top-15 ('timer4_public')."""
    sents = (d / "sentences.txt").read_text().splitlines()
    qs = {str(q["quid"]): q for q in load_json(d / questions_file)}
    if setting == "timer4_public":
        return {key: qs[key]["timer4_top15_raw"][:k] for key in keys}
    order = load_json(d / "rerank" / "question_order.json")["question_ids"]
    pos = {str(x): i for i, x in enumerate(order)}
    top = np.load(d / "rerank" / f"{setting}_{tag}_top30_ids.npy", mmap_mode="r")
    return {key: [sents[int(i)] for i in top[pos[key], :k]] for key in keys}


def output_path(pred_dir: Path, setting: str, tag: str, model: str, k: int, prompt: str, bench: bool,
                tune: bool) -> Path:
    name = setting if setting == "timer4_public" else f"{setting}_{tag}"
    suffix = ("__bench" if bench else "") + ("__tune" if tune else "")
    return pred_dir / f"{name}__{Path(model).name}__k{k}__p-{prompt}{suffix}.jsonl"


def run_reader(d: Path, pred_dir: Path, setting: str, tag: str, sample_file: str, model: str, k: int,
               max_tokens: int, prompt: str = "timer4", batch: int = 1, limit: int = 0, start: int = 0,
               end: int | None = None, questions_file: str = "questions.json",
               granularity: dict | None = None) -> Path:
    """questions_file selects the question version; the prompt reads the stage text (query, else rewrite).
    granularity: {quid: 'year' / 'month' / 'day' / None} for prompt v3g (tear/labeling/granularity.py)."""
    sample = select_questions(load_json(d / sample_file)["quids"], limit, start, end)
    qs = {str(q["quid"]): q for q in load_json(d / questions_file)}
    facts = facts_for(d, setting, tag, k, sample, questions_file)
    pred_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_path(pred_dir, setting, tag, model, k, prompt, bool(limit), sample_file.startswith("tune_"))
    done = set()
    if out_path.exists():
        done = {json.loads(l)["quid"] for l in out_path.read_text().splitlines() if l.strip()}
    todo = [key for key in sample if key not in done]
    reader = MLXReader(model, max_tokens)
    t0 = time.time(); n = 0
    with out_path.open("a") as fo:
        for lo in range(0, len(todo), batch):
            keys = todo[lo:lo + batch]
            prompts = [build_prompt(facts[key], stage_query(qs[key]), prompt, (granularity or {}).get(key))
                       for key in keys]
            outs = reader.answer_batch(prompts) if batch > 1 else [reader.answer(prompts[0])]
            for key, out in zip(keys, outs):
                q = qs[key]
                fo.write(json.dumps({"quid": key, "prediction": out, "answers": q["answers"], "qtype": q["qtype"],
                                     "qlabel": q["qlabel"], "answer_type": q["answer_type"]},
                                    ensure_ascii=False) + "\n")
            fo.flush(); n += len(keys)
            if (n // batch) % max(1, 200 // batch) == 0 or n == len(todo):
                el = time.time() - t0
                print(f"{n}/{len(todo)} done, {el / n:.2f} s/question, ETA {(len(todo) - n) * el / n / 3600:.2f} h",
                      flush=True)
    el = time.time() - t0
    print(f"finished {n} questions in {el:.0f}s ({el / max(n, 1):.2f} s/question) -> {out_path}")
    return out_path
