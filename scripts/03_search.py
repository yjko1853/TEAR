"""Stage 2: encode questions and run exact top-N search (FAISS IndexFlatIP, or --engine numpy).

MultiTQ reads the stage text: 'query' in questions_v2.json, otherwise the TimeR4 rewrite ('rewrite'; in multitq_orig
that field holds the original question). TimelineKGQA uses the original question. Question encoding (torch) runs in a
child process because torch + faiss in one process hangs on macOS. --incremental --suffix _v2 re-encodes and
re-searches only the questions whose text differs from questions.json and copies every other row.
"""
import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from tear.data import is_multitq  # noqa: E402
from tear.paths import CACHE, ensure_dirs  # noqa: E402
from tear.retrieval.encoder import encode_questions  # noqa: E402
from tear.retrieval.search import faiss_topn, numpy_topn  # noqa: E402
from tear.data.multitq import stage_query  # noqa: E402
from tear.utils.io import load_json, save_json  # noqa: E402


def question_texts(qs, dataset: str, field: str | None):
    """Explicit --field, else the stage text for MultiTQ ('query' if present, else 'rewrite'), else 'question'."""
    if field:
        return [q[field] for q in qs], field
    if is_multitq(dataset):
        return [stage_query(q) for q in qs], "query|rewrite"
    return [q["question"] for q in qs], "question"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--questions", default="questions.json", help="question file in cache/<ds>/ (e.g. questions_v2.json)")
    ap.add_argument("--field", default=None, help="question field (default: query, else rewrite, for multitq*; question otherwise)")
    ap.add_argument("--suffix", default="", help="output names q_emb<suffix>.npy and top<N><suffix>_{ids,scores}.npy")
    ap.add_argument("--incremental", action="store_true",
                    help="only questions whose text differs from questions.json are re-encoded and searched; the other "
                         "rows are copied from q_emb.npy / top<N>_{ids,scores}.npy")
    ap.add_argument("--topn", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--qbatch", type=int, default=2048)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--threads", type=int, default=8, help="faiss threads")
    ap.add_argument("--torch_threads", type=int, default=0, help="cap torch CPU threads in the encoding child (0 = default)")
    ap.add_argument("--engine", choices=["faiss", "numpy"], default="faiss")
    ap.add_argument("--model", default=None, help="question encoder (must match the fact encoder; default all-mpnet-base-v2)")
    ap.add_argument("--qemb", default=None, help="question embeddings .npy (default cache/<ds>/q_emb<suffix>.npy)")
    ap.add_argument("--out_dir", default=None, help="where top<N><suffix>_{ids,scores}.npy go (default cache/<ds>)")
    ap.add_argument("--limit", type=int, default=0, help="debug: only the first L questions")
    ap.add_argument("--rows", default=None, help="internal: JSON list of question rows to encode")
    ap.add_argument("--encode_only", action="store_true", help="internal: encode questions and exit")
    args = ap.parse_args()
    ensure_dirs()
    d = CACHE / args.dataset
    qs = load_json(d / args.questions)
    texts, field = question_texts(qs, args.dataset, args.field)
    print(f"{len(texts)} questions from {args.questions}, field={field}", flush=True)
    qemb_path = Path(args.qemb) if args.qemb else d / f"q_emb{args.suffix}.npy"
    if args.encode_only:
        if args.rows:
            texts = [texts[i] for i in load_json(args.rows)]
        encode_questions(texts, qemb_path, args.dataset, args.batch, args.device, args.torch_threads or None, args.model)
        return

    def encode_child(out_path, rows_path=None):
        cmd = [sys.executable, "-I", str(Path(__file__).resolve()), "--dataset", args.dataset, "--questions",
               args.questions, "--batch", str(args.batch), "--device", args.device, "--qemb", str(out_path),
               "--torch_threads", str(args.torch_threads), "--encode_only"]
        if args.field:
            cmd += ["--field", args.field]
        if args.model:
            cmd += ["--model", args.model]
        if rows_path:
            cmd += ["--rows", str(rows_path)]
        subprocess.run(cmd, check=True)

    def search(qemb):
        femb = np.load(d / "fact_emb.npy", mmap_mode="r")
        if args.engine == "faiss":
            return faiss_topn(qemb, femb, args.topn, args.dataset, args.qbatch, args.threads)
        return numpy_topn(qemb, femb, args.topn, args.dataset)

    out_dir = Path(args.out_dir) if args.out_dir else d
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.incremental:
        if not args.suffix:
            raise SystemExit("--incremental needs --suffix (the reference arrays must not be overwritten)")
        ref_qs = load_json(d / "questions.json")
        assert [q["quid" if is_multitq(args.dataset) else "qid"] for q in ref_qs] == \
               [q["quid" if is_multitq(args.dataset) else "qid"] for q in qs], "question order differs"
        ref_texts, _ = question_texts(ref_qs, args.dataset, args.field)
        rows = [i for i, (a, b) in enumerate(zip(texts, ref_texts)) if a != b]
        print(f"incremental: {len(rows)} of {len(texts)} questions changed", flush=True)
        qemb = np.load(d / "q_emb.npy").copy()
        ids = np.load(d / f"top{args.topn}_ids.npy").copy()
        scores = np.load(d / f"top{args.topn}_scores.npy").copy()
        if rows:
            rows_path = qemb_path.with_name(qemb_path.stem + "_rows.json")
            sub_path = qemb_path.with_name(qemb_path.stem + "_rows.npy")
            save_json(rows, rows_path)
            encode_child(sub_path, rows_path)
            sub = np.load(sub_path)
            qemb[rows] = sub
            sub_ids, sub_scores = search(sub)
            ids[rows], scores[rows] = sub_ids, sub_scores
            rows_path.unlink(); sub_path.unlink()
        np.save(qemb_path, qemb)
    else:
        if not qemb_path.exists():
            encode_child(qemb_path)
        qemb = np.load(qemb_path)
        if args.limit:
            qemb = qemb[:args.limit]
        print("question embeddings", qemb.shape, flush=True)
        ids, scores = search(qemb)
    np.save(out_dir / f"top{args.topn}{args.suffix}_ids.npy", ids)
    np.save(out_dir / f"top{args.topn}{args.suffix}_scores.npy", scores)
    print("saved", out_dir / f"top{args.topn}{args.suffix}_ids.npy", flush=True)


if __name__ == "__main__":
    main()
