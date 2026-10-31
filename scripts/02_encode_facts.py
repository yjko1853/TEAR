"""Stage 1: encode fact sentences with all-mpnet-base-v2 (MPS -> CPU fallback); resumable chunks."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tear.paths import CACHE, ensure_dirs  # noqa: E402
from tear.retrieval.encoder import encode_sentences_chunked  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--chunk", type=int, default=16384)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--sentences", default=None, help="override sentences file")
    ap.add_argument("--out", default=None, help="override output .npy")
    ap.add_argument("--tag", default="fact", help="prefix for parts dir / output name")
    ap.add_argument("--model", default=None, help="sentence-transformers model (default all-mpnet-base-v2)")
    args = ap.parse_args()
    ensure_dirs()
    d = CACHE / args.dataset
    sent_path = Path(args.sentences) if args.sentences else d / "sentences.txt"
    out_path = Path(args.out) if args.out else d / f"{args.tag}_emb.npy"
    parts_dir = out_path.parent / f"{args.tag}_emb_parts"
    if out_path.exists():
        print(f"{out_path} exists; nothing to do (delete it to re-encode)", flush=True)
        return
    with open(sent_path, encoding="utf-8") as f:
        texts = [line.rstrip("\n") for line in f]
    print(f"{len(texts)} sentences from {sent_path}", flush=True)
    encode_sentences_chunked(texts, out_path, parts_dir, args.dataset, args.tag, args.batch, args.chunk, args.device,
                             args.model)


if __name__ == "__main__":
    main()
