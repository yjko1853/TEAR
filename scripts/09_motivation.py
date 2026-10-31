"""Motivation analysis: cosine similarity of fact pairs sharing (s, r, o) and differing only in time."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tear.analysis.motivation import same_sro_similarity  # noqa: E402
from tear.paths import CACHE, RESULTS, ensure_dirs  # noqa: E402
from tear.utils.timing import timed  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--pairs", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--results_dir", default=None, help="default results/")
    args = ap.parse_args()
    ensure_dirs()
    results = Path(args.results_dir) if args.results_dir else RESULTS
    results.mkdir(parents=True, exist_ok=True)
    with timed("S8_same_sro_similarity", args.dataset, f"pairs={args.pairs}"):
        df, hist = same_sro_similarity(CACHE / args.dataset, args.dataset, args.pairs, args.seed)
    print(df.round(4).to_string())
    df.to_csv(results / f"{args.dataset}_same_sro_similarity.csv", index=False)
    hist.to_csv(results / f"{args.dataset}_same_sro_similarity_hist.csv", index=False)


if __name__ == "__main__":
    main()
