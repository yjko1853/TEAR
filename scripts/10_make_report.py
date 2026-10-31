"""Assemble results/results.md from the CSV outputs."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tear.reporting import build_report  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=None, help="default results/results.md")
    args = ap.parse_args()
    build_report(Path(args.out) if args.out else None)


if __name__ == "__main__":
    main()
