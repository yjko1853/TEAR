"""Repository paths. TEAR_HOME overrides the root directory; TEAR_LOGS overrides only the log directory
(useful for regression runs that must not touch logs/timing.csv)."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.environ["TEAR_HOME"]).resolve() if os.environ.get("TEAR_HOME") else Path(__file__).resolve().parents[1]
EXTERNAL = ROOT / "external"   # third-party clones (TimeR4, TimelineKGQA, original TEAR); never executed
CACHE = ROOT / "cache"         # intermediate artefacts, one sub-directory per dataset
RESULTS = ROOT / "results"     # CSV tables and reports
LOGS = Path(os.environ["TEAR_LOGS"]).resolve() if os.environ.get("TEAR_LOGS") else ROOT / "logs"  # timing.csv, run logs


def ensure_dirs() -> None:
    for d in (CACHE, RESULTS, LOGS):
        d.mkdir(parents=True, exist_ok=True)


def dataset_dir(dataset: str) -> Path:
    return CACHE / dataset
