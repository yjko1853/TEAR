from __future__ import annotations

import csv
import time
from contextlib import contextmanager
from datetime import datetime

from ..paths import LOGS


@contextmanager
def timed(step: str, dataset: str = "", note: str = ""):
    """Print start/end and append the wall-clock seconds to logs/timing.csv."""
    t0 = time.time()
    print(f"[{datetime.now():%H:%M:%S}] START {step} {dataset} {note}", flush=True)
    try:
        yield
    finally:
        dt = time.time() - t0
        print(f"[{datetime.now():%H:%M:%S}] END   {step} {dataset} {dt:.1f}s", flush=True)
        LOGS.mkdir(parents=True, exist_ok=True)
        path = LOGS / "timing.csv"
        new = not path.exists()
        with open(path, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["timestamp", "step", "dataset", "seconds", "note"])
            w.writerow([datetime.now().isoformat(timespec="seconds"), step, dataset, f"{dt:.1f}", note])
