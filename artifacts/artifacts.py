"""Pack / unpack the released artefacts (artifacts/README.md). numpy only; no model, no network.

  python -I artifacts/artifacts.py unpack [--cache cache] [--force]
      artifacts/<ds>/*.npz      -> cache/<ds>/<rerank_dir>/<setting>_<tag>_top30_ids.npy + question_order.json
      artifacts/<ds>/*.json.gz  -> cache/<ds>/<name>.json (byte-identical to the files the pipeline wrote)
      artifacts/<ds>/*.json     -> cache/<ds>/<name>.json
      Existing files are kept unless --force.
  python -I artifacts/artifacts.py pack --cache <path to a full cache/> [--out artifacts]
      rebuilds the artefacts from a cache/ written by the pipeline (how this folder was made).

Each .npz holds one reranking run: ids (questions x settings x 30, int32; fact i = line i of cache/<ds>/sentences.txt),
settings (reranker names, axis 1), question_ids (row order, = rerank/question_order.json), tag and rerank_dir.
The settings are stacked on axis 1 so that zlib sees the largely shared lists of one question together.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TAG = "rule_v2_ground_exact_anydate"
SETTINGS = ["none", "timer4", "timer4_sym", "chrono", "hardwindow", "atc", "tr", "tear", "tear_window"]
# dataset -> (rerank directories to pack, JSON files to gzip, JSON files to copy as is)
SPEC = {
    "multitq_orig": (["rerank"], ["slots_rule_v2.json", "anchors.json", "sample_6000.json",
                      "sample_6000_dateslot.json"], []),
    "multitq_bge": (["rerank"], ["anchors.json"], []),
    "icews18": (["rerank", "rerank_dev"],
                ["questions.json", "questions_dev.json", "slots_rule_v2.json", "slots_rule_v2_dev.json",
                 "anchors.json", "anchors_dev.json", "sample_6000.json"],
                ["questions.meta.json", "questions_dev.meta.json"]),
    "multitq_para_src": ([], ["sample.json", "paraphrases.json"], []),
}


def _npz_name(rerank_dir: str) -> str:
    return f"{rerank_dir}_{TAG}_top30.npz"


def pack(cache: Path, out: Path) -> None:
    lines = []
    for ds, (rdirs, gz, plain) in SPEC.items():
        od = out / ds
        od.mkdir(parents=True, exist_ok=True)
        for rd in rdirs:
            src = cache / ds / rd
            arrs = [np.load(src / f"{s}_{TAG}_top30_ids.npy") for s in SETTINGS]
            ids = np.stack(arrs, axis=1).astype(np.int32)
            order = json.loads((src / "question_order.json").read_text(encoding="utf-8"))
            qids = np.asarray(order["question_ids"], dtype=np.int64)
            assert ids.shape[0] == len(qids), (ds, rd, ids.shape, len(qids))
            p = od / _npz_name(rd)
            np.savez_compressed(p, ids=ids, settings=np.asarray(SETTINGS), question_ids=qids,
                                tag=np.asarray(TAG), rerank_dir=np.asarray(rd))
            z = np.load(p)
            for j, s in enumerate(SETTINGS):  # round-trip check
                assert np.array_equal(z["ids"][:, j, :], arrs[j]), (ds, rd, s)
            print(f"{p.relative_to(out)}  ids {ids.shape}")
        for name in gz:
            raw = (cache / ds / name).read_bytes()
            p = od / f"{name}.gz"
            with open(p, "wb") as f, gzip.GzipFile(filename=name, mode="wb", fileobj=f, compresslevel=9, mtime=0) as g:
                g.write(raw)
            assert gzip.decompress(p.read_bytes()) == raw
            print(f"{p.relative_to(out)}")
        for name in plain:
            shutil.copyfile(cache / ds / name, od / name)
            print(f"{(od / name).relative_to(out)}")
    for p in sorted(out.rglob("*")):
        if p.is_file() and p.suffix in {".npz", ".gz", ".json"} and p.parent != out:
            lines.append(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(out).as_posix()}")
    (out / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write(p: Path, data: bytes, force: bool) -> bool:
    if p.exists() and not force:
        return False
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    return True


def unpack(cache: Path, force: bool) -> None:
    n_new = n_kept = 0
    for ds in SPEC:
        ad = HERE / ds
        cd = cache / ds
        for p in sorted(ad.glob("*.npz")):
            z = np.load(p)
            rd, tag = str(z["rerank_dir"]), str(z["tag"])
            order = json.dumps({"question_ids": [int(q) for q in z["question_ids"]]}).encode("utf-8")
            ok = _write(cd / rd / "question_order.json", order, force)
            n_new, n_kept = n_new + ok, n_kept + (not ok)
            for j, s in enumerate(z["settings"]):
                t = cd / rd / f"{s}_{tag}_top30_ids.npy"
                if t.exists() and not force:
                    n_kept += 1
                    continue
                t.parent.mkdir(parents=True, exist_ok=True)
                np.save(t, np.ascontiguousarray(z["ids"][:, j, :]))
                n_new += 1
        for p in sorted(ad.glob("*.json.gz")):
            ok = _write(cd / p.name[:-3], gzip.decompress(p.read_bytes()), force)
            n_new, n_kept = n_new + ok, n_kept + (not ok)
        for p in sorted(ad.glob("*.json")):
            ok = _write(cd / p.name, p.read_bytes(), force)
            n_new, n_kept = n_new + ok, n_kept + (not ok)
    print(f"written {n_new} files under {cache}, kept {n_kept} existing (use --force to overwrite)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    u = sub.add_parser("unpack")
    u.add_argument("--cache", type=Path, default=ROOT / "cache")
    u.add_argument("--force", action="store_true")
    k = sub.add_parser("pack")
    k.add_argument("--cache", type=Path, required=True)
    k.add_argument("--out", type=Path, default=HERE)
    args = ap.parse_args()
    if args.cmd == "pack":
        pack(args.cache.resolve(), args.out.resolve())
    else:
        unpack(args.cache.resolve(), args.force)


if __name__ == "__main__":
    main()
