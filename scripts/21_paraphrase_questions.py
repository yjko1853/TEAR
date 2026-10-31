"""Paraphrase robustness set (pre-registered: results/prereg_paraphrase_robustness.md).

  --select      draw the sample (200 per multi-constraint type, seed 2026) -> cache/multitq_para_src/sample.json
  --smoke N     paraphrase N (<= 5) dev-split questions to check the output format only (nothing saved)
  --generate    paraphrase the sample with the local MLX model (greedy, resumable)
                -> cache/multitq_para_src/paraphrases.json {quid: {paraphrase, raw, valid, reasons, lost_entities}}
  --recheck     recompute the validity filter for every stored paraphrase (after a filter amendment)
  --build       write the dataset variant cache/multitq_para (symlinked fact files, questions.json)
  --meaning     write the 50-question meaning-check sheet (seed 7) -> results/paraphrase_meaning_check.csv
  --meaning_import XLSX  read the filled check (검수/바꿔쓰기_의미검수_50.xlsx, 판정 같음/다름) into that CSV
Run --generate with .venv-mlx; the other steps with .venv.
"""
import argparse
import csv
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tear.data import paraphrase as para  # noqa: E402
from tear.paths import CACHE, EXTERNAL, RESULTS  # noqa: E402
from tear.utils.io import load_json, save_json  # noqa: E402

SRC = CACHE / "multitq_para_src"


def mentioner():
    from tear.labeling.rules import build_mentioner
    return build_mentioner(load_json(CACHE / "multitq_orig" / "vocab.json")["entities"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--select", action="store_true")
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--smoke_seed", type=int, default=1, help="dev questions of the smoke test (1: first, 2: after amendment 1)")
    ap.add_argument("--generate", action="store_true")
    ap.add_argument("--recheck", action="store_true")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--meaning", action="store_true")
    ap.add_argument("--meaning_import", default=None)
    ap.add_argument("--n_per", type=int, default=200)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--max_tokens", type=int, default=80)
    args = ap.parse_args()
    SRC.mkdir(parents=True, exist_ok=True)
    qs = load_json(CACHE / "multitq_orig" / "questions.json")
    by_id = {str(q["quid"]): q for q in qs}

    if args.select:
        quids = para.draw_sample(qs, args.n_per, args.seed)
        save_json({"seed": args.seed, "n_per": args.n_per, "population": "multi-constraint test questions without a "
                   "date or month name", "quids": quids}, SRC / "sample.json")
        print(f"sample: {len(quids)} questions -> {SRC / 'sample.json'}")

    if args.smoke:
        from tear.reader.mlx_reader import MLXReader
        dev = [q for q in load_json(EXTERNAL / "MultiTQ_hf" / "dev.json")
               if q.get("qtype") in para.MULTI and para.in_population(q)]
        reader = MLXReader(max_tokens=args.max_tokens)
        for q in random.Random(args.smoke_seed).sample(dev, min(5, args.smoke)):
            print(q["question"], "\n  ->", para.clean_output(reader.answer(para.prompt_for(q["question"]))))

    if args.generate:
        from tear.reader.mlx_reader import MLXReader
        quids = load_json(SRC / "sample.json")["quids"]
        out_path = SRC / "paraphrases.json"
        done = para.load_paraphrases(out_path)
        todo = [k for k in quids if k not in done]
        print(f"paraphrases done {len(done)}, todo {len(todo)}", flush=True)
        if todo:
            reader, mention, t0 = MLXReader(max_tokens=args.max_tokens), mentioner(), time.time()
            for i, k in enumerate(todo, 1):
                raw = reader.answer(para.prompt_for(by_id[k]["question"]))
                p = para.clean_output(raw)
                done[k] = {"paraphrase": p, "raw": raw, **para.check(by_id[k]["question"], p, mention)}
                if i % 25 == 0 or i == len(todo):
                    save_json(done, out_path)
                    print(f"{i}/{len(todo)}  {(time.time() - t0) / i:.2f} s/question", flush=True)
        n_valid = sum(v["valid"] for v in done.values())
        print(f"valid {n_valid}/{len(done)}")

    if args.recheck:
        paras, mention = para.load_paraphrases(SRC / "paraphrases.json"), mentioner()
        for k, v in paras.items():
            v.update(para.check(by_id[k]["question"], v["paraphrase"], mention))
        save_json(paras, SRC / "paraphrases.json")
        from collections import Counter
        print(f"rechecked {len(paras)}: valid {sum(v['valid'] for v in paras.values())}",
              dict(Counter(r for v in paras.values() for r in v["reasons"])))

    if args.build:
        quids = load_json(SRC / "sample.json")["quids"]
        paras = para.load_paraphrases(SRC / "paraphrases.json")
        missing = [k for k in quids if k not in paras]
        if missing:
            raise SystemExit(f"{len(missing)} sample questions have no paraphrase yet")
        out = para.make_variant(CACHE / "multitq_orig", CACHE / "multitq_para", qs, {k: paras[k] for k in quids})
        print(f"cache/multitq_para: {len(out)} questions, valid {sum(q['paraphrase_valid'] for q in out)}")

    if args.meaning:
        paras = para.load_paraphrases(SRC / "paraphrases.json")
        valid = sorted(k for k, v in paras.items() if v["valid"])
        pick = random.Random(7).sample(valid, 50)
        with open(RESULTS / "paraphrase_meaning_check.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["quid", "qtype", "original", "paraphrase", "same_meaning", "note"])
            for k in pick:
                w.writerow([k, by_id[k]["qtype"], by_id[k]["question"], paras[k]["paraphrase"], "", ""])
        print(f"meaning-check sheet: {RESULTS / 'paraphrase_meaning_check.csv'}")


    if args.meaning_import:
        from openpyxl import load_workbook
        ws = load_workbook(args.meaning_import, data_only=True)["검수_50"]
        rows_x = list(ws.iter_rows(values_only=True))
        col = {h: j for j, h in enumerate(rows_x[0])}            # columns by header: 판정, 메모, quid
        cq, cj, cn = col["quid"], col["판정"], col["메모"]
        body = [r for r in rows_x[1:] if r[cq] is not None]
        judged = {str(r[cq]): {"같음": "yes", "다름": "no"}.get(str(r[cj] or "").strip(), "") for r in body}
        notes = {str(r[cq]): r[cn] or "" for r in body}
        path = RESULTS / "paraphrase_meaning_check.csv"
        rows = list(csv.DictReader(open(path, encoding="utf-8")))
        for r in rows:
            r["same_meaning"], r["note"] = judged.get(r["quid"], ""), notes.get(r["quid"], "")
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        print(f"imported {sum(bool(r['same_meaning']) for r in rows)}/{len(rows)} judgements -> {path}")


if __name__ == "__main__":
    main()
