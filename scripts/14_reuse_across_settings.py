"""Stage 5c: copy reader predictions across rerank settings when the reader's input is identical.

Within one dataset, tag, prompt version and k, two settings give the reader the same prompt for a question when their
top-k fact lists are identical (same fact ids in the same order); the question text is the same by construction.
Decoding is greedy, so the output is identical too (checked on chunk 0 of the grounded run: 4,493 identical-input
pairs, 4,493 identical outputs). Before a setting is run on a slice of the shuffled sample, this script copies, for
every question in the slice that the target has not answered yet, the prediction of the first source setting whose
top-k list is identical. Copied records carry "copied_from". timer4_public is never a target or source (different
fact lists).

  .venv/bin/python -I scripts/14_reuse_across_settings.py --dataset multitq_orig --tag rule_v2_ground_exact_anydate \
      --pred_dir results/qwen_ground --target tear --sources none,tear_window,timer4 --start 1000 --end 2000
"""
import argparse, json, random, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='multitq_orig')
    ap.add_argument('--tag', required=True)
    ap.add_argument('--pred_dir', required=True)
    ap.add_argument('--target', required=True)
    ap.add_argument('--sources', required=True, help='comma-separated settings, tried in order')
    ap.add_argument('--sample', default='sample_6000.json')
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--end', type=int, default=None)
    ap.add_argument('--model', default='Qwen2.5-7B-Instruct-4bit')
    ap.add_argument('--prompt', default='v3')
    ap.add_argument('--k', type=int, default=15)
    ap.add_argument('--seed', type=int, default=2026, help='shuffle seed of the reader (same as 07)')
    ap.add_argument('--source_dir', default=None, help='directory of the source predictions (default: --pred_dir)')
    ap.add_argument('--dry_run', action='store_true')
    a = ap.parse_args()
    cache = ROOT / 'cache' / a.dataset
    order = [str(x) for x in json.load(open(cache / 'rerank' / 'question_order.json'))['question_ids']]
    pos = {q: i for i, q in enumerate(order)}
    sample = [str(x) for x in json.load(open(cache / a.sample))['quids']]
    random.Random(a.seed).shuffle(sample)
    sl = sample[a.start:a.end]
    pdir = ROOT / a.pred_dir
    sdir = ROOT / a.source_dir if a.source_dir else pdir
    fname = lambda s, d=pdir: d / f'{s}_{a.tag}__{a.model}__k{a.k}__p-{a.prompt}.jsonl'
    def read(p):
        return {json.loads(l)['quid']: json.loads(l) for l in open(p)} if p.exists() else {}
    tgt_arr = np.load(cache / 'rerank' / f'{a.target}_{a.tag}_top30_ids.npy', mmap_mode='r')
    have = read(fname(a.target))
    srcs = [s for s in a.sources.split(',') if s and s not in (a.target, 'timer4_public')]
    src = [(s, np.load(cache / 'rerank' / f'{s}_{a.tag}_top30_ids.npy', mmap_mode='r'), read(fname(s, sdir))) for s in srcs]
    out, counts = [], {}
    for q in sl:
        if q in have:
            continue
        i = pos[q]; t = np.asarray(tgt_arr[i, :a.k])
        for s, arr, preds in src:
            if q in preds and np.array_equal(t, np.asarray(arr[i, :a.k])):
                r = dict(preds[q]); r['copied_from'] = r.get('copied_from', s)
                out.append(r); counts[s] = counts.get(s, 0) + 1
                break
    print(json.dumps({'target': a.target, 'slice': [a.start, a.end], 'todo_before': sum(1 for q in sl if q not in have),
                      'copied': len(out), 'by_source': counts, 'dry_run': a.dry_run}))
    if not a.dry_run and out:
        with open(fname(a.target), 'a') as f:
            for r in out:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
