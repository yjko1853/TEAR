"""Second, larger reader (pre-registered: results/prereg_second_reader.md): Qwen2.5-14B-Instruct-4bit next to the
7B reader on the same questions (first 1,000 of the MultiTQ reader order), settings none / tear_window / hardwindow.

Scoring and statistics follow scripts/27_hardwindow_reader_report.py (exact Hit@1 of the first answer; reader+date;
direct = LLM-free top-1; McNemar with continuity correction; paired bootstrap, 10,000 draws, generator seeded 0 per
comparison). Every number is on the questions that both readers have answered for every listed setting (n reported).

  .venv/bin/python -I scripts/29_second_reader_report.py [--pred_dir results/qwen14b_ground --model Qwen2.5-14B-Instruct-4bit]
      [--label 14B --out_name second_reader]  -> results/<out_name>.md, results/<out_name>_{summary,significance}.csv
  The fallback reader (see the pre-registration): --pred_dir results/qwen_llama8b_ground --model Meta-Llama-3.1-8B-Instruct-4bit
      --label 8B-t48 --out_name second_reader_t48; the 256-token deviation: --pred_dir results/qwen_llama8b_ground_t256
      --label 8B-t256 --out_name second_reader_t256
"""
import argparse
import json
import math
import random
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tear.reader.parse import parse_list  # noqa: E402
from tear.text import normalize  # noqa: E402

TAG = 'rule_v2_ground_exact_anydate'
SETTINGS = ['none', 'tear_window', 'hardwindow']
PAIRS = [('none', 'tear_window'), ('none', 'hardwindow'), ('hardwindow', 'tear_window')]
METRICS = ('reader', 'reader_date', 'direct')


def gran(q):
    ql = q.lower()
    return 4 if re.search(r'\byear\b', ql) else (7 if re.search(r'\bmonth\b', ql) else 10)


def cut_date(p, q):
    m = re.search(r'(\d{4})(?:-(\d{2}))?(?:-(\d{2}))?', p)
    if not m:
        return p
    s = m.group(1) + ('-' + m.group(2) if m.group(2) else '') + ('-' + m.group(3) if m.group(3) else '')
    g = gran(q)
    return s[:g] if len(s) >= g else p


def ok(pred, q):
    return any(normalize(pred) == normalize(a) for a in q['answers'])


def sig(a, b):
    a = np.asarray(a, bool); b = np.asarray(b, bool); rng = np.random.default_rng(0)
    b01 = int((~a & b).sum()); b10 = int((a & ~b).sum())
    p = math.erfc(math.sqrt(((abs(b01 - b10) - 1) ** 2 / max(1, b01 + b10)) / 2)) if b01 + b10 else 1.0
    d = b.astype(float) - a.astype(float); bs = d[rng.integers(0, len(d), (10000, len(d)))].mean(1)
    return d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5), p, b01, b10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred_dir', default='results/qwen14b_ground')
    ap.add_argument('--model', default='Qwen2.5-14B-Instruct-4bit')
    ap.add_argument('--n', type=int, default=1000)
    ap.add_argument('--label', default='14B', help='name of the second reader in the output tables')
    ap.add_argument('--out_name', default='second_reader', help='results/<out_name>.md and _{summary,significance}.csv')
    a = ap.parse_args()
    L2 = a.label
    D = ROOT / 'cache' / 'multitq_orig'
    Q = {str(q['quid']): q for q in json.load(open(D / 'questions.json'))}
    S = json.load(open(D / 'slots_rule_v2.json'))
    order = [str(x) for x in json.load(open(D / 'rerank' / 'question_order.json'))['question_ids']]
    pos = {k: i for i, k in enumerate(order)}
    F = np.load(D / 'facts.npz'); V = json.load(open(D / 'vocab.json')); E, T = V['entities'], V['timestamps']
    sample = [str(x) for x in json.load(open(D / 'sample_6000.json'))['quids']]
    random.Random(2026).shuffle(sample); sample = sample[:a.n]
    readers = {'7B': ('results/qwen_ground', 'Qwen2.5-7B-Instruct-4bit'), L2: (a.pred_dir, a.model)}
    preds = {}
    for rd, (pdir, model) in readers.items():
        for s in SETTINGS:
            p = ROOT / pdir / f'{s}_{TAG}__{model}__k15__p-v3.jsonl'
            preds[(rd, s)] = {json.loads(l)['quid']: json.loads(l) for l in open(p)} if p.exists() else {}
    present = [s for s in SETTINGS if all(preds[(rd, s)] for rd in readers)]
    if not present:
        print('no second-reader predictions yet'); return
    keys = [k for k in sample if all(k in preds[(rd, s)] for rd in readers for s in present)]
    rows = []
    for s in present:
        top = np.load(D / 'rerank' / f'{s}_{TAG}_top30_ids.npy', mmap_mode='r')[:, 0]
        for rd in readers:
            for k in keys:
                q = Q[k]; pred = preds[(rd, s)][k]['prediction']; lst = parse_list(pred); first = lst[0] if lst else ''
                has_list = bool(re.search(r'\[.*\]', pred, re.S))  # the output holds a bracketed answer list
                fd = cut_date(first, q['question']) if q['answer_type'] == 'time' else first
                sl = S[k]['answer_slot']; fid = int(top[pos[k]])
                d = (T[int(F['t_id'][fid])][:gran(q['question'])] if sl == 'timestamp'
                     else (E[int(F['o_id'][fid])] if sl == 'object' else E[int(F['s_id'][fid])]))
                rows.append({'reader': rd, 'setting': s, 'quid': k, 'qtype': q['qtype'], 'answer_type': q['answer_type'],
                             'reader': rd, 'metric_reader': ok(first, q), 'metric_reader_date': ok(fd, q),
                             'metric_direct': ok(d, q), 'has_list': has_list})
    df = pd.DataFrame(rows)
    summ, sigs = [], []
    types = sorted(df.qtype.unique())
    for col in METRICS:
        c = f'metric_{col}'
        for rd in readers:
            for s in present:
                sub = df[(df.reader == rd) & (df.setting == s)]
                g = sub.groupby('qtype')[c].mean()
                summ.append({'reader': rd, 'metric': col, 'setting': s, 'n': len(sub), 'hit1': sub[c].mean(), 'list_share': sub['has_list'].mean(),
                             'entity': sub[sub.answer_type == 'entity'][c].mean(),
                             'time': sub[sub.answer_type == 'time'][c].mean(),
                             **{f'qt_{t}': g.get(t, np.nan) for t in types}})
            for base, s in PAIRS:
                if base not in present or s not in present:
                    continue
                x = df[(df.reader == rd) & (df.setting == base)].set_index('quid')[c]
                y = df[(df.reader == rd) & (df.setting == s)].set_index('quid')[c].loc[x.index]
                m, lo, hi, p, b01, b10 = sig(x.values, y.values)
                sigs.append({'reader': rd, 'metric': col, 'baseline': base, 'setting': s, 'n': len(x), 'diff': m,
                             'ci_low': lo, 'ci_high': hi, 'mcnemar_p': p, 'gains': b01, 'losses': b10})
        # the two readers on the same setting
        for s in present:
            x = df[(df.reader == '7B') & (df.setting == s)].set_index('quid')[c]
            y = df[(df.reader == L2) & (df.setting == s)].set_index('quid')[c].loc[x.index]
            m, lo, hi, p, b01, b10 = sig(x.values, y.values)
            sigs.append({'reader': f'{L2}-7B', 'metric': col, 'baseline': s, 'setting': s, 'n': len(x), 'diff': m,
                         'ci_low': lo, 'ci_high': hi, 'mcnemar_p': p, 'gains': b01, 'losses': b10})
    SM, SG = pd.DataFrame(summ), pd.DataFrame(sigs)
    out = ROOT / 'results'
    SM.to_csv(out / f'{a.out_name}_summary.csv', index=False); SG.to_csv(out / f'{a.out_name}_significance.csv', index=False)
    lines = ['# Second reader: ' + a.model + ' next to Qwen2.5-7B-Instruct-4bit', '',
             'Generated by `scripts/29_second_reader_report.py` (pre-registration: `results/prereg_second_reader.md`). '
             'First 1,000 questions of the MultiTQ reader order, prompt v3, k = 15, exact Hit@1 of the first answer; '
             'n = questions both readers have answered for every listed setting.', '',
             SM.to_markdown(index=False, floatfmt='.4f'), '']
    g = SG.copy(); g['mcnemar_p'] = g.mcnemar_p.map(lambda v: f'{v:.1e}')
    lines += [g.to_markdown(index=False, floatfmt='.4f'), '']
    (out / f'{a.out_name}.md').write_text('\n'.join(lines))
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
