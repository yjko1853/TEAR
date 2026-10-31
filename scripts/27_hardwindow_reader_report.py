"""Reader with the label-free hard window + anchor grounding (pre-registered: results/prereg_hardwindow_reader.md).

Splits the reader gain of TEAR+window over no reranking into the share of the hard window with grounding and the
share that ATC and TR add on top of it, on MultiTQ (6,000-question sample) and the generated ICEWS18 questions (first
2,000 of the reader order). Scoring is that of scripts/18_final_reader_report.py and
paper/derived/derive_reader_icews18.py (exact Hit@1 of the first answer; reader+date cuts time answers chosen by
the gold answer type to the granularity the question asks; direct = LLM-free top-1 fact). The statistics are those of
script 18 (McNemar with continuity correction, paired bootstrap with 10,000 draws), but every comparison draws from
its own generator seeded 0, so adding a setting here leaves the intervals of script 18 unchanged. any_match is the
list-level rule of paper/derived/derive_reader_ground.py (a gold answer anywhere in the list, lenient match).
While the run is incomplete, every number is on the questions that all settings have answered (n is reported).

  .venv/bin/python -I scripts/27_hardwindow_reader_report.py
      -> results/hardwindow_reader.md, results/hardwindow_reader_{summary,significance}.csv
"""
import collections
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
from tear.evaluation.hits import score_reader_record  # noqa: E402
from tear.reader.parse import parse_list  # noqa: E402
from tear.text import normalize  # noqa: E402

TAG, MODEL = 'rule_v2_ground_exact_anydate', 'Qwen2.5-7B-Instruct-4bit'
RUNS = {  # dataset: (prediction dir, reader slice, settings)
    'multitq_orig': ('results/qwen_ground', 6000, ['none', 'timer4', 'hardwindow', 'atc', 'tr', 'tear', 'tear_window']),
    'icews18': ('results/qwen_icews18', 2000, ['none', 'timer4', 'hardwindow', 'tear', 'tear_window']),
}
PAIRS = [('none', 'hardwindow'), ('hardwindow', 'tear_window'), ('timer4', 'hardwindow'), ('hardwindow', 'tear'),
         ('none', 'tear_window')]
METRICS = ('reader', 'reader_date', 'direct', 'any_match')
MULTI = ('equal_multi', 'after_first', 'before_last')


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
    """b - a (a = baseline): mean difference, 95% bootstrap interval, McNemar p (continuity corrected)."""
    a = np.asarray(a, bool); b = np.asarray(b, bool); rng = np.random.default_rng(0)
    b01 = int((~a & b).sum()); b10 = int((a & ~b).sum())
    p = math.erfc(math.sqrt(((abs(b01 - b10) - 1) ** 2 / max(1, b01 + b10)) / 2)) if b01 + b10 else 1.0
    d = b.astype(float) - a.astype(float); bs = d[rng.integers(0, len(d), (10000, len(d)))].mean(1)
    return d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5), p, b01, b10


def score(ds, pred_dir, n, settings):
    D = ROOT / 'cache' / ds
    Q = {str(q['quid']): q for q in json.load(open(D / 'questions.json'))}
    S = json.load(open(D / 'slots_rule_v2.json'))
    order = [str(x) for x in json.load(open(D / 'rerank' / 'question_order.json'))['question_ids']]
    pos = {k: i for i, k in enumerate(order)}
    F = np.load(D / 'facts.npz'); V = json.load(open(D / 'vocab.json')); E, T = V['entities'], V['timestamps']
    sample = [str(x) for x in json.load(open(D / 'sample_6000.json'))['quids']]
    random.Random(2026).shuffle(sample)
    sample = sample[:n]
    preds = {}
    for s in settings:
        p = ROOT / pred_dir / f'{s}_{TAG}__{MODEL}__k15__p-v3.jsonl'
        preds[s] = {json.loads(l)['quid']: json.loads(l) for l in open(p)} if p.exists() else {}
    keys = [k for k in sample if all(k in preds[s] for s in settings)]
    rows = []
    for s in settings:
        top = np.load(D / 'rerank' / f'{s}_{TAG}_top30_ids.npy', mmap_mode='r')[:, 0]
        for k in keys:
            q = Q[k]; lst = parse_list(preds[s][k]['prediction']); first = lst[0] if lst else ''
            fd = cut_date(first, q['question']) if q['answer_type'] == 'time' else first
            sl = S[k]['answer_slot']; fid = int(top[pos[k]])
            a = (T[int(F['t_id'][fid])][:gran(q['question'])] if sl == 'timestamp'
                 else (E[int(F['o_id'][fid])] if sl == 'object' else E[int(F['s_id'][fid])]))
            rows.append({'dataset': ds, 'setting': s, 'quid': k, 'qtype': q['qtype'], 'answer_type': q['answer_type'],
                         'reader': ok(first, q), 'reader_date': ok(fd, q), 'direct': ok(a, q),
                         'any_match': bool(score_reader_record(preds[s][k])['hit_any_lenient'])})
    share = collections.Counter(q['qtype'] for q in Q.values())
    return pd.DataFrame(rows), share, len(sample)


def main():
    summ, sigs = [], []
    for ds, (pred_dir, n, settings) in RUNS.items():
        df, share, n_slice = score(ds, pred_dir, n, settings)
        if df.empty:
            continue
        N = sum(share.values())
        for col in METRICS:
            for s in settings:
                sub = df[df.setting == s]
                g = sub.groupby('qtype')[col].mean()
                summ.append({'dataset': ds, 'metric': col, 'setting': s, 'n': len(sub), 'of': n_slice,
                             'hit1': sub[col].mean(),
                             'test_weighted': sum(g[t] * share[t] for t in g.index) / N if ds == 'multitq_orig' else np.nan,
                             'entity': sub[sub.answer_type == 'entity'][col].mean(),
                             'time': sub[sub.answer_type == 'time'][col].mean(),
                             'multiple': sub[sub.qtype.isin(MULTI)][col].mean(),
                             **{f'qt_{t}': g.get(t, np.nan) for t in sorted(share)}})
            for base, s in PAIRS:
                if base not in settings or s not in settings:
                    continue
                a = df[df.setting == base].set_index('quid')[col]; b = df[df.setting == s].set_index('quid')[col]
                m, lo, hi, p, b01, b10 = sig(a.values, b.loc[a.index].values)
                sigs.append({'dataset': ds, 'metric': col, 'baseline': base, 'setting': s, 'n': len(a), 'diff': m,
                             'ci_low': lo, 'ci_high': hi, 'mcnemar_p': p, 'gains': b01, 'losses': b10})
    SM, SG = pd.DataFrame(summ), pd.DataFrame(sigs)
    out = ROOT / 'results'
    SM.to_csv(out / 'hardwindow_reader_summary.csv', index=False)
    SG.to_csv(out / 'hardwindow_reader_significance.csv', index=False)
    lines = ['# Reader with the hard window + anchor grounding', '',
             'Generated by `scripts/27_hardwindow_reader_report.py` (pre-registration: '
             '`results/prereg_hardwindow_reader.md`). Qwen2.5-7B-Instruct-4bit, prompt v3, k = 15, exact Hit@1 of the '
             'first answer. n = questions answered by every setting so far.', '']
    for ds in SM.dataset.unique():
        lines += [f'## {ds}', '']
        t = SM[(SM.dataset == ds)][['metric', 'setting', 'n', 'of', 'hit1', 'test_weighted', 'entity', 'time']]
        lines += [t.to_markdown(index=False, floatfmt='.4f'), '']
        g = SG[SG.dataset == ds].drop(columns='dataset').copy()
        g['mcnemar_p'] = g.mcnemar_p.map(lambda v: f'{v:.2e}')
        lines += [g.to_markdown(index=False, floatfmt='.4f'), '']
    (out / 'hardwindow_reader.md').write_text('\n'.join(lines))
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
