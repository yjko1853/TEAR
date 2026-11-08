"""TR-promote (within-group reorder) variant of the representative-fact rule (pre-registered:
results/prereg_tr_promote.md).

LLM-free Hit@1 on the full question sets (results/tr_promote/, scripts/06) with paired statistics of each variant
against its TR counterpart, the reader comparison on the first/last-cue samples (results/qwen_ground_promote/,
results/qwen_icews18_promote/) against the settings read earlier (results/qwen_ground/, results/qwen_icews18/), and a
diagnostic of how many other occurrences of the top-1 fact's group each setting keeps in the top-15. Scoring and
statistics are those of scripts/27_hardwindow_reader_report.py (the comparison settings are also paired against `none`
on the same questions, so that a table can give every row's difference to no reranking; exact Hit@1 of the first answer, reader+date cut by the
gold answer type, McNemar with continuity correction, paired bootstrap with 10,000 draws, generator seeded 0 per
comparison). While a reader run is incomplete, every reader number is on the questions both settings have answered
(n is reported). The reader diagnostic (tr_promote_reader_diagnostic.csv) gives, per setting and question type on the
same questions: how often the reader's first answer equals the answer read off the top-ranked fact ("follows top-1"),
its reader+date accuracy when it does and when it does not, the mean number of distinct groups among the 15 candidates,
the share of questions with a gold answer among them, and the mean number of distinct gold answers among them.

  .venv/bin/python -I scripts/28_tr_promote_report.py [--reader_only]
      -> results/tr_promote/tr_promote_report.md, tr_promote_{llmfree_pairs,reader_summary,reader_significance,
         reader_diagnostic,group_diagnostic}.csv
  --reader_only: recompute only the reader tables, re-using the LLM-free and group-diagnostic CSVs on disk.
"""
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
from tear.evaluation.hits import llmfree_rows  # noqa: E402
from tear.reader.parse import parse_list  # noqa: E402
from tear.text import normalize  # noqa: E402

TAG, MODEL = 'rule_v2_ground_exact_anydate', 'Qwen2.5-7B-Instruct-4bit'
OUT = ROOT / 'results' / 'tr_promote'
VARIANTS = [('tr', 'tr_promote'), ('tear', 'tear_promote'), ('tear_window', 'tear_window_promote')]
READER_RUNS = [  # dataset, sample file, variant pred dir, source pred dir, variant, comparison settings
    ('multitq_orig', 'sample_firstlast_cue_3000.json', 'results/qwen_ground_promote', 'results/qwen_ground',
     'tear_window_promote', ['tear_window', 'hardwindow', 'none']),
    ('multitq_orig', 'sample_first_last_1000.json', 'results/qwen_ground_promote', 'results/qwen_ground',
     'tr_promote', ['tr', 'none']),
    ('icews18', 'sample_reader2000_first_last.json', 'results/qwen_icews18_promote', 'results/qwen_icews18',
     'tear_window_promote', ['tear_window', 'hardwindow', 'none']),
]
CUE_TYPES = ('first_last', 'after_first', 'before_last')


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
    """b - a: mean difference, 95% bootstrap interval, McNemar p (continuity corrected), gains, losses."""
    a = np.asarray(a, bool); b = np.asarray(b, bool); rng = np.random.default_rng(0)
    b01 = int((~a & b).sum()); b10 = int((a & ~b).sum())
    p = math.erfc(math.sqrt(((abs(b01 - b10) - 1) ** 2 / max(1, b01 + b10)) / 2)) if b01 + b10 else 1.0
    d = b.astype(float) - a.astype(float); bs = d[rng.integers(0, len(d), (10000, len(d)))].mean(1)
    return d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5), p, b01, b10


def load_ds(ds):
    D = ROOT / 'cache' / ds
    Q = {str(q['quid']): q for q in json.load(open(D / 'questions.json'))}
    S = json.load(open(D / 'slots_rule_v2.json'))
    order = [str(x) for x in json.load(open(D / 'rerank' / 'question_order.json'))['question_ids']]
    F = np.load(D / 'facts.npz'); V = json.load(open(D / 'vocab.json'))
    return D, Q, S, order, F, V


def llmfree_pairs():
    rows = []
    for ds in ('multitq_orig', 'icews18'):
        D, Q, S, order, F, V = load_ds(ds)
        per = {}
        for base, var in VARIANTS:
            for s in (base, var):
                if s in per:
                    continue
                top = np.load(D / 'rerank' / f'{s}_{TAG}_top30_ids.npy', mmap_mode='r')[:, 0]
                r = llmfree_rows(top, order, Q, S, F, V['entities'], V['timestamps'], s)
                per[s] = pd.DataFrame(r, columns=['reranker', 'qtype', 'qlabel', 'answer_type', 'exact', 'lenient'])
        for base, var in VARIANTS:
            a, b = per[base], per[var]
            groups = [('all', np.ones(len(a), bool))] + [(t, (a.qtype == t).values) for t in sorted(a.qtype.unique())]
            for name, m in groups:
                d, lo, hi, p, g, l = sig(a.exact.values[m], b.exact.values[m])
                rows.append({'dataset': ds, 'baseline': base, 'setting': var, 'group': name, 'n': int(m.sum()),
                             'base_hit1': a.exact.values[m].mean(), 'var_hit1': b.exact.values[m].mean(), 'diff': d,
                             'ci_low': lo, 'ci_high': hi, 'mcnemar_p': p, 'gains': g, 'losses': l})
    return pd.DataFrame(rows)


def group_keys(F, V, cands, sl):
    """Group key of each candidate fact for answer slot `sl` (the slots that must stay fixed, as in TR)."""
    E, R = len(V['entities']), len(V.get('relations', V.get('predicates')))
    s_id, o_id = F['s_id'][cands], F['o_id'][cands]; r_id = (F['r_id'] if 'r_id' in F else F['p_id'])[cands]
    if sl == 'subject':
        return r_id.astype(np.int64) * E + o_id
    if sl == 'object':
        return s_id.astype(np.int64) * R + r_id
    return (s_id.astype(np.int64) * R + r_id) * E + o_id


def group_diagnostic():
    """Among first/last-cue questions: mean number of top-15 candidates in the same group as the top-1 fact, and the
    mean number of distinct groups in the top-15 (group = the slots that must stay fixed, as in TR)."""
    rows = []
    for ds in ('multitq_orig', 'icews18'):
        D, Q, S, order, F, V = load_ds(ds)
        E, R = len(V['entities']), len(V.get('relations', V.get('predicates')))
        s_id, o_id = F['s_id'], F['o_id']; r_id = F['r_id'] if 'r_id' in F else F['p_id']
        qt = np.array([Q[k]['qtype'] for k in order])
        slot = np.array([S.get(k, {}).get('answer_slot') or '' for k in order])
        for s in ['none', 'tr', 'tear', 'tear_window', 'tr_promote', 'tear_promote', 'tear_window_promote', 'hardwindow']:
            top = np.load(D / 'rerank' / f'{s}_{TAG}_top30_ids.npy', mmap_mode='r')[:, :15]
            for t in CUE_TYPES:
                m = (qt == t) & (slot != '')
                same, ngroups = [], []
                for i in np.nonzero(m)[0]:
                    c = np.asarray(top[i]); sl = slot[i]
                    if sl == 'subject':
                        key = r_id[c].astype(np.int64) * E + o_id[c]
                    elif sl == 'object':
                        key = s_id[c].astype(np.int64) * R + r_id[c]
                    else:
                        key = (s_id[c].astype(np.int64) * R + r_id[c]) * E + o_id[c]
                    same.append(int((key == key[0]).sum()) - 1); ngroups.append(len(np.unique(key)))
                rows.append({'dataset': ds, 'setting': s, 'qtype': t, 'n': int(m.sum()),
                             'other_occurrences_of_top1_group_in_top15': float(np.mean(same)),
                             'share_top1_group_alone': float(np.mean(np.array(same) == 0)),
                             'distinct_groups_in_top15': float(np.mean(ngroups))})
    return pd.DataFrame(rows)


def reader():
    summ, sigs, diag = [], [], []
    for ds, sample_file, vdir, sdir, var, comps in READER_RUNS:
        D, Q, S, order, F, V = load_ds(ds)
        pos = {k: i for i, k in enumerate(order)}; E, T = V['entities'], V['timestamps']
        sample = [str(x) for x in json.load(open(D / sample_file))['quids']]
        preds = {}
        for s, d in [(var, vdir)] + [(c, sdir) for c in comps]:
            p = ROOT / d / f'{s}_{TAG}__{MODEL}__k15__p-v3.jsonl'
            preds[s] = {json.loads(l)['quid']: json.loads(l) for l in open(p)} if p.exists() else {}
        keys = [k for k in sample if all(k in preds[s] for s in preds)]
        if not keys:
            continue
        rows = []
        for s in preds:
            top = np.load(D / 'rerank' / f'{s}_{TAG}_top30_ids.npy', mmap_mode='r')[:, :15]
            for k in keys:
                q = Q[k]; lst = parse_list(preds[s][k]['prediction']); first = lst[0] if lst else ''
                fd = cut_date(first, q['question']) if q['answer_type'] == 'time' else first
                sl = S[k]['answer_slot']; cands = np.asarray(top[pos[k]]); g = gran(q['question'])
                answers = ([T[int(t)][:g] for t in F['t_id'][cands]] if sl == 'timestamp'
                           else [E[int(e)] for e in (F['o_id'] if sl == 'object' else F['s_id'])[cands]])
                a = answers[0]
                gold = {normalize(x) for x in q['answers']}; in15 = {normalize(x) for x in answers} & gold
                rows.append({'setting': s, 'quid': k, 'qtype': q['qtype'], 'answer_type': q['answer_type'],
                             'reader': ok(first, q), 'reader_date': ok(fd, q), 'direct': ok(a, q),
                             'follows_top1': normalize(fd) == normalize(a),
                             'distinct_groups15': len(np.unique(group_keys(F, V, cands, sl))),
                             'gold_in_top15': bool(in15), 'n_gold_in_top15': len(in15)})
        df = pd.DataFrame(rows)
        for s in preds:
            for t in [None] + sorted(df.qtype.unique()):
                sub = df[df.setting == s] if t is None else df[(df.setting == s) & (df.qtype == t)]
                if len(sub) == 0:
                    continue
                f = sub.follows_top1.values.astype(bool)
                diag.append({'dataset': ds, 'sample': sample_file, 'setting': s, 'qtype': t or 'all', 'n': len(sub),
                             'reader_date': sub.reader_date.mean(), 'direct': sub.direct.mean(),
                             'follows_top1': f.mean(),
                             'reader_date_if_follows': sub.reader_date.values[f].mean() if f.any() else float('nan'),
                             'reader_date_if_not': sub.reader_date.values[~f].mean() if (~f).any() else float('nan'),
                             'distinct_groups15': sub.distinct_groups15.mean(), 'gold_in_top15': sub.gold_in_top15.mean(),
                             'n_gold_in_top15': sub.n_gold_in_top15.mean()})
        groups = [('all', None)] + [(t, t) for t in sorted(df.qtype.unique())]
        for col in ('reader', 'reader_date', 'direct'):
            for s in preds:
                sub = df[df.setting == s]
                rec = {'dataset': ds, 'sample': sample_file, 'metric': col, 'setting': s, 'n': len(sub), 'of': len(sample),
                       'hit1': sub[col].mean()}
                rec.update({f'qt_{t}': sub[sub.qtype == t][col].mean() for t in sorted(df.qtype.unique())})
                summ.append(rec)
            pairs = [(c, var) for c in comps] + [('none', c) for c in comps if c != 'none' and 'none' in comps]
            for c, v in pairs:
                for name, t in groups:
                    a = df[df.setting == c]; b = df[df.setting == v]
                    if t is not None:
                        a = a[a.qtype == t]; b = b[b.qtype == t]
                    a = a.set_index('quid')[col]; b = b.set_index('quid')[col].loc[a.index]
                    if len(a) == 0:
                        continue
                    d, lo, hi, p, g, l = sig(a.values, b.values)
                    sigs.append({'dataset': ds, 'sample': sample_file, 'metric': col, 'baseline': c, 'setting': v,
                                 'group': name, 'n': len(a), 'base_hit1': a.mean(), 'var_hit1': b.mean(), 'diff': d,
                                 'ci_low': lo, 'ci_high': hi, 'mcnemar_p': p, 'gains': g, 'losses': l})
    return pd.DataFrame(summ), pd.DataFrame(sigs), pd.DataFrame(diag)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    reader_only = '--reader_only' in sys.argv[1:]
    if reader_only and (OUT / 'tr_promote_llmfree_pairs.csv').exists():
        LP = pd.read_csv(OUT / 'tr_promote_llmfree_pairs.csv'); GD = pd.read_csv(OUT / 'tr_promote_group_diagnostic.csv')
    else:
        LP = llmfree_pairs(); LP.to_csv(OUT / 'tr_promote_llmfree_pairs.csv', index=False)
        GD = group_diagnostic(); GD.to_csv(OUT / 'tr_promote_group_diagnostic.csv', index=False)
    RS, RG, RD = reader()
    RS.to_csv(OUT / 'tr_promote_reader_summary.csv', index=False)
    RG.to_csv(OUT / 'tr_promote_reader_significance.csv', index=False)
    RD.to_csv(OUT / 'tr_promote_reader_diagnostic.csv', index=False)
    fmt = lambda df: df.to_markdown(index=False, floatfmt='.4f')
    lines = ['# TR-promote variant (within-group reorder) of the representative-fact rule', '',
             'Generated by `scripts/28_tr_promote_report.py` (pre-registration: `results/prereg_tr_promote.md`). '
             'Tag `rule_v2_ground_exact_anydate`; reader Qwen2.5-7B-Instruct-4bit, prompt v3, k = 15, exact Hit@1 of '
             'the first answer. Reader n = questions answered by every compared setting so far.', '',
             '## LLM-free Hit@1, full question sets: variant minus its TR counterpart', '']
    lp = LP.copy(); lp['mcnemar_p'] = lp.mcnemar_p.map(lambda v: f'{v:.1e}')
    lines += [fmt(lp), '', '## Other occurrences of the top-1 fact\'s group in the top-15 (first/last-cue questions)', '',
              fmt(GD), '']
    if not RS.empty:
        lines += ['## Reader on the first/last-cue samples', '', fmt(RS), '']
        rg = RG.copy(); rg['mcnemar_p'] = rg.mcnemar_p.map(lambda v: f'{v:.1e}')
        lines += [fmt(rg), '', '## Reader diagnostic on the same questions (follows top-1 = first answer equals the '
                  'answer read off the top-ranked fact; gold_in_top15 = a gold answer among the 15 candidates)', '',
                  fmt(RD), '']
    (OUT / 'tr_promote_report.md').write_text('\n'.join(lines))
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
