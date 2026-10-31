"""Final report of the MultiTQ reader experiment (original questions + KG anchor grounding, Qwen2.5-7B-Instruct-4bit,
prompt v3, k = 15, stratified 6,000-question sample, 1,000 per qtype, fixed shuffle seed 2026).

Scoring rules (Hit@1, exact match on the first listed answer; lenient = gold is a substring of it):
  reader        the reader's first answer
  reader+date   reader, with time answers cut to the granularity the question asks (year -> YYYY, month -> YYYY-MM)
  direct        LLM-free: the answer-slot value of the top-1 reranked fact (dates cut the same way)
  hybrid        pre-registered on 2026-10-17 (project log): first/last questions -> direct, otherwise
                reader+date. Designed on chunk 0 of the earlier rewrite run, so it is validated on chunks 1-5 only.
Overall numbers are on the balanced sample; 'test-weighted' re-weights the per-qtype rates by MultiTQ test shares.
Significance: McNemar (continuity corrected) and paired bootstrap (10,000 draws, seed 0) against the baseline.

  .venv/bin/python -I scripts/18_final_reader_report.py   ->  results/qwen_ground_final.md, results/qwen_ground_final_*.csv
"""
import json, math, random, re, sys, collections
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tear.reader.parse import parse_list  # noqa: E402
from tear.text import normalize  # noqa: E402
from tear.temporal import extract_temporal_cue  # noqa: E402

D = ROOT / 'cache' / 'multitq_orig'; PRED = ROOT / 'results' / 'qwen_ground'; OUT = ROOT / 'results'
TAG = 'rule_v2_ground_exact_anydate'; MODEL = 'Qwen2.5-7B-Instruct-4bit'
SETTINGS = ['none', 'timer4', 'atc', 'tr', 'tear', 'tear_window', 'timer4_public']
LABEL = {'none': 'no reranking', 'timer4': 'temporal-distance penalty (TimeR4/PoK style)', 'atc': 'ATC only',
         'tr': 'TR only', 'tear': 'TEAR (ATC+TR)', 'tear_window': 'TEAR + window',
         'timer4_public': "TimeR4's released top-15 (fine-tuned retriever + GPT-3.5 rewrite)"}

Q = {str(q['quid']): q for q in json.load(open(D / 'questions.json'))}
S = json.load(open(D / 'slots_rule_v2.json'))
order = [str(x) for x in json.load(open(D / 'rerank' / 'question_order.json'))['question_ids']]
pos = {k: i for i, k in enumerate(order)}
F = np.load(D / 'facts.npz'); V = json.load(open(D / 'vocab.json')); E, T = V['entities'], V['timestamps']
sample = [str(x) for x in json.load(open(D / 'sample_6000.json'))['quids']]
random.Random(2026).shuffle(sample)
chunk = {k: i // 1000 for i, k in enumerate(sample)}


def gran(q):
    ql = q.lower(); return 4 if re.search(r'\byear\b', ql) else (7 if re.search(r'\bmonth\b', ql) else 10)


def cut_date(p, q):
    m = re.search(r'(\d{4})(?:-(\d{2}))?(?:-(\d{2}))?', p)
    if not m: return p
    s = m.group(1) + ('-' + m.group(2) if m.group(2) else '') + ('-' + m.group(3) if m.group(3) else ''); g = gran(q)
    return s[:g] if len(s) >= g else p


def ok(pred, q, exact=True):
    return any((normalize(pred) == normalize(a)) if exact else (normalize(a) in normalize(pred)) for a in q['answers'])


def pfile(s):
    return PRED / (f'{s}__{MODEL}__k15__p-v3.jsonl' if s == 'timer4_public' else f'{s}_{TAG}__{MODEL}__k15__p-v3.jsonl')


rows = []
for s in SETTINGS:
    R = {json.loads(l)['quid']: json.loads(l) for l in open(pfile(s))}
    top = np.load(D / 'rerank' / f'{s}_{TAG}_top30_ids.npy', mmap_mode='r')[:, 0] if s != 'timer4_public' else None
    for k in sample:
        q = Q[k]; r = R[k]; lst = parse_list(r['prediction']); first = lst[0] if lst else ''
        fd = cut_date(first, q['question']) if q['answer_type'] == 'time' else first
        rec = {'setting': s, 'quid': k, 'chunk': chunk[k], 'qtype': q['qtype'], 'answer_type': q['answer_type'],
               'reader': ok(first, q), 'reader_lenient': ok(first, q, False), 'reader_date': ok(fd, q)}
        if top is not None:
            sl = S[k]['answer_slot']; fid = int(top[pos[k]])
            p = T[int(F['t_id'][fid])][:gran(q['question'])] if sl == 'timestamp' else (E[int(F['o_id'][fid])] if sl == 'object' else E[int(F['s_id'][fid])])
            rec['direct'] = ok(p, q)
            ordinal = bool(set(extract_temporal_cue(q['question'])) & {'first', 'last'})
            rec['hybrid'] = rec['direct'] if ordinal else rec['reader_date']
        rows.append(rec)
df = pd.DataFrame(rows)
df.to_csv(OUT / 'qwen_ground_final_per_question.csv', index=False)
test_share = collections.Counter(q['qtype'] for q in Q.values()); N = sum(test_share.values())


def tw(sub, col):
    g = sub.groupby('qtype')[col].mean(); return sum(g[t] * test_share[t] for t in g.index) / N


def sig(a, b, rng=np.random.default_rng(0)):
    a = np.asarray(a, bool); b = np.asarray(b, bool)
    b01 = int((~a & b).sum()); b10 = int((a & ~b).sum())
    p = math.erfc(math.sqrt(((abs(b01 - b10) - 1) ** 2 / max(1, b01 + b10)) / 2)) if b01 + b10 else 1.0
    d = b.astype(float) - a.astype(float); idx = rng.integers(0, len(d), (10000, len(d))); bs = d[idx].mean(1)
    return d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5), p


summ, sigs = [], []
for col in ('reader', 'reader_date', 'direct', 'hybrid'):
    for s in SETTINGS:
        sub = df[df.setting == s]
        if col in ('direct', 'hybrid') and s == 'timer4_public': continue
        use = sub[sub.chunk >= 1] if col == 'hybrid' else sub
        summ.append({'metric': col, 'setting': s, 'n': len(use), 'hit1': use[col].mean(), 'test_weighted': tw(use, col),
                     **{f'qt_{t}': use[use.qtype == t][col].mean() for t in sorted(test_share)}})
        for base in ('none', 'timer4', 'timer4_public'):
            if s == base or (col in ('direct', 'hybrid') and base == 'timer4_public'): continue
            bsub = df[df.setting == base]; bsub = bsub[bsub.chunk >= 1] if col == 'hybrid' else bsub
            m, lo, hi, p = sig(bsub.set_index('quid').loc[use.quid, col].values, use[col].values)
            sigs.append({'metric': col, 'setting': s, 'baseline': base, 'n': len(use), 'diff': m, 'ci_low': lo, 'ci_high': hi, 'mcnemar_p': p})
# hybrid vs reader_date within TEAR+window (the pre-registered test, chunks 1-5)
tw_sub = df[(df.setting == 'tear_window') & (df.chunk >= 1)]
m, lo, hi, p = sig(tw_sub.reader_date.values, tw_sub.hybrid.values)
sigs.append({'metric': 'hybrid_vs_reader_date', 'setting': 'tear_window', 'baseline': 'tear_window reader_date', 'n': len(tw_sub), 'diff': m, 'ci_low': lo, 'ci_high': hi, 'mcnemar_p': p})
SM = pd.DataFrame(summ); SG = pd.DataFrame(sigs)
SM.to_csv(OUT / 'qwen_ground_final_summary.csv', index=False); SG.to_csv(OUT / 'qwen_ground_final_significance.csv', index=False)

def md(d):
    cols = list(d.columns); lines = ['| ' + ' | '.join(cols) + ' |', '|' + '---|' * len(cols)]
    for _, r in d.iterrows():
        lines.append('| ' + ' | '.join(f'{v:.3f}' if isinstance(v, float) else (f'{v:.1e}' if isinstance(v, float) else str(v)) for v in r.values) + ' |')
    return '\n'.join(lines)
out = ['# MultiTQ reader experiment, final (original questions + KG anchor grounding)', '',
       'Generated by `scripts/18_final_reader_report.py`. Qwen2.5-7B-Instruct-4bit (MLX, local, zero-shot), prompt v3, '
       'k = 15, 6,000 questions (1,000 per qtype). Hit@1 exact on the first listed answer. See the script docstring '
       'for the scoring rules. The hybrid rule is evaluated on chunks 1-5 (5,000 questions) only.', '']
for col, title in (('reader', 'Reader'), ('reader_date', 'Reader + date granularity'), ('direct', 'LLM-free (top-1 fact)'), ('hybrid', 'Pre-registered hybrid (chunks 1-5)')):
    t = SM[SM.metric == col][['setting', 'n', 'hit1', 'test_weighted'] + [c for c in SM.columns if c.startswith('qt_')]].copy()
    t['setting'] = t.setting.map(LABEL)
    out += [f'## {title}', '', md(t), '']
g = SG.copy(); g['mcnemar_p'] = g.mcnemar_p.map(lambda v: f'{v:.1e}')
out += ['## Paired significance', '', md(g), '']
(OUT / 'qwen_ground_final.md').write_text('\n'.join(out))
print('\n'.join(out[:4]))
for col in ('reader_date', 'direct', 'hybrid'):
    print(col, SM[SM.metric == col][['setting', 'hit1', 'test_weighted']].round(3).to_string(index=False))
print(SG[(SG.baseline == 'none') & (SG.setting == 'tear_window') | (SG.metric == 'hybrid_vs_reader_date') | ((SG.setting == 'tear_window') & (SG.baseline.isin(['timer4', 'timer4_public'])))].round(4).to_string(index=False))
