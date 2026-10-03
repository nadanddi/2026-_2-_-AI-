"""Claude context reload: recompute stored public OOF, with scalar cross-checks.

No fitting, raw labels, EL1 scoring, test prediction, or adoption.
"""
from pathlib import Path
import sys
import json
import math
import hashlib

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / '집/클로드/research'))
import env
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
SRC = ROOT / '집/클로드/research/local'
DEST = HERE / 'reload_verification_v3.json'
assert not DEST.exists()
SEEDS = [7, 101, 2024]
VALIDATORS = ['DIAG10', 'A', 'B', 'EXT10', 'EXT12']
result = {'scope': 'cached public OOF only; no refit or EL1 scoring',
          'claude_git_checked': '4ab2a52', 'experiments': {}, 'checks': 0}

def close(a, b, tol=1e-12):
    assert abs(float(a) - float(b)) < tol, (a, b)
    result['checks'] += 1

def read_public(path):
    d = pd.concat([c[c.validator.isin(VALIDATORS)] for c in
                   pd.read_csv(path, chunksize=2048, float_precision='round_trip')],
                  ignore_index=True)
    assert not d.duplicated(['validator', 'validation_fold', 'row_id']).any()
    assert set(d.validator) == set(VALIDATORS)
    return d

for tag, prefix in [('HG1', 'hg'), ('HM1', 'hm'), ('HM2', 'hm'),
                    ('DC7', 'dc7'), ('TF1', 'tf')]:
    path = SRC / f'ec3_{tag}_all.csv'
    d = read_public(path)
    cells = []
    for v in VALIDATORS:
        g = d[d.validator == v]
        for seed in SEEDS:
            ea = g[f'r3s_{seed}'].to_numpy() - g.sub_ec.to_numpy()
            eb = g[f'{prefix}_{seed}'].to_numpy() - g.sub_ec.to_numpy()
            assert np.isfinite(ea).all() and np.isfinite(eb).all()
            a, b = np.sqrt(np.mean(ea ** 2)), np.sqrt(np.mean(eb ** 2))
            close(a, math.sqrt(math.fsum(float(e) ** 2 for e in ea) / len(ea)))
            close(b, math.sqrt(math.fsum(float(e) ** 2 for e in eb) / len(eb)))
            cells.append({'validator': v, 'seed': seed, 'n': len(g),
                          'baseline_rmse': float(a), 'candidate_rmse': float(b),
                          'change_pct': float(100 * (b / a - 1))})
    diag = d[d.validator == 'DIAG10'].copy()
    rng = np.random.default_rng(20261004)
    p_worse = []
    for seed in SEEDS:
        delta = ((diag[f'{prefix}_{seed}'] - diag.sub_ec) ** 2
                 - (diag[f'r3s_{seed}'] - diag.sub_ec) ** 2)
        cl = delta.groupby(diag.farm + '_' + (diag.day // 5).astype(str)).agg(['sum', 'count'])
        ix = rng.integers(0, len(cl), (20000, len(cl)))
        boot = cl['sum'].to_numpy()[ix].sum(1) / cl['count'].to_numpy()[ix].sum(1)
        p_worse.append(float((boot >= 0).mean()))
    result['experiments'][tag] = {
        'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'cells': cells,
        'p_worse': p_worse,
        'all_primary_cells_improve': all(c['change_pct'] < 0 for c in cells
                                        if c['validator'] in ['DIAG10', 'A', 'B'])}
    if tag == 'HM1':
        high = diag.groupby(['farm', 'day']).sub_ec.transform('mean') >= 1
        flag = diag.flag_7.astype(bool)
        flagged = {}
        for name, mask in [('true_high', flag & high), ('false_high', flag & ~high)]:
            g = diag[mask]
            base = (g.r3s_7 - g.sub_ec).to_numpy()
            cand = (g.hm_7 - g.sub_ec).to_numpy()
            bs, cs = np.sum(base ** 2), np.sum(cand ** 2)
            close(bs, math.fsum(float(e) ** 2 for e in base))
            close(cs, math.fsum(float(e) ** 2 for e in cand))
            flagged[name] = {'n': len(g), 'baseline_sse': float(bs), 'candidate_sse': float(cs)}
        result['HM1_flagged'] = flagged

dc5 = SRC / 'ec2_DC5_oof.csv'
d = pd.concat([c[c.validator == 'DIAG10'] for c in
               pd.read_csv(dc5, chunksize=2048, float_precision='round_trip')], ignore_index=True)
d['prediction'] = d[[f'r3s_{s}' for s in SEEDS]].mean(axis=1)
t = d.groupby(['farm', 'day'])[['sub_ec', 'prediction']].mean()
manual_days = []
for key, g in d.groupby(['farm', 'day']):
    y = math.fsum(float(x) for x in g.sub_ec) / len(g)
    p = math.fsum(float(x) for x in g.prediction) / len(g)
    close(y, t.loc[key, 'sub_ec'])
    close(p, t.loc[key, 'prediction'])
    manual_days.append((y, p))
h = t[t.sub_ec >= 1]
e = (h.prediction - h.sub_ec).to_numpy()
mse, shift, var = np.mean(e ** 2), np.mean(e) ** 2, np.var(e)
close(mse, shift + var)
close(mse, math.fsum(float(x) ** 2 for x in e) / len(e))
close(shift, (math.fsum(float(x) for x in e) / len(e)) ** 2)
assert len(h) == sum(y >= 1 for y, _ in manual_days)
result['HC2'] = {'all_days': len(t), 'high_days': len(h),
                 'actual_high_mean': float(h.sub_ec.mean()),
                 'predicted_high_mean': float(h.prediction.mean()),
                 'common_underprediction_share': float(shift / mse),
                 'across_day_residual_share': float(var / mse),
                 'source_sha256': hashlib.sha256(dc5.read_bytes()).hexdigest()}

old_public = json.loads((HERE / 'verified_public_v1.json').read_text(encoding='utf-8'))
old_followup = json.loads((ROOT / '집/코덱스/analysis/ec_routing_trace_20261004_v1/followup_verification_v1.json').read_text(encoding='utf-8'))
for tag, rec in result['experiments'].items():
    old = old_followup['TF1'] if tag == 'TF1' else old_public['experiments'][tag]
    for seed, p in zip(SEEDS, rec['p_worse']):
        close(p, old['p_worse'][SEEDS.index(seed)])
    for cell in rec['cells']:
        match = [c for c in old['cells'] if c['validator'] == cell['validator'] and c['seed'] == cell['seed']]
        assert len(match) == 1
        close(cell['change_pct'], match[0]['change_pct'])
    if tag == 'TF1':
        assert rec['sha256'] == old['source_sha']
    else:
        assert rec['sha256'] == old_public['sha256'][f'ec3_{tag}_all.csv']
close(result['HC2']['common_underprediction_share'], old_public['HC2']['common_underprediction_share'])
close(result['HC2']['across_day_residual_share'], old_public['HC2']['across_day_residual_share'])
result['status'] = 'PASS'
result['limits'] = [
    'R3S baseline differs from actual submission season-v2; no adoption.',
    'Repeated public-validation exploration is not untouched holdout evidence.',
    'External transfer experiments are read as source/log reports, not independently retrained.',
    'Failed candidates do not prove absent input information or agricultural causation.'
]
DEST.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps({'status': result['status'], 'checks': result['checks'],
                  'HC2': result['HC2'], 'HM1_flagged': result['HM1_flagged'],
                  'experiments': {k: {'DIAG_pct': [c['change_pct'] for c in v['cells'] if c['validator'] == 'DIAG10'],
                                      'p_worse': v['p_worse'],
                                      'all_primary_cells_improve': v['all_primary_cells_improve']}
                                  for k, v in result['experiments'].items()}}, ensure_ascii=False, indent=2))
