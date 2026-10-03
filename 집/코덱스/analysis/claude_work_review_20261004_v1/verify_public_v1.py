"""Cached public OOF only; no fit, raw labels, lock or EL1 scoring."""
from pathlib import Path
import sys, math, json, hashlib
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / '집/클로드/research'))
import env
import numpy as np
import pandas as pd
OUT = Path(__file__).parent
SRC = ROOT / '집/클로드/research/local'
result = {'experiments': {}, 'sha256': {}, 'scope': 'public cached OOF only; EL1 excluded'}
seeds = [7, 101, 2024]
for tag, prefix in [('HG1','hg'), ('HM1','hm'), ('HM2','hm'), ('DC7','dc7')]:
    path = SRC / ('ec3_' + tag + '_all.csv')
    result['sha256'][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    chunks = [x[x.validator != 'EL1'] for x in pd.read_csv(path, chunksize=2048)]
    d = pd.concat(chunks, ignore_index=True)
    assert not d.duplicated(['validator','validation_fold','row_id']).any()
    cells = []
    for v in ['DIAG10','A','B','EXT10','EXT12']:
        g = d[d.validator == v]
        for s in seeds:
            a = g[f'r3s_{s}'].to_numpy() - g.sub_ec.to_numpy()
            b = g[f'{prefix}_{s}'].to_numpy() - g.sub_ec.to_numpy()
            assert np.isfinite(a).all() and np.isfinite(b).all()
            ra, rb = np.sqrt(np.mean(a*a)), np.sqrt(np.mean(b*b))
            for e, r in [(a,ra),(b,rb)]:
                assert abs(math.sqrt(math.fsum(float(z)*float(z) for z in e)/len(e)) - r) < 1e-12
            cells.append({'validator':v,'seed':s,'n':len(g),'baseline':float(ra),'candidate':float(rb),'change_pct':float(100*(rb/ra-1))})
    diag = d[d.validator == 'DIAG10'].copy()
    rng = np.random.default_rng(20261004)
    ps = []
    for s in seeds:
        diag['delta'] = (diag[f'{prefix}_{s}']-diag.sub_ec)**2-(diag[f'r3s_{s}']-diag.sub_ec)**2
        cl = diag.groupby(diag.farm+'_'+(diag.day//5).astype(str)).delta.agg(['sum','count'])
        ix = rng.integers(0,len(cl),(20000,len(cl)))
        ps.append(float((cl['sum'].to_numpy()[ix].sum(1)/cl['count'].to_numpy()[ix].sum(1) >= 0).mean()))
    result['experiments'][tag] = {'cells':cells,'p_worse':ps,'all_DIAG_A_B_better':all(c['change_pct'] < 0 for c in cells if c['validator'] in ['DIAG10','A','B'])}
    if tag == 'HM1':
        hi = diag.groupby(['farm','day']).sub_ec.transform('mean') >= 1
        flagged = diag.flag_7.astype(bool)
        result['HM1_flagged'] = {}
        for name, m in [('true_high',flagged & hi),('false_high',flagged & ~hi)]:
            g = diag[m]
            result['HM1_flagged'][name] = {'n':len(g),'baseline_sse':float(((g.r3s_7-g.sub_ec)**2).sum()),'candidate_sse':float(((g.hm_7-g.sub_ec)**2).sum())}
d = pd.read_csv(SRC/'ec2_DC5_oof.csv')
d = d[d.validator == 'DIAG10'].copy()
d['p'] = d[[f'r3s_{s}' for s in seeds]].mean(axis=1)
days = []
for (f,day),g in d.groupby(['farm','day']):
    e = g.p.to_numpy()-g.sub_ec.to_numpy()
    mse = float(np.mean(e*e)); bias = float(e.mean()**2); var = float(e.var())
    assert abs(mse-bias-var)<1e-12
    manual = [float(p)-float(y) for p,y in zip(g.p,g.sub_ec)]
    assert abs(math.fsum(z*z for z in manual)/len(manual)-mse)<1e-12
    days.append({'farm':f,'day':int(day),'y':float(g.sub_ec.mean()),'p':float(g.p.mean()),'mse':mse,'bias2':bias,'var':var,'n':len(g)})
t = pd.DataFrame(days); h = t[t.y >= 1]
assert len(h)==sum(x['y'] >= 1 for x in days)
e = h.p.to_numpy()-h.y.to_numpy()
daily_mse = float(np.mean(e*e))
common_bias2 = float(e.mean()**2)
across_day_var = float(e.var())
manual = [float(x['p'])-float(x['y']) for x in days if x['y'] >= 1]
assert abs(math.fsum(z*z for z in manual)/len(manual)-daily_mse)<1e-12
assert abs((math.fsum(manual)/len(manual))**2-common_bias2)<1e-12
assert abs(daily_mse-common_bias2-across_day_var)<1e-12
result['HC2'] = {'high_days':len(h),'all_days':len(t),'actual_mean':float(h.y.mean()),'predicted_mean':float(h.p.mean()),'daily_mean_error_mse':daily_mse,'common_underprediction_share':common_bias2/daily_mse,'across_day_residual_share':across_day_var/daily_mse,'high_share_of_all_daily_mean_squared_errors':float(np.sum(e*e)/np.sum((t.p-t.y)**2)), 'separate_hourly_error_decomposition':{'daily_offset_share':float(h.bias2.sum()/h.mse.sum()),'within_day_shape_share':float(h['var'].sum()/h.mse.sum()),'high_share_all_hourly_sse':float((h.mse*h.n).sum()/(t.mse*t.n).sum())}}
(OUT/'verified_public_v1.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'HC2':result['HC2'],'HM1_flagged':result['HM1_flagged'],'experiments':{k:{'DIAG_pct':[c['change_pct'] for c in v['cells'] if c['validator']=='DIAG10'],'p_worse':v['p_worse'],'all_DIAG_A_B_better':v['all_DIAG_A_B_better']} for k,v in result['experiments'].items()}},ensure_ascii=False,indent=2))
