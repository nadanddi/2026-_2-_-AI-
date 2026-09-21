"""Local-only, reproducible descriptive audit; never uploads source data.

Run: python analysis/profile_tabular.py
Outputs contain competition-derived information; keep analysis/local ignored.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / '온라인대회자료/정형데이터/참가자_배포'
OUT = ROOT / 'analysis/local'
TARGETS = ['sub_temp', 'sub_ec']


def load(name):
    df = pd.read_csv(DATA / name)
    keys = df.row_id.str.extract(r'^(F\d{2})_(\d{3})_(\d{2})$')
    assert not keys.isna().any().any(), f'invalid ID in {name}'
    df['farm'] = keys[0]
    df['day'] = keys[1].astype(int)
    df['hour'] = keys[2].astype(int)
    assert df.hour.between(0, 23).all()
    df['time'] = df.day * 24 + df.hour
    assert not df.row_id.duplicated().any(), f'duplicate ID in {name}'
    return df


def stats(s):
    a = s.dropna()
    if a.empty:
        return dict(n=0, missing=len(s), missing_rate=1.0, unique=0)
    return dict(n=len(a), missing=int(s.isna().sum()), missing_rate=float(s.isna().mean()),
                unique=int(a.nunique()), mean=float(a.mean()), std=float(a.std()),
                min=float(a.min()), p01=float(a.quantile(.01)), p25=float(a.quantile(.25)),
                median=float(a.median()), p75=float(a.quantile(.75)), p99=float(a.quantile(.99)),
                max=float(a.max()), zeros=int(a.eq(0).sum()), negatives=int(a.lt(0).sum()),
                integer_fraction=float(np.isclose(a, np.round(a), atol=1e-8, rtol=0).mean()))


def max_run(mask, times):
    breaks = ~mask | (times.diff().ne(1))
    return int(mask.groupby(breaks.cumsum()).sum().max()) if len(mask) else 0


def profile_group(g, cols):
    g = g.sort_values('time')
    result = dict(n=len(g), start_day=int(g.day.min()), end_day=int(g.day.max()),
                  gaps=int(g.time.diff().gt(1).sum()),
                  missing_hours=int(g.time.max()-g.time.min()+1-len(g)),
                  columns={})
    for c in cols:
        s = g[c]
        z = stats(s)
        z['max_missing_run_hours'] = max_run(s.isna(), g.time)
        eq = s.eq(s.shift()) & s.notna() & g.time.diff().eq(1)
        z['equal_adjacent_pairs'] = int(eq.sum())
        z['max_constant_run_hours'] = int(eq.groupby((~eq).cumsum()).sum().max())+1 if s.notna().any() else 0
        result['columns'][c] = z
    return result


def correlation(a, b):
    mask = a.notna() & b.notna()
    if mask.sum() < 5 or a[mask].nunique() < 2 or b[mask].nunique() < 2:
        return None
    return float(a[mask].corr(b[mask]))


def psi(a, b):
    a, b = a.dropna(), b.dropna()
    if len(a) < 20 or len(b) < 20:
        return None
    edges = np.unique(np.quantile(a, np.linspace(0, 1, 11)))
    if len(edges) < 3:
        return None
    edges[0], edges[-1] = -np.inf, np.inf
    pa = np.maximum(np.histogram(a, edges)[0]/len(a), 1e-5)
    pb = np.maximum(np.histogram(b, edges)[0]/len(b), 1e-5)
    return float(np.sum((pb-pa)*np.log(pb/pa)))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    x, y, test, sample = [load(n) for n in ['train_X.csv','train_y.csv','test_X.csv','sample_submission.csv']]
    cols = list(x.columns[1:20])
    assert len(cols) == 19 and list(test.columns) == list(x.columns)
    assert set(y.row_id) <= set(x.row_id)
    assert set(x.row_id).isdisjoint(test.row_id)
    assert test.row_id.equals(sample.row_id)
    merged = x.merge(y[['row_id']+TARGETS], on='row_id', how='left', validate='one_to_one')
    farms = sorted(test.farm.unique())
    r = dict(source_sha256={n:hashlib.sha256((DATA/n).read_bytes()).hexdigest() for n in
             ['train_X.csv','train_y.csv','test_X.csv','sample_submission.csv']},
             sizes=dict(train_x=len(x), train_y=len(y), test_x=len(test), farms=x.farm.nunique(),
                        unlabeled_x=len(x)-len(y)), test_farms=farms,
             train_stats={c:stats(merged[c]) for c in cols+TARGETS},
             test_stats={c:stats(test[c]) for c in cols},
             farm_stats={f:profile_group(g, cols+TARGETS) for f,g in merged.groupby('farm')},
             test_farm_stats={f:profile_group(g, cols) for f,g in test.groupby('farm')})
    print('Computed global and farm statistics', flush=True)
    r['target_coverage'] = {}
    for f,g in merged.groupby('farm'):
        r['target_coverage'][f] = {c:dict(n=int(g[c].notna().sum()),
                 start_day=int(g.loc[g[c].notna(),'day'].min()) if g[c].notna().any() else None,
                 end_day=int(g.loc[g[c].notna(),'day'].max()) if g[c].notna().any() else None)
                 for c in TARGETS}
    # Exact feature-vector repeats (not automatically invalid observations).
    hashes = pd.util.hash_pandas_object(x[cols], index=False)
    r['repeated_vectors'] = dict(extra_rows=int(hashes.duplicated().sum()),
                                all_members=int(hashes.duplicated(False).sum()),
                                largest_groups=hashes.value_counts().head(10).tolist(),
                                train_test_matches=int(pd.util.hash_pandas_object(test[cols],index=False).isin(hashes).sum()))
    repeat_groups = pd.DataFrame({'h':hashes,'farm':x.farm}).groupby('h').farm.nunique()
    r['repeated_vectors']['cross_farm_groups'] = int(repeat_groups.gt(1).sum())
    r['missing_patterns'] = x[cols].isna().astype(int).astype(str).agg(''.join,axis=1).value_counts().head(20).to_dict()
    r['bounds'] = {}
    for c in cols:
        if c.endswith('_hum') or c.startswith('act_'):
            bad = merged[c].notna() & ~merged[c].between(0,100)
        elif c in ['out_rad','in_rad','out_wspd','in_co2']:
            bad = merged[c].lt(0)
        else:
            bad = merged[c].notna() & ~merged[c].between(-40,70)
        r['bounds'][c] = dict(n=int(bad.sum()), by_farm=merged.loc[bad].groupby('farm').size().to_dict())
    r['correlations'] = {}
    for f in ['ALL']+farms:
        g = merged if f=='ALL' else merged[merged.farm==f]
        r['correlations'][f] = {c:{t:correlation(g[c],g[t]) for t in TARGETS} for c in cols+['day','hour']}
    r['strong_feature_pairs'] = []
    cc = merged[cols].corr()
    for i,a in enumerate(cols):
        for b in cols[i+1:]:
            if abs(cc.loc[a,b]) > .9:
                r['strong_feature_pairs'].append(dict(a=a,b=b,r=float(cc.loc[a,b])))
    r['temporal'] = {}
    r['distribution_shift'] = {}
    for f in farms:
        g = merged[merged.farm==f].sort_values('time').set_index('time',drop=False)
        te = test[test.farm==f].sort_values('time')
        r['distribution_shift'][f] = {c:dict(psi=psi(g[c],te[c]),
             train_missing=float(g[c].isna().mean()),test_missing=float(te[c].isna().mean()),
             test_outside_train_range=int(((te[c]<g[c].min())|(te[c]>g[c].max())).sum()),
             train_mean=float(g[c].mean()),test_mean=float(te[c].mean())) for c in cols}
        temporal = dict(train_days=sorted(g.day.unique().tolist()),test_days=sorted(te.day.unique().tolist()),
                        lags={},hourly={},daily={})
        for t in TARGETS:
            temporal['lags'][t] = {}
            for c in cols+[t]:
                vals = {}
                for lag in [0,1,2,3,6,12,24,48,72,168]:
                    previous = g[c].reindex(g.index-lag)
                    previous.index = g.index
                    vals[str(lag)] = dict(r=correlation(previous,g[t]), n=int((previous.notna()&g[t].notna()).sum()))
                temporal['lags'][t][c] = vals
            temporal['hourly'][t] = g.groupby('hour')[t].agg(['count','mean','std']).round(6).to_dict('index')
            temporal['daily'][t] = g.groupby('day')[t].agg(['count','mean','std','min','max']).round(6).to_dict('index')
        r['temporal'][f] = temporal
    # Each source column's precision, repeated triplets, and long constant stretches.
    r['precision'] = {}
    for c in cols+TARGETS:
        values = merged[c].dropna()
        r['precision'][c] = {str(d):float(np.isclose(values,values.round(d),atol=1e-8,rtol=0).mean()) if len(values) else None for d in [0,1,2,3,4]}
    r['linear_triplets'] = {}
    for f,g in merged.sort_values(['farm','time']).groupby('farm'):
        consecutive = g.time.diff().eq(1)&g.time.diff().shift().eq(1)
        r['linear_triplets'][f] = {}
        for c in cols:
            valid = consecutive & g[c].notna() & g[c].shift().notna() & g[c].shift(2).notna()
            changes = g[c].diff()
            linear = valid & changes.sub(changes.shift()).abs().lt(1e-8) & changes.abs().gt(1e-8)
            r['linear_triplets'][f][c] = dict(n=int(linear.sum()),possible=int(valid.sum()))
    # Save full numeric audit locally; console emits only selected aggregate results.
    def clean(v):
        if isinstance(v,dict): return {str(k):clean(w) for k,w in v.items()}
        if isinstance(v,(list,tuple)): return [clean(w) for w in v]
        if isinstance(v,np.integer): return int(v)
        if isinstance(v,(float,np.floating)): return float(v) if np.isfinite(v) else None
        return v
    (OUT/'profile.json').write_text(json.dumps(clean(r),ensure_ascii=False,indent=2),encoding='utf-8')
    farm_rows=[]
    for f,p in r['farm_stats'].items():
        row={'farm':f,**{k:v for k,v in p.items() if k!='columns'}}
        for c,s in p['columns'].items():
            for k in ['n','missing_rate','unique','mean','std','min','max','max_constant_run_hours']:
                row[c+'__'+k]=s.get(k)
        farm_rows.append(row)
    pd.DataFrame(farm_rows).to_csv(OUT/'farm_profile.csv',index=False,encoding='utf-8-sig')
    print(json.dumps(clean(dict(sizes=r['sizes'],test_farms=farms,repeated_vectors=r['repeated_vectors'],
          target_coverage={f:p for f,p in r['target_coverage'].items() if p['sub_ec']['n']>0},
          bounds=r['bounds'],strong_feature_pairs=r['strong_feature_pairs'])),ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
