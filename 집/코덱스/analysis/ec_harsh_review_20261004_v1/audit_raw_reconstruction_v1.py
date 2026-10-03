from pathlib import Path
import json,math,hashlib
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[4]
OUT=Path(__file__).resolve().parent
CACHE=ROOT/'집/코덱스/local/ec_dc4_integration_20261002_v1'
ref=pd.read_csv(CACHE/'v2_integration_oof.csv',float_precision='round_trip')

def load(name):
    path=CACHE/name
    meta=json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
    assert hashlib.sha256(path.read_bytes()).hexdigest()==meta['prediction_sha256']
    with np.load(path,allow_pickle=False) as z:
        arr={k:z[k] for k in z.files if k in ['row_id','raw_et','raw_r3','raw_lgb','raw_mlp','raw_pfn']}
    return arr,meta

def smooth(raw,frame):
    x=frame[['farm','day','hour']].copy().reset_index(drop=True)
    x['p']=raw
    x=x.sort_values(['farm','day','hour'])
    x['m']=x.groupby(['farm','day']).p.transform(lambda s:s.expanding().mean())
    result=np.empty(len(x));result[x.index]=.5*(x.p+x.m)
    scalar=np.empty(len(x))
    for _,g in x.groupby(['farm','day']):
        history=[]
        for i,p in zip(g.index,g.p):
            history.append(float(p));scalar[i]=.5*(float(p)+math.fsum(history)/len(history))
    assert np.max(np.abs(result-scalar))<1e-12
    return result

cells=[]
for (v,k),d in ref.groupby(['validator','validation_fold'],sort=False):
    assert v!='EL1'
    bag=[];ids=None
    for s in [1,2,3,4]:
        a,m=load(f'{v}_{k}_pfn_{s}.npz')
        if ids is None:ids=a['row_id']
        assert np.array_equal(ids,a['row_id'])
        bag.append(a['raw_pfn'])
    bag=np.mean(bag,axis=0)
    for s in [7,101,2024]:
        a,m=load(f'{v}_{k}_r3_{s}.npz')
        assert np.array_equal(ids,a['row_id'])
        assert np.array_equal(a['raw_r3'],.6*a['raw_et']+.3*a['raw_lgb']+.1*a['raw_mlp'])
        g=d[d.seed.eq(s)].set_index('row_id').reindex(ids).reset_index()
        assert len(g)==len(ids) and g.season_v2.notna().all()
        raw=.8*a['raw_r3']+.2*bag
        sm=smooth(raw,g)
        lo,hi=m['provenance']['train_target_bounds']
        reconstructed=np.clip(sm,lo,hi)
        maxgap=float(np.max(np.abs(reconstructed-g.season_v2.to_numpy())))
        assert maxgap<1e-12,maxgap
        cells.append(dict(validator=v,fold=int(k),seed=s,n=len(g),clipped=int(np.count_nonzero((sm<lo)|(sm>hi))),max_reconstruction_gap=maxgap,clip_gap=float(np.max(np.abs(sm-reconstructed)))))
result=dict(status='PASS',scope='Public 22-fold original member caches, no labels newly read or models fitted.',n_cells=len(cells),rows=sum(c['n'] for c in cells),clipped_rows=sum(c['clipped'] for c in cells),max_reconstruction_gap=max(c['max_reconstruction_gap'] for c in cells),max_clip_gap=max(c['clip_gap'] for c in cells),cells=cells)
(OUT/'raw_reconstruction_v1.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k!='cells'},ensure_ascii=False))
