from pathlib import Path
import json,hashlib,math
import pandas as pd
import numpy as np
ROOT=Path(__file__).resolve().parents[4]
OUT=Path(__file__).resolve().parent
DC=ROOT/'집/코덱스/local/ec_dc4_integration_20261002_v1'
old=ROOT/'집/클로드/research/local/ec3_DI1_all.csv'
chunks=[c[c.validator.ne('EL1')] for c in pd.read_csv(old,chunksize=2048,float_precision='round_trip')]
cache=pd.concat(chunks,ignore_index=True).set_index(['validator','validation_fold','row_id'])
cells=[]
for p in sorted(DC.glob('*_r3_*.npz')):
    parts=p.stem.split('_');v,k,_,seed=parts
    if v=='EL1':continue
    meta=json.loads(p.with_suffix('.json').read_text(encoding='utf-8'))
    assert hashlib.sha256(p.read_bytes()).hexdigest()==meta['prediction_sha256']
    with np.load(p,allow_pickle=False) as z:
        ids=z['row_id'];raw=z['raw_et']
    d=pd.DataFrame({'row_id':ids,'p':raw})
    d['farm']=d.row_id.str[:3];d['day']=d.row_id.str[4:7].astype(int);d['hour']=d.row_id.str[8:10].astype(int)
    d=d.sort_values(['farm','day','hour'])
    m=d.groupby(['farm','day']).p.transform(lambda t:t.expanding().mean())
    shrink=.5*(d.p.to_numpy()+m.to_numpy())
    reference=cache.reindex([(v,int(k),r) for r in d.row_id])[f'etS_{seed}'].to_numpy()
    assert np.isfinite(reference).all()
    gap=float(np.max(np.abs(shrink-reference)))
    assert gap<1e-12,gap
    lo,hi=meta['provenance']['train_target_bounds']
    clipped=int(np.sum((shrink<lo)|(shrink>hi)))
    assert clipped==0
    cells.append(dict(validator=v,fold=int(k),seed=int(seed),rows=len(d),maxgap=gap,clipped=clipped))
result=dict(status='PASS',n_cells=len(cells),rows=sum(c['rows'] for c in cells),maxgap=max(c['maxgap'] for c in cells),clipped_rows=sum(c['clipped'] for c in cells),scope='Original public raw ET cache vs stored DI1 final ET; no fit or EL1 scoring',cells=cells)
(OUT/'et_delta_cache_v1.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k!='cells'},ensure_ascii=False))
