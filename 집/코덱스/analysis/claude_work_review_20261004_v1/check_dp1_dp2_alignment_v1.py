"""Preserved code for public-only DP1/DP2 alignment executed from stdin on Oct4."""
from pathlib import Path
import sys, json
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'집/클로드/research'));import env
sys.path.insert(0,str(ROOT/'집/코덱스/analysis/statistical_experiments_20261003_v1'));import support as S
import pandas as pd
import numpy as np
p=Path(__file__).resolve().parent
lab,core,wv,folds,outer=S.loadec()
vs=['DIAG10','A','B','EXT10','EXT12']
def load(tag):
    return pd.concat([x[x.validator.isin(vs)] for x in pd.read_csv(Path(env.LOCAL)/f'ec3_{tag}_all.csv',chunksize=2048,float_precision='round_trip')],ignore_index=True)
a,b=load('DP1'),load('DP2');keys=['validator','validation_fold','row_id']
a=a.set_index(keys).sort_index();b=b.set_index(keys).sort_index()
assert a.index.equals(b.index)
checks=0
for c in ['farm','day','hour','sub_ec']:
    assert np.array_equal(a[c],b[c]);checks+=len(a)
for seed in [7,101,2024]:
    assert np.max(np.abs(a[f'r3s_{seed}']-b[f'r3s_{seed}']))<1e-12;checks+=len(a)
by=lab.set_index('row_id').sub_ec
for d in [a,b]:
    yy=by.reindex(d.index.get_level_values('row_id')).to_numpy()
    assert np.isfinite(yy).all() and np.max(np.abs(yy-d.sub_ec.to_numpy()))<1e-12;checks+=len(d)
res={'status':'PASS','checks':checks,'public_rows_each':len(a),'matching':'same ordered keys/labels/farm/day/hour/three R3S baselines; both labels agree with prior permitted public cache', 'limitations':['Paired outputs do not identify individual feature contributions; six features removed simultaneously and tree randomness may change.','No refit/no EL1 scoring/no new candidate.']}
q=p/'dp1_dp2_alignment_v1.json';assert not q.exists();q.write_text(json.dumps(res,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(res,ensure_ascii=False))
