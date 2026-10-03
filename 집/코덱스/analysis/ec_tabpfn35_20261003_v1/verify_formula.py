from pathlib import Path
import sys,math,json
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;ROOT=H.parents[3]
sys.path.insert(0,str(ROOT/'집/클로드/research'));import env
import numpy as np,pandas as pd
OUT=ROOT/'집/코덱스/local/ec_tabpfn35_20261003_v1';paths=list(OUT.glob('*_pred.csv'));assert len(paths)==66
checks=0;maximum=0.;rows=0
for path in paths:
 d=pd.read_csv(path,float_precision='round_trip');v,k,seed=d.validator.iloc[0],int(d.fold.iloc[0]),int(d.seed.iloc[0]);old=dict(np.load(OUT/f'{v}_{k}_baseline.npz'));assert np.array_equal(old['row_id'],d.row_id)
 new=[]
 for c in [1,2,3,4]:
  z=dict(np.load(OUT/f'{v}_{k}_new_pfn_{c}.npz'));assert np.array_equal(z['row_id'],d.row_id) and np.array_equal(z['context_row_id'],old[f'context_{c}']);new.append(z['prediction']);checks+=2
 bag=np.mean(new,axis=0);assert np.array_equal(bag,d.new_pfn_raw);assert np.array_equal(old[f'r3_{seed}'],d.r3_raw);assert np.array_equal(old[f'baseline_{seed}'],d.baseline);checks+=3
 for key,g in d.groupby(['farm','day']):
  past=[]
  for r in g.sort_values('hour').itertuples():
   raw=.8*r.r3_raw+.2*r.new_pfn_raw;past.append(raw);sm=.5*raw+.5*math.fsum(past)/len(past);expected=min(old['hi'],max(old['lo'],sm));diff=abs(expected-r.candidate);assert diff<1e-12;assert abs((sm-r.baseline)-r.correction)<1e-12;maximum=max(maximum,diff);checks+=2
 rows+=len(d)
assert rows==83160
(H/'formula_verification_v1.json').write_text(json.dumps(dict(status='PASS',checks=checks,rows=rows,scalar_clipped_maxdiff=maximum,scope='all raw-member weights/prefix smoothing/clip/cache/context identities; no model refitting'),indent=2),encoding='utf-8')
print('PASS',checks,maximum)
