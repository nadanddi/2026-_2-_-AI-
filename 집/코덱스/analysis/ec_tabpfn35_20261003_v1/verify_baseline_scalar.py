from pathlib import Path
import sys,json,math
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;ROOT=H.parents[3]
sys.path.insert(0,str(ROOT/'집/클로드/research'));import env
import numpy as np
OUT=ROOT/'집/코덱스/local/ec_tabpfn35_20261003_v1';paths=list(OUT.glob('*_baseline.npz'));assert len(paths)==22
checks=0;maximum=0.
for path in paths:
 with np.load(path) as z:d={k:z[k] for k in z.files}
 order=np.argsort(d['row_id'])
 for seed in [7,101,2024]:
  previous={}
  for i in order:
   key=str(d['row_id'][i])[:7];raw=.8*float(d[f'r3_{seed}'][i])+.2*float(d['old_pfn_raw'][i]);past=previous.setdefault(key,[]);past.append(raw)
   expected=min(float(d['hi']),max(float(d['lo']),.5*raw+.5*math.fsum(past)/len(past)));delta=abs(expected-float(d[f'baseline_{seed}'][i]));assert delta<1e-12;maximum=max(maximum,delta);checks+=1
assert checks==83160
(H/'baseline_scalar_verification_v1.json').write_text(json.dumps(dict(status='PASS',checks=checks,folds=22,scalar_baseline_maxdiff=maximum,scope='all public saved V2 raw-weight/prefix/clip replay by scalar fsum, separate from preparation pipeline'),indent=2),encoding='utf-8');print('PASS',checks,maximum)
