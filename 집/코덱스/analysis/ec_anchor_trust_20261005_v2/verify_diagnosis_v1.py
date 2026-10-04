from pathlib import Path
import sys,json,hashlib
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;ROOT=H.parents[3]
sys.path.insert(0,str(ROOT/'집/클로드/research'));import env
import pandas as pd,numpy as np
old=json.loads((H/'case_diagnosis_v2.json').read_text(encoding='utf-8'));checks=0
for mode in ['GATE','RIDGE']:
 d=pd.read_csv(ROOT/'집/코덱스/local'/H.name/mode/'oof.csv',float_precision='round_trip');d=d[d.validator=='DIAG10'].copy();d['dm']=d.groupby(['farm','day','seed']).y.transform('mean');d['dd']=(d.candidate-d.y)**2-(d.baseline-d.y)**2
 for seg in ['high','ordinary']:
  g=d[(d.dm>=1)==(seg=='high')];r=old[mode]['direction_by_segment'][seg];assert int((g.delta!=0).sum())==r['changed_hours'] and int((g.delta<0).sum())==r['negative_hours'] and int((g.delta>0).sum())==r['positive_hours'];checks+=3
 daily=d.groupby(['farm','day','seed']).agg(delta_sse=('dd','sum'),candidate_mean=('candidate','mean'),baseline_mean=('baseline','mean'),ymean=('y','mean'))
 for r in old[mode]['selected_cases']:
  v=daily.loc[(r['farm'],r['day'],r['seed'])]
  for c in ['delta_sse','candidate_mean','baseline_mean','ymean']:assert abs(float(v[c])-r[c])<1e-12;checks+=1
  assert v.delta_sse<0
with (H/'diagnosis_crosscheck_v1.json').open('x',encoding='utf-8') as f:json.dump(dict(status='PASS_NUMPY_PANDAS_VS_SCALAR',checks=checks,case_improvements=18,source=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),diagnosis=hashlib.sha256((H/'case_diagnosis_v2.json').read_bytes()).hexdigest()),f,indent=2)
print('diagnosis independent PASS',checks)
