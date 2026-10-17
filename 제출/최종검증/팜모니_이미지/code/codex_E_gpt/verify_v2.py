import os
os.environ['OPENBLAS_NUM_THREADS']='4'
import sys,json,hashlib
from pathlib import Path
import numpy as np
OUT=Path(__file__).resolve().parent/'corrected_v3'
sys.path.insert(0,str(OUT.parent));import run_v3 as R
L=R.L;old=Path.home()/'storage/nadan/codex/im_m2_20261016_v1/preflight'
hashold=json.loads((old/'input_hashes_v1.json').read_text());checks={}
for sn in ['im_set_h3','im_set_h3j','im_set_h3s','im_set_h3g','im_set_gptr','im_set_gptr_src']+R.UN:
 for p in [L/'feats_h'/f'{sn}.npz',L/sn/'manifest.csv']:
  assert str(p) in hashold,str(p)
  checks[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()==hashold[str(p)]
assert all(checks.values()),checks
baseline={}
for sn in ['im_set_gpt1','im_set_gpt3','im_set_r2','im_set_r3','im_set_r4','pub21','jig500','sealed_real','sealed_brown','sealed_bright']:
 oldp=old/f'eval_{sn}.npz';newp=OUT/(f'logo_{sn}_eval.npz' if sn in R.POS else f'eval_{sn}.npz')
 if not newp.exists():continue
 with np.load(oldp) as a,np.load(newp) as b:
  baseline[sn]=dict(n=len(a['p9']),maxdiff=float(np.max(np.abs(a['p9']-b['p9']))),label_equal=bool(np.array_equal(a['y'],b['y'])))
  assert baseline[sn]['maxdiff']<1e-12 and baseline[sn]['label_equal'],baseline[sn]
res=json.loads((OUT/'result.json').read_text());condition={}
lg=[]
for sn in R.POS:
 with np.load(OUT/f'logo_{sn}_eval.npz') as z:
  fix=int(((z['p9']<.5)&(z['pnew']>=.5)&(z['y']==1)).sum());lg.append(fix)
  assert fix==next(r['fixed'] for r in res['logo'] if r['held']==sn)
condition['a']=sum(n>=1 for n in lg)>=3 and sum(lg)>=5
fps={};bas={};pub=0
for sn in res['metrics']:
 with np.load(OUT/f'eval_{sn}.npz') as z:
  y=z['y'];b=z['p9']>=.5;p=z['pnew']>=.5;re=y==0;fa=y==1
  assert (z['pnew']>=z['p9']).all()
  m=res['metrics'][sn];assert int((p&re).sum())==m['new_fp'];assert int((b&re).sum())==m['base_fp']
  if re.any():fps[sn]=float((p[re]).mean()-b[re].mean())
  if fa.any() and re.any():bas[sn]=float(.5*((~p[re]).mean()+p[fa].mean())-.5*((~b[re]).mean()+b[fa].mean()))
  if sn=='pub21':pub=int((p==fa).sum())
condition['b_real']=all(fps[s]<=.005+1e-12 for s in R.UN+['sealed_real','sealed_brown','sealed_bright','jig500'])
with np.load(OUT/'union_pool.npz') as z:
 m=res['budget'];delta=float(z['union'].mean()-z['O9'].mean());assert abs(delta-m['delta'])<1e-15
 assert len(z['keys'])==len(set(z['keys'].tolist()))
 condition['b_union']=delta<=.001+1e-12
condition['c_pub']=pub==21;condition['c_ba']=all(bas[s]>=-.002-1e-12 for s in ['im_set_r2','im_set_r3','im_set_r4','im_set_r5'])
for k,v in condition.items():assert v==res['conditions'][k]
proof=dict(input_hash_checks=checks,baseline=baseline,independent_conditions=condition,logo_fixed=lg,union_delta=delta,verified=True)
(OUT/'verification_v1.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(proof,ensure_ascii=False),flush=True)
