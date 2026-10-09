"""Empirical check of a classifier trained exclusively on positive days."""
from pathlib import Path
import sys,json,math
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;ROOT=H.parents[3];OLD=ROOT/'집/코덱스/analysis/ec_highday_classifier_20261009_v1';LOO=ROOT/'집/코덱스/analysis/ec_highday_positive_loo_20261009_v1';sys.path.insert(0,str(ROOT/'집/클로드/research'));import env
sys.path.insert(0,str(OLD));import runtime_v1,recipe_v1 as C,run_v1 as P
import numpy as np,pandas as pd
from threadpoolctl import threadpool_limits
L=ROOT/'집/코덱스/local'/H.name

def prepare():
 P.check();reg=json.loads((LOO/'registration_v1.json').read_text(encoding='utf8'));meta=pd.read_csv(P.L/'metadata_public_v1.csv');folds=[]
 for r in reg['folds']:
  idx=[i for i in r['train_indices'] if int(meta.high.iloc[i])==1];assert idx and set(meta.high.iloc[idx])=={1};assert len(idx)//24==r['train_high'];folds.append(dict(key=r['key'],train_rows=len(idx),train_days=len(idx)//24,classes=[1]))
 pins={str(p.relative_to(ROOT)):P.sha(p) for p in [P.L/'features_public_v1.csv',P.L/'metadata_public_v1.csv',LOO/'registration_v1.json',OLD/'recipe_v1.py',H/'PLAN_v1.md',H/'run_v1.py']}
 P.save(H/'registration_v1.json',dict(source_pins=pins,all26_single_class_folds=folds,seeds=list(C.SEEDS),actual_fold='LOO_000',adoption=False))

def run(seed):
 P.check();reg=json.loads((H/'registration_v1.json').read_text(encoding='utf8'))
 for name,digest in reg['source_pins'].items():assert P.sha(ROOT/name)==digest
 f=pd.read_csv(P.L/'features_public_v1.csv',float_precision='round_trip');meta=pd.read_csv(P.L/'metadata_public_v1.csv');loo=json.loads((LOO/'registration_v1.json').read_text(encoding='utf8'));r=loo['folds'][0];assert r['key']=='LOO_000'
 tr=[i for i in r['train_indices'] if int(meta.high.iloc[i])==1];pos=r['query_indices'];neg=meta.index[meta.high==0].tolist();assert set(tr).isdisjoint(pos) and set(tr).isdisjoint(neg) and set(meta.high.iloc[tr])=={1}
 model=C.factory('et',seed)
 with threadpool_limits(limits=2):model.fit(f[C.COLS].iloc[tr],meta.high.iloc[tr],clf__sample_weight=np.full(len(tr),1/24))
 assert list(model.classes_)==[1];idx=pos+neg;q=meta.iloc[idx][['row_id','farm','day','hour','high']].copy();q['score']=C.positive_score(model,f[C.COLS].iloc[idx]);q['predicted']=(q.score>=.5).astype(int);q['scope']=['held_high']*len(pos)+['unseen_ordinary']*len(neg);assert q.score.eq(1).all() and q.predicted.eq(1).all()
 a=q[q.scope=='held_high'];b=q[q.scope=='unseen_ordinary'];receipt=dict(seed=seed,classes=[1],train_days=len(tr)//24,train_rows=len(tr),held_high_days=len(pos)//24,held_high_rows=len(pos),high_recall=float(a.predicted.mean()),ordinary_days=len(neg)//24,ordinary_rows=len(neg),ordinary_false_positive_rate=float(b.predicted.mean()),score_min=float(q.score.min()),score_max=float(q.score.max()),model_saved=False,extra40_rescored=False)
 P.csvnew(L/f'prediction_{seed}_v1.csv',q);P.save(H/f'score_{seed}_v1.json',receipt);print(json.dumps(receipt),flush=True)

if __name__=='__main__':
 stage=sys.argv[1]
 if stage=='prepare':prepare()
 elif stage=='first':run(C.SEEDS[0])
 elif stage=='rest':
  assert (H/'critique_midpoint_v1.md').exists()
  for seed in C.SEEDS[1:]:run(seed)
 else:raise ValueError(stage)
