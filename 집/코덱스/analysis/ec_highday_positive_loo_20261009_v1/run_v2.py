"""Fixed positive-day LOO of the previously registered binary classifier."""
from pathlib import Path
import sys,json,math,time,gc,warnings,hashlib
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;ROOT=H.parents[3];OLD=ROOT/'집/코덱스/analysis/ec_highday_classifier_20261009_v1';sys.path.insert(0,str(OLD))
import runtime_v1,recipe_v1 as C,run_v1 as P
import numpy as np,pandas as pd
from threadpoolctl import threadpool_limits
from sklearn.exceptions import ConvergenceWarning
L=ROOT/'집/코덱스/local'/H.name
sha=P.sha

def prepare():
 old=P.check();meta=pd.read_csv(P.L/'metadata_public_v1.csv',float_precision='round_trip');f=pd.read_csv(P.L/'features_public_v1.csv',float_precision='round_trip');assert meta.row_id.equals(f.row_id)
 days=list(zip(meta.farm,meta.day));high=meta[meta.high==1][['farm','day']].drop_duplicates().sort_values(['farm','day']);assert len(high)==26
 lock=set(map(tuple,old['locked_days']));folds=[]
 for k,(farm,day) in enumerate(high.itertuples(index=False,name=None)):
  day=int(day);other='F47' if farm=='F13' else 'F13';forbidden={(farm,day+j) for j in (-1,0,1)}|{(other,day+j) for j in range(-3,4)}|{(ff,dd+j) for ff,dd in lock for j in (-1,0,1)}
  tr=[i for i,d in enumerate(days) if d not in forbidden];va=[i for i,d in enumerate(days) if d==(farm,day)]
  assert len(va)==24 and not set(tr)&set(va) and set(meta.high.iloc[tr])=={0,1} and meta.high.iloc[va].eq(1).all()
  train_days=meta.iloc[tr][['farm','day','high']].drop_duplicates();folds.append(dict(key=f'LOO_{k:03d}',farm=farm,day=day,train_indices=tr,query_indices=va,forbidden_days=sorted(forbidden),train_days=len(train_days),train_high=int(train_days.high.sum()),train_ordinary=int((train_days.high==0).sum()),train_sha=P.arrayhash(f[C.COLS].to_numpy(float)[tr])))
 pins={str(p.relative_to(ROOT)):sha(p) for p in [OLD/'registration_v1.json',OLD/'recipe_v1.py',OLD/'runtime_v1.py',OLD/'run_v1.py',P.L/'features_public_v1.csv',P.L/'metadata_public_v1.csv',*sorted((P.L/'cv_predictions').glob('DIAG10*_v1.csv')),H/'PLAN_v2.md',H/'run_v2.py']}
 P.save(H/'registration_v1.json',dict(source_pins=pins,folds=folds,seeds=list(C.SEEDS),threshold=C.EC_THRESHOLD,score_threshold=.5,primary_hour=15,adoption=False,official_extra40_labels_loaded=False));print('PREPARED',len(folds),'FOLDS',flush=True)

def check():
 P.check();reg=json.loads((H/'registration_v1.json').read_text(encoding='utf8'))
 for path,digest in reg['source_pins'].items():assert sha(ROOT/path)==digest,path
 return reg

def run(keys):
 reg=check();f=pd.read_csv(P.L/'features_public_v1.csv',float_precision='round_trip');meta=pd.read_csv(P.L/'metadata_public_v1.csv',float_precision='round_trip');X=f[C.COLS];y=meta.high.to_numpy(int)
 for r in reg['folds']:
  if r['key'] not in keys:continue
  out=L/'predictions'/f"{r['key']}_v1.csv";assert not out.exists();tr=r['train_indices'];va=r['query_indices'];assert P.arrayhash(X.iloc[tr].to_numpy(float))==r['train_sha'];q=meta.iloc[va].copy();q['loo_fold']=r['key'];q['train_days']=r['train_days'];q['train_high']=r['train_high'];q['train_ordinary']=r['train_ordinary'];q['prior']=float(y[tr].mean());receipts=[]
  for kind,seed in [('logit',0)]+[('et',seed) for seed in C.SEEDS]:
   t=time.monotonic();model=C.factory(kind,seed)
   with warnings.catch_warnings(record=True) as ww,threadpool_limits(limits=2):
    warnings.simplefilter('always');model.fit(X.iloc[tr],y[tr],clf__sample_weight=np.full(len(tr),1/24));p=C.positive_score(model,X.iloc[va]);pt=C.positive_score(model,X.iloc[tr])
   assert not any(issubclass(w.category,ConvergenceWarning) for w in ww)
   q['logit' if kind=='logit' else f'et_{seed}']=p;receipts.append(dict(kind=kind,seed=seed,training_brier=float(np.mean((pt-y[tr])**2)),seconds=time.monotonic()-t,train_sha=r['train_sha']));del model;gc.collect()
  q['ensemble']=q[[f'et_{seed}' for seed in C.SEEDS]].mean(axis=1);assert np.isfinite(q.select_dtypes(include='number')).all().all();P.csvnew(out,q);P.save(out.with_suffix('.json'),dict(farm=r['farm'],day=r['day'],train_days=r['train_days'],train_high=r['train_high'],train_ordinary=r['train_ordinary'],query_rows=len(va),receipts=receipts));print('DONE',r['key'],r['farm'],r['day'],'TRAIN',r['train_days'],'HIGH',r['train_high'],'SECONDS',round(sum(v['seconds'] for v in receipts),2),flush=True)

def metric(p):
 a=np.asarray(p,float);assert np.isfinite(a).all() and ((0<=a)&(a<=1)).all()
 return dict(days=len(a),caught05=int((a>=.5).sum()),missed05=int((a<.5).sum()),recall05=float((a>=.5).mean()),caught02=int((a>=.2).sum()),mean_score=float(a.mean()),conditional_positive_brier=float(((a-1)**2).mean()),positive_logloss=float(-np.log(np.clip(a,1e-7,1)).mean()))

def bootstrap(q):
 q=q.copy();q['block']=q.day//5;keys=sorted(set(zip(q.farm,q.block)));rows=[q[(q.farm==ff)&(q.block==bb)] for ff,bb in keys];d=np.array([sum((r.ensemble-1)**2-(r.diag_ensemble-1)**2) for r in rows]);n=np.array([len(r) for r in rows]);pools=[[i for i,k in enumerate(keys) if k[0]==ff] for ff in ['F13','F47']];rng=np.random.default_rng(202610095);values=[]
 for _ in range(20):
  ix=np.concatenate([rng.choice(pool,size=(1000,len(pool)),replace=True) for pool in pools],axis=1);values.extend((d[ix].sum(1)/n[ix].sum(1)).tolist())
 return dict(replicates=20000,blocks=len(keys),delta_brier_ci95=np.quantile(values,[.025,.975]).tolist(),p_worse=float((np.asarray(values)>=0).mean()),interpretation='exploratory positive-only; no general-classification adoption')

def score(stage):
 reg=check();paths=sorted((L/'predictions').glob('*_v1.csv'));g=pd.concat([pd.read_csv(p,float_precision='round_trip') for p in paths],ignore_index=True);assert (g.groupby(['farm','day']).size()==24).all() and g.high.eq(1).all();groups=[]
 for h in [0,8,15,23]:
  gh=g[g.hour==h]
  for arm in ['prior','logit']+[f'et_{seed}' for seed in C.SEEDS]+['ensemble']:
   groups.append(dict(hour=h,arm=arm,**metric(gh[arm])))
 diag=pd.concat([pd.read_csv(p,float_precision='round_trip') for p in sorted((P.L/'cv_predictions').glob('DIAG10*_v1.csv'))],ignore_index=True);diag=diag.rename(columns={'ensemble':'diag_ensemble'})
 paired=g.merge(diag[['row_id','diag_ensemble']],on='row_id',validate='one_to_one');assert len(paired)==len(g);q=paired[paired.hour==15].copy();q['loo_caught05']=q.ensemble>=.5;q['diag_caught05']=q.diag_ensemble>=.5;q['delta_positive_brier']=(q.ensemble-1)**2-(q.diag_ensemble-1)**2
 segments=[dict(farm=ff,pass2=bool(p2),**metric(v.ensemble),diag=metric(v.diag_ensemble)) for (ff,p2),v in q.assign(pass2=q.day>=179).groupby(['farm','pass2'])]
 full=len(paths)==len(reg['folds']);out=dict(status='FULL_LOO' if full else 'PARTIAL_LOO',groups=groups,segments=segments,paired_h15=dict(loo=metric(q.ensemble),diag=metric(q.diag_ensemble),gained=int((q.loo_caught05&~q.diag_caught05).sum()),lost=int((~q.loo_caught05&q.diag_caught05).sum()),delta_positive_brier=float(q.delta_positive_brier.mean()),**bootstrap(q)),train_days=dict(min=int(q.train_days.min()),max=int(q.train_days.max()),mean=float(q.train_days.mean())),train_high=dict(min=int(q.train_high.min()),max=int(q.train_high.max()),mean=float(q.train_high.mean())),adoption=False,no_negative_evaluation=True,extra40_rescored=False)
 P.save(H/f'{stage}_score_v1.json',out);P.csvnew(L/f'{stage}_paired_h15_v1.csv',q);print(json.dumps(out,ensure_ascii=True),flush=True)

if __name__=='__main__':
 stage=sys.argv[1]
 if stage=='prepare':prepare()
 elif stage=='first':run(['LOO_000','LOO_001']);score('midpoint')
 elif stage=='rest':
  assert (H/'critique_midpoint_v1.md').exists();r=check();run([f['key'] for f in r['folds'] if f['key'] not in ['LOO_000','LOO_001']]);score('final')
 else:raise ValueError(stage)
