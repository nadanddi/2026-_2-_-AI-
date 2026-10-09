"""High-only OneClassSVM fit; retain both classes in registered CV validation."""
from pathlib import Path
import sys,json,time,warnings,math,gc
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;ROOT=H.parents[3];sys.path.insert(0,str(ROOT/'집/클로드/research'));import env
OLD=ROOT/'집/코덱스/analysis/ec_highday_classifier_20261009_v1';sys.path.insert(0,str(OLD));import runtime_v1,recipe_v1 as C,run_v1 as P
import numpy as np,pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
L=ROOT/'집/코덱스/local'/H.name
PARAM=dict(kernel='rbf',nu=.1,gamma='scale',tol=.001,max_iter=-1)

def prepare():
 old=P.check();meta=pd.read_csv(P.L/'metadata_public_v1.csv');features=pd.read_csv(P.L/'features_public_v1.csv',float_precision='round_trip');assert meta.row_id.equals(features.row_id);folds=[]
 for r in old['folds']:
  tr=[i for i in r['train_indices'] if int(meta.high.iloc[i])==1];va=r['query_indices'];assert tr and set(meta.high.iloc[tr])=={1} and not set(tr)&set(va) and len(tr)%24==len(va)%24==0
  baseline=P.L/'cv_predictions'/f"{r['key']}_v1.csv";b=pd.read_csv(baseline,float_precision='round_trip');assert set(b.row_id)==set(meta.iloc[va].row_id)
  folds.append(dict(key=r['key'],name=r['name'],index=r['index'],train_indices=tr,query_indices=va,train_high_days=len(tr)//24,query_days=len(va)//24,train_feature_sha=P.arrayhash(features[C.COLS].to_numpy(float)[tr]),baseline=str(baseline.relative_to(ROOT))))
 files=[H/'PLAN_v2.md',H/'run_v2.py',OLD/'recipe_v1.py',OLD/'runtime_v1.py',OLD/'run_v1.py',OLD/'registration_v1.json',P.L/'features_public_v1.csv',P.L/'metadata_public_v1.csv']+[ROOT/r['baseline'] for r in folds]
 P.save(H/'registration_v1.json',dict(source_pins={str(p.relative_to(ROOT)):P.sha(p) for p in files},folds=folds,parameters=PARAM,feature_columns=C.COLS,deterministic=True,adoption=False,extra40_rescored=False));print('PREPARED',len(folds),'FOLDS',flush=True)

def check():
 P.check();r=json.loads((H/'registration_v1.json').read_text(encoding='utf8'))
 for name,digest in r['source_pins'].items():assert P.sha(ROOT/name)==digest,name
 assert r['parameters']==PARAM;return r

def run(keys):
 r=check();meta=pd.read_csv(P.L/'metadata_public_v1.csv');features=pd.read_csv(P.L/'features_public_v1.csv',float_precision='round_trip');X=features[C.COLS]
 for f in r['folds']:
  if f['key'] not in keys:continue
  out=L/'predictions'/f"{f['key']}_v1.csv";assert not out.exists();tr=f['train_indices'];va=f['query_indices'];assert set(meta.high.iloc[tr])=={1} and P.arrayhash(X.iloc[tr].to_numpy(float))==f['train_feature_sha']
  model=Pipeline([('imputer',SimpleImputer(strategy='median',keep_empty_features=True)),('scaler',StandardScaler()),('svm',OneClassSVM(**PARAM))]);start=time.monotonic()
  with warnings.catch_warnings(record=True) as ww,threadpool_limits(limits=2):
   warnings.simplefilter('always');model.fit(X.iloc[tr]);pred=model.predict(X.iloc[va]);score=model.decision_function(X.iloc[va]);train_pred=model.predict(X.iloc[tr])
  assert not any(issubclass(w.category,ConvergenceWarning) for w in ww) and model.named_steps['svm'].fit_status_==0 and np.isfinite(score).all() and set(pred).issubset({-1,1});assert np.array_equal(pred==1,score>0)
  q=meta.iloc[va].copy().reset_index(drop=True);q['validator']=f['name'];q['fold']=f['index'];q['oneclass_score']=score;q['oneclass_high']=(pred==1).astype(int)
  b=pd.read_csv(ROOT/f['baseline'],usecols=['row_id','ensemble'],float_precision='round_trip').rename(columns={'ensemble':'et_score'});q=q.merge(b,on='row_id',validate='one_to_one');q['et_high']=(q.et_score>=.5).astype(int);assert q.notna().all().all()
  receipt=dict(key=f['key'],train_rows=len(tr),train_high_days=len(tr)//24,train_ordinary_days=0,query_rows=len(va),train_query_overlap=0,train_feature_sha=f['train_feature_sha'],training_inlier_fraction_all24h=float((train_pred==1).mean()),support_vectors=int(len(model.named_steps['svm'].support_)),fit_status=int(model.named_steps['svm'].fit_status_),warnings=[str(w.message) for w in ww],seconds=time.monotonic()-start,imputer_fit_rows=len(tr),scaler_fit_rows=len(tr),deterministic=True)
  P.csvnew(out,q);P.save(out.with_suffix('.json'),receipt);print('DONE',f['key'],'TRAIN_HIGH',len(tr)//24,'QUERY',len(va)//24,'SECONDS',round(receipt['seconds'],2),flush=True);del model;gc.collect()

def metrics(y,p):
 y=np.asarray(y,int);p=np.asarray(p,int);assert set(y).issubset({0,1}) and set(p).issubset({0,1});tp=int(((y==1)&(p==1)).sum());fn=int(((y==1)&(p==0)).sum());fp=int(((y==0)&(p==1)).sum());tn=int(((y==0)&(p==0)).sum());pos=tp+fn;neg=fp+tn
 return dict(days=len(y),high_days=pos,ordinary_days=neg,tp=tp,fn=fn,fp=fp,tn=tn,recall=tp/pos if pos else None,ordinary_false_positive_rate=fp/neg if neg else None,precision=tp/(tp+fp) if tp+fp else 0.,balanced_accuracy=.5*(tp/pos+tn/neg) if pos and neg else None)

def score(stage):
 r=check();g=pd.concat([pd.read_csv(p,float_precision='round_trip') for p in sorted((L/'predictions').glob('*_v1.csv'))],ignore_index=True);assert g[['validator','row_id']].duplicated().sum()==0 and (g.groupby(['validator','farm','day']).size()==24).all();groups=[];segments=[]
 for v,z in g.groupby('validator'):
  for h in [0,8,15,23]:
   gh=z[z.hour==h]
   for scope,q in [('all',gh),('pass2',gh[gh.day>=179])]:
    if q.empty:continue
    for arm,p in [('oneclass',q.oneclass_high),('et',q.et_high),('always_high',np.ones(len(q),int)),('always_normal',np.zeros(len(q),int))]:groups.append(dict(validator=v,hour=h,scope=scope,arm=arm,**metrics(q.high,p)))
  for (farm,p2),q in z[z.hour==15].assign(pass2=z[z.hour==15].day>=179).groupby(['farm','pass2']):segments.append(dict(validator=v,farm=farm,pass2=bool(p2),**metrics(q.high,q.oneclass_high)))
 keys=g[['validator','fold']].drop_duplicates();full=len(keys)==len(r['folds']);out=dict(status='FULL_CV' if full else 'PARTIAL_CV',groups=groups,segments=segments,completed_folds=keys.to_dict('records'),parameters=PARAM,deterministic=True,probability_calibrated=False,adoption=False,extra40_rescored=False)
 P.save(H/f'{stage}_score_v1.json',out);P.csvnew(L/f'{stage}_h15_v1.csv',g[g.hour==15]);print(json.dumps(dict(status=out['status'],metrics=[m for m in groups if m['hour']==15 and m['scope']=='all']),ensure_ascii=True),flush=True)

if __name__=='__main__':
 stage=sys.argv[1]
 if stage=='prepare':prepare()
 elif stage=='first':run(['DIAG10_000','DIAG10_001']);score('midpoint')
 elif stage=='rest':
  assert (H/'critique_midpoint_v1.md').exists();r=check();run([f['key'] for f in r['folds'] if f['key'] not in ['DIAG10_000','DIAG10_001']]);score('final')
 else:raise ValueError(stage)
