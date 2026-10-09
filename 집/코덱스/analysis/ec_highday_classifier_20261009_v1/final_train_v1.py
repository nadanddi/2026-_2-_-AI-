"""Train full 400 days after fixed public CV, without rescoring the consumed40."""
from pathlib import Path
import sys,json,time,gc
sys.dont_write_bytecode=True
import run_v1 as P
import recipe_v1 as C
from predict_v1 import predict
import numpy as np,pandas as pd,joblib
from threadpoolctl import threadpool_limits

def final_train():
 reg=P.check();s=json.loads((P.H/'final_score_v1.json').read_text(encoding='utf8'));assert s['status']=='FULL_CV' and len(s['completed_folds'])==len(reg['folds']);assert (P.H/'critique_midpoint_v1.md').exists()
 dest=P.L/'model_full_v1';assert not dest.exists();dest.mkdir(parents=True)
 # Extra40 targets first loaded only after every fixed CV fold/score is saved.
 x=pd.read_csv(P.DATA/'train_X.csv');x=x[x.row_id.str[:3].isin(['F13','F47'])].reset_index(drop=True)
 y=pd.read_csv(P.DATA/'train_y.csv',usecols=['row_id','sub_ec'],float_precision='round_trip');y=y[y.row_id.isin(x.row_id)].reset_index(drop=True);assert len(x)==len(y)==9600 and x.row_id.is_unique and y.row_id.is_unique and set(x.row_id)==set(y.row_id)
 f=C.features(x[['row_id']+C.RAW]);ym=y.copy();ids=C.identify(ym);ym['farm']=ids.farm;ym['day']=ids.day
 import math
 dm=ym.groupby(['farm','day']).sub_ec.agg(lambda v:math.fsum(v)/len(v));assert len(dm)==400
 labels=np.array([float(dm[(ff,dd)])>=C.EC_THRESHOLD for ff,dd in zip(f.farm,f.day)],int)
 assert set(labels)=={0,1};X=f[C.COLS]
 public=pd.read_csv(P.L/'raw_public_v1.csv');meta=C.identify(public);days=meta[['farm','day']].drop_duplicates();days['pass2']=days.day>=179;probe_keys=set(days.groupby(['farm','pass2']).head(2)[['farm','day']].itertuples(index=False,name=None));probe=public[[(ff,dd) in probe_keys for ff,dd in zip(meta.farm,meta.day)]].reset_index(drop=True);pf=C.features(probe[['row_id']+C.RAW]);assert len(probe)==192
 models=[];receipts=[];direct=[]
 for seed in C.SEEDS:
  start=time.monotonic();model=C.factory('et',seed)
  with threadpool_limits(limits=2):model.fit(X,labels,clf__sample_weight=np.full(len(labels),1/24))
  model.named_steps['clf'].n_jobs=1;p=C.positive_score(model,pf[C.COLS]);direct.append(p)
  path=dest/f'et_{seed}_v1.joblib';joblib.dump(model,path,compress=3);del model;gc.collect();loaded=joblib.load(path);pp=C.positive_score(loaded,pf[C.COLS]);gap=float(np.max(abs(p-pp)));assert gap<1e-12;del loaded
  models.append(dict(seed=seed,file=path.name,sha=P.sha(path)));receipts.append(dict(seed=seed,train_rows=9600,train_days=400,high_days=int(sum(v>=C.EC_THRESHOLD for v in dm)),feature_sha=P.arrayhash(X.to_numpy(float)),reload_maxdiff=gap,seconds=time.monotonic()-start));print('FINAL_FIT',seed,'seconds',round(receipts[-1]['seconds'],2),'reloadgap',gap,flush=True)
 manifest=dict(status='TRAINED_EXPERIMENTAL_CLASSIFIER',target='daily mean sub_ec >= 1.2; current prefix inputs',ec_threshold=C.EC_THRESHOLD,score_threshold=.5,score_name='uncalibrated high_ec_score',feature_columns=C.COLS,seeds=list(C.SEEDS),models=models,training_rows=9600,training_days=400,high_days=int(sum(v>=C.EC_THRESHOLD for v in dm)),ordinary_days=int(sum(v<C.EC_THRESHOLD for v in dm)),training_days_list=sorted((str(ff),int(dd)) for ff,dd in dm.index),code_hashes=P.codes(),source_sha=reg['inputs'],public_cv_score_sha=P.sha(P.H/'final_score_v1.json'),extra40_train_only=True,extra40_rescored=False,adoption=False,submission_artifact=False,versions=reg['versions'])
 P.save(dest/'model_manifest_v1.json',manifest);P.save(dest/'fit_receipts_v1.json',receipts)
 predicted=predict(dest,probe);mean=np.mean(direct,axis=0);gap=float(np.max(abs(predicted.high_ec_score.to_numpy()-mean)));assert gap<1e-12
 P.csvnew(dest/'probe_input_v1.csv',probe);P.csvnew(dest/'probe_prediction_v1.csv',predicted);P.save(P.H/'final_model_reload_v1.json',dict(status='PASS',probe_rows=192,direct_vs_reload_maxdiff=gap,per_seed=receipts))
 print('FINAL_MODEL_CREATED',str(dest),flush=True)
