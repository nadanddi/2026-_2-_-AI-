"""Audit saved high-only EC regressors, without fitting or reading extra40 targets."""
from pathlib import Path
import sys,json,math,hashlib
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;ROOT=H.parents[3];sys.path.insert(0,str(ROOT/'집/클로드/research'));import env
sys.path.insert(0,str(ROOT/'집/코덱스/analysis/ec_highday_classifier_20261009_v1'));import runtime_v1,run_v1 as P
import numpy as np,pandas as pd
SEEDS=(3737,5858,7979);CK=ROOT/'집/클로드/research/local/mx1_ckpt';L=ROOT/'집/코덱스/local'/H.name

def sources():
 return [*sorted(CK.glob('DIAG10_*.csv')),*P.OOFS,ROOT/'집/클로드/research/mx1_normal_high_mixture_v1.py',ROOT/'집/클로드/research/ec3_WT0_r3_member_weights_v1.py',P.H/'registration_v1.json',P.L/'metadata_public_v1.csv',H/'PLAN_v2.md',H/'run_v1.py']

def data():
 reg=P.check();meta=pd.read_csv(P.L/'metadata_public_v1.csv',float_precision='round_trip');ref=pd.concat([pd.read_csv(p,usecols=['row_id','farm','day','hour','sub_ec','validation_fold'],float_precision='round_trip') for p in P.OOFS],ignore_index=True);assert len(ref)==8640 and ref.row_id.is_unique
 frames=[pd.read_csv(p,float_precision='round_trip') for p in sorted(CK.glob('DIAG10_*.csv'))];assert len(frames)==10;g=pd.concat(frames,ignore_index=True);assert len(g)==8640 and g.row_id.is_unique and set(g.row_id)==set(ref.row_id)
 joined=g[['row_id','farm','day','hour','sub_ec','validation_fold']].merge(ref,on='row_id',suffixes=('_mx','_ref'),validate='one_to_one');assert all(joined[c+'_mx'].equals(joined[c+'_ref']) for c in ['farm','day','hour','sub_ec','validation_fold'])
 label=meta[['row_id','daily_ec','high']];g=g.merge(label,on='row_id',validate='one_to_one');assert (g.groupby(['farm','day']).size()==24).all()
 ordered=meta[['row_id','farm','day','high']].merge(ref[['row_id','sub_ec']],on='row_id',validate='one_to_one');assert ordered.row_id.equals(meta.row_id)
 fold_summary=[]
 for r in reg['folds']:
  if r['name']!='DIAG10':continue
  q=g[g.validation_fold==r['index']];assert set(q.row_id)==set(meta.iloc[r['query_indices']].row_id)
  tr=ordered.iloc[r['train_indices']];ht=tr[tr.high==1];n=len(ht)//24;assert q.n_high_train_days.eq(n).all() and n>=3
  constant=math.fsum(ht.sub_ec)/len(ht);g.loc[g.validation_fold==r['index'],'high_constant']=constant
  fold_summary.append(dict(fold=r['index'],train_high_days=n,constant=constant,query_days=len(q)//24))
 for arm in ['BASE','H']:g[arm+'_mean']=g[[f'{arm}_{seed}' for seed in SEEDS]].mean(axis=1)
 assert g.notna().all().all();return g,fold_summary

def metrics(g,arm):
 y=g.sub_ec.to_numpy(float);p=g[arm].to_numpy(float);e=p-y;day=g.assign(pred=p).groupby(['farm','day']).agg(actual=('sub_ec','mean'),pred=('pred','mean'));de=day.pred-day.actual
 return dict(rows=len(g),days=len(day),rmse=float(np.sqrt(np.mean(e**2))),mae=float(np.mean(abs(e))),bias=float(np.mean(e)),actual_mean=float(np.mean(y)),prediction_mean=float(np.mean(p)),actual_std=float(np.std(y)),prediction_std=float(np.std(p)),daily_mean_rmse=float(np.sqrt(np.mean(de**2))),daily_mean_mae=float(np.mean(abs(de))),daily_mean_within01=int((abs(de)<=.1).sum()),daily_mean_within02=int((abs(de)<=.2).sum()),daily_mean_within03=int((abs(de)<=.3).sum()))

def prepare():
 P.check();pins={str(p.relative_to(ROOT)):P.sha(p) for p in sources()};P.save(H/'registration_v1.json',dict(source_pins=pins,seeds=list(SEEDS),high_threshold=1.2,model_fits=0,scope='saved MX1 H conditional numeric EC forecast',official_extra40_labels_loaded=False));print('REGISTERED',len(pins),'SOURCE_PINS',flush=True)

def run(stage):
 r=json.loads((H/'registration_v1.json').read_text(encoding='utf8'));assert r['source_pins']=={str(p.relative_to(ROOT)):P.sha(p) for p in sources()};g,folds=data()
 if stage=='midpoint':g=g[g.validation_fold.isin([0,1])]
 elif stage=='final':assert (H/'critique_midpoint_v1.md').exists()
 else:raise ValueError(stage)
 q=g[g.high==1].copy();assert len(q)==24*len(q[['farm','day']].drop_duplicates());groups=[]
 for scope,gg in [('all_high',q),('pass1_high',q[q.day<179]),('pass2_high',q[q.day>=179])]:
  if gg.empty:continue
  for arm in ['BASE_mean','H_mean','high_constant']+[f'{a}_{s}' for a in ['BASE','H'] for s in SEEDS]:groups.append(dict(scope=scope,arm=arm,**metrics(gg,arm)))
 segments=[]
 for (farm,p2),v in q.assign(pass2=q.day>=179).groupby(['farm','pass2']):
  for arm in ['BASE_mean','H_mean','high_constant']:segments.append(dict(farm=farm,pass2=bool(p2),arm=arm,**metrics(v,arm)))
 daily=q.groupby(['farm','day']).agg(actual=('sub_ec','mean'),base=('BASE_mean','mean'),high_pred=('H_mean','mean'),high_constant=('high_constant','mean')).reset_index();daily['base_error']=daily.base-daily.actual;daily['high_error']=daily.high_pred-daily.actual
 out=dict(status=stage.upper(),groups=groups,segments=segments,source_checks=dict(public_rows=8640,same_row_ids=True,same_folds=True,targets_exact_equal=True,train_high_counts_equal=True),folds=folds,model_fits=0,extra40_rescored=False,high_threshold=1.2,shared_season_and_postprocessing=True,loo=False)
 P.save(H/f'{stage}_score_v1.json',out);P.csvnew(L/f'{stage}_high_rows_v1.csv',q);P.csvnew(L/f'{stage}_high_daily_v1.csv',daily);print(json.dumps(out,ensure_ascii=True),flush=True)

if __name__=='__main__':
 stage=sys.argv[1]
 if stage=='prepare':prepare()
 else:run(stage)
