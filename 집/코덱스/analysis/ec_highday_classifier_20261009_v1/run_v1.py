"""Registered high-EC day classifier experiment; no submissions."""
from pathlib import Path
import sys,json,hashlib,time,warnings,gc,math
sys.dont_write_bytecode=True
import recipe_v1 as C
import numpy as np,pandas as pd,joblib
from sklearn.metrics import roc_auc_score,average_precision_score,log_loss,brier_score_loss
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
H=Path(__file__).resolve().parent;ROOT=H.parents[3];L=ROOT/'집/코덱스/local'/H.name;DATA=ROOT/'공용/대회자료/정형데이터/참가자_배포'
OOFS=sorted((ROOT/'집/클로드/research/local/ct1_ckpt').glob('DIAG10_*.csv'));assert len(OOFS)==10
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):
 p.parent.mkdir(parents=True,exist_ok=True);assert not p.exists(),str(p);p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
def csvnew(p,d):
 p.parent.mkdir(parents=True,exist_ok=True);assert not p.exists(),str(p);d.to_csv(p,index=False)
def arrayhash(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def inputs():return {str(p.relative_to(ROOT)):sha(p) for p in [DATA/'train_X.csv',DATA/'train_y.csv',*OOFS]}
def codes():return {n:sha(H/n) for n in ['runtime_v1.py','recipe_v1.py','run_v1.py','predict_v1.py','final_train_v1.py']}
def prepare():
 assert not (H/'registration_v1.json').exists();pins=inputs();code=codes()
 o=pd.concat([pd.read_csv(p,usecols=['row_id','farm','day','hour','sub_ec','validation_fold'],float_precision='round_trip') for p in OOFS],ignore_index=True)
 assert len(o)==8640 and o.row_id.is_unique
 dm=o.groupby(['farm','day']).sub_ec.agg(lambda y:math.fsum(y)/len(y));assert len(dm)==360
 x=pd.read_csv(DATA/'train_X.csv');fullmeta=C.identify(x[x.row_id.str[:3].isin(['F13','F47'])]);all_days=set(fullmeta[['farm','day']].itertuples(index=False,name=None));pubdays=set(dm.index);lock=all_days-pubdays;assert len(all_days)==400 and len(lock)==40
 x=x[x.row_id.isin(o.row_id)].reset_index(drop=True);f=C.features(x[['row_id']+C.RAW]);meta=f[['row_id','farm','day','hour']].copy();meta['daily_ec']=[float(dm[(ff,dd)]) for ff,dd in zip(meta.farm,meta.day)];meta['high']=(meta.daily_ec>=C.EC_THRESHOLD).astype(int)
 assert len(meta)==8640 and (meta.groupby(['farm','day']).size()==24).all()
 design=[]
 for k,g in o.groupby('validation_fold'):design.append(('DIAG10',int(k),set(g[['farm','day']].itertuples(index=False,name=None))))
 d2={ff:sorted(dd for f0,dd in pubdays if f0==ff and dd>=179) for ff in ['F13','F47']}
 for k in range(7):
  q=set()
  for ff in ['F13','F47']:
   for j in range(0,len(d2[ff]),3):
    if (j//3+(0 if ff=='F13' else 3))%7==k:q.update((ff,dd) for dd in d2[ff][j:j+3])
  if q:design.append(('FRESH7',k,q))
 k=0
 for ff in ['F13','F47']:
  for j in range(0,len(d2[ff]),5):design.append(('EL1',k,{(ff,dd) for dd in d2[ff][j:j+5]}));k+=1
 rowdays=list(zip(meta.farm,meta.day));folds=[]
 for name,k,q in design:
  forb={(ff,dd+j) for ff,dd in q for j in (-1,0,1)}|{('F47' if ff=='F13' else 'F13',dd+j) for ff,dd in q for j in range(-3,4)}|{(ff,dd+j) for ff,dd in lock for j in (-1,0,1)}
  tr=[i for i,d in enumerate(rowdays) if d not in forb];va=[i for i,d in enumerate(rowdays) if d in q];assert tr and va and not set(tr)&set(va)
  yy=meta.high.to_numpy()[tr];assert set(yy)=={0,1},(name,k)
  folds.append(dict(name=name,index=k,key=f'{name}_{k:03d}',train_indices=tr,query_indices=va,train_days=sorted(set(rowdays[i] for i in tr)),query_days=sorted(q),forbidden_days=sorted(forb),train_classes_days={str(c):len({rowdays[i] for i in tr if meta.high.iloc[i]==c}) for c in (0,1)},feature_train_sha=arrayhash(f[C.COLS].to_numpy(float)[tr])))
 for name in ['DIAG10','FRESH7','EL1']:
  qs=[d for rec in folds if rec['name']==name for d in map(tuple,rec['query_days'])];assert len(qs)==len(set(qs)) and len(qs)==(360 if name=='DIAG10' else 46)
 L.mkdir(parents=True,exist_ok=True);csvnew(L/'raw_public_v1.csv',x);csvnew(L/'metadata_public_v1.csv',meta);csvnew(L/'features_public_v1.csv',f)
 reg=dict(inputs=pins,code_hashes=code,plan_sha=sha(H/'PLAN_v3.md'),supplement_sha=sha(H/'plan_supplement_v1.md'),threshold=C.EC_THRESHOLD,seeds=list(C.SEEDS),feature_columns=C.COLS,folds=folds,dataset_hashes={p.name:sha(p) for p in L.glob('*.csv')},locked_days=sorted(lock),primary_hour=15,adoption=False,versions=dict(python=sys.version,numpy=np.__version__,pandas=pd.__version__,sklearn=__import__('sklearn').__version__))
 save(H/'registration_v1.json',reg)
 profile=dict(days=360,rows=8640,high_days=int(meta[['farm','day','high']].drop_duplicates().high.sum()),ordinary_days=360-int(meta[['farm','day','high']].drop_duplicates().high.sum()),pass2=meta[meta.day>=179][['farm','day','high']].drop_duplicates().groupby('high').size().to_dict(),features=len(C.COLS),missing=f[C.COLS].isna().sum().to_dict(),fold_count=len(folds),official_extra40_labels_loaded=False)
 save(H/'profile_v1.json',profile);print(json.dumps(profile,ensure_ascii=True),flush=True)
def check():
 reg=json.loads((H/'registration_v1.json').read_text(encoding='utf8'));assert reg['inputs']==inputs() and reg['code_hashes']==codes() and reg['plan_sha']==sha(H/'PLAN_v3.md') and reg['supplement_sha']==sha(H/'plan_supplement_v1.md')
 for n,d in reg['dataset_hashes'].items():assert sha(L/n)==d
 return reg

def metrics(y,p,threshold=.5):
 y=np.asarray(y,int);p=np.asarray(p,float);assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all();a=p>=threshold
 tp=int((a&(y==1)).sum());fp=int((a&(y==0)).sum());tn=int((~a&(y==0)).sum());fn=int((~a&(y==1)).sum());pos=int(y.sum());n=len(y)
 return dict(days=n,high_days=pos,ordinary_days=n-pos,roc_auc=float(roc_auc_score(y,p)) if 0<pos<n else None,average_precision=float(average_precision_score(y,p)) if pos>0 else None,brier=float(brier_score_loss(y,p)),logloss=float(log_loss(y,np.clip(p,1e-7,1-1e-7),labels=[0,1])),precision=tp/(tp+fp) if tp+fp else 0.,recall=tp/pos if pos else None,f2=5*tp/(5*tp+4*fn+fp) if 5*tp+4*fn+fp else 0.,tp=tp,fp=fp,tn=tn,fn=fn,threshold=threshold,accuracy=(tp+tn)/n)

def run(keys):
 reg=check();f=pd.read_csv(L/'features_public_v1.csv',float_precision='round_trip');meta=pd.read_csv(L/'metadata_public_v1.csv',float_precision='round_trip');X=f[C.COLS];y=meta.high.to_numpy(int)
 for rec in reg['folds']:
  key=rec['key']
  if key not in keys:continue
  out=L/'cv_predictions'/f'{key}_v1.csv';assert not out.exists()
  tr,va=rec['train_indices'],rec['query_indices'];Xt,Xv=X.iloc[tr],X.iloc[va];yt=y[tr];assert arrayhash(Xt.to_numpy(float))==rec['feature_train_sha'] and set(yt)=={0,1}
  g=meta.iloc[va].copy().reset_index(drop=True);g['validator']=rec['name'];g['fold']=rec['index'];g['prior']=float(yt.mean());receipts=[]
  for kind,seed in [('logit',0)]+[('et',s) for s in C.SEEDS]:
   t=time.monotonic();model=C.factory(kind,seed)
   with warnings.catch_warnings(record=True) as ww,threadpool_limits(limits=2):
    warnings.simplefilter('always');model.fit(Xt,yt,clf__sample_weight=np.full(len(tr),1/24));p=C.positive_score(model,Xv);pt=C.positive_score(model,Xt)
   assert not any(issubclass(w.category,ConvergenceWarning) for w in ww)
   col='logit' if kind=='logit' else f'et_{seed}';g[col]=p
   receipts.append(dict(kind=kind,seed=seed,train_rows=len(tr),train_days=len(tr)//24,train_classes_days=rec['train_classes_days'],feature_sha=rec['feature_train_sha'],training_brier=float(np.mean((pt-yt)**2)),warnings=[str(w.message) for w in ww],seconds=time.monotonic()-t));del model;gc.collect()
  g['ensemble']=g[[f'et_{s}' for s in C.SEEDS]].mean(axis=1);assert g.notna().all().all();csvnew(out,g);save(L/'cv_predictions'/f'{key}_v1.json',dict(receipts=receipts,query_rows=len(va),query_ids=g.row_id.tolist(),train_query_overlap=0));print('DONE',key,'query_days',len(va)//24,'train_days',len(tr)//24,'seconds',round(sum(x['seconds'] for x in receipts),2),flush=True)

def bootstrap(g,arm='prior'):
 gg=g.copy();gg['block']=gg.day//5;keys=list(gg.groupby(['farm','block']).groups);d=gg.groupby(['farm','block']).apply(lambda z:float(((z.ensemble-z.high)**2-(z[arm]-z.high)**2).sum()),include_groups=False).reindex(keys).to_numpy();n=gg.groupby(['farm','block']).size().reindex(keys).to_numpy();idx={ff:[i for i,k in enumerate(keys) if k[0]==ff] for ff in ['F13','F47']};rng=np.random.default_rng(202610095);delta=[]
 for _ in range(20):
  ii=np.concatenate([rng.choice(ids,size=(1000,len(ids)),replace=True) for ids in idx.values()],axis=1);delta.extend((d[ii].sum(1)/n[ii].sum(1)).tolist())
 return dict(blocks=len(keys),replicates=20000,delta_brier_ci95=np.quantile(delta,[.025,.975]).tolist(),p_worse=float(np.mean(np.array(delta)>=0)))

def score(label):
 reg=check();paths=sorted((L/'cv_predictions').glob('*_v1.csv'));g=pd.concat([pd.read_csv(p,float_precision='round_trip') for p in paths],ignore_index=True);groups=[]
 for v in g.validator.unique():
  for h in [0,8,15,23]:
   gh=g[(g.validator==v)&(g.hour==h)]
   for scope,m in [('all',np.ones(len(gh),bool)),('pass2',gh.day.to_numpy()>=179)]:
    q=gh[m]
    if q.empty:continue
    for arm in ['prior','logit']+[f'et_{s}' for s in C.SEEDS]+['ensemble']:
     z=dict(validator=v,hour=h,scope=scope,arm=arm,**metrics(q.high,q[arm]));z['threshold02']=metrics(q.high,q[arm],.2);groups.append(z)
 boots=[];q=g[(g.validator=='DIAG10')&(g.hour==15)].copy()
 for arm in ['prior','logit']:boots.append(dict(validator='DIAG10',scope='all',hour=15,comparator=arm,**bootstrap(q,arm)))
 reliability=[]
 q=g[(g.hour==15)].copy();q['bin']=pd.cut(q.ensemble,bins=[0,.1,.25,.5,.75,1],include_lowest=True)
 for (v,b),z in q.groupby(['validator','bin'],observed=True):reliability.append(dict(validator=v,bin=str(b),days=len(z),mean_score=float(z.ensemble.mean()),high_rate=float(z.high.mean())))
 completed=g[['validator','fold']].drop_duplicates();full=len(completed)==len(reg['folds']);lookup={(z['validator'],z['hour'],z['scope'],z['arm']):z for z in groups};useful=False
 if full:
  cells=[lookup[(v,15,'all',f'et_{s}')]['brier']<lookup[(v,15,'all','prior')]['brier'] for v in ['DIAG10','FRESH7','EL1'] for s in C.SEEDS]
  b=boots[0];useful=all(cells) and b['p_worse']<.025 and b['delta_brier_ci95'][1]<0
 save(H/f'{label}_score_v1.json',dict(status='FULL_CV' if full else 'PARTIAL_CV',groups=groups,bootstrap=boots,reliability=reliability,completed_folds=completed.to_dict('records'),prior_usefulness_supported=useful,adoption=False,extra40_rescored=False,threshold=C.EC_THRESHOLD,primary_hour=15))
 if full:
  errors=g[g.hour==15].copy();errors['hard_prediction']=(errors.ensemble>=.5).astype(int);csvnew(L/'classification_oof_h15_v1.csv',errors);csvnew(L/'misclassified_h15_v1.csv',errors[errors.hard_prediction!=errors.high])
 print(json.dumps({'status':'FULL_CV' if full else 'PARTIAL_CV','prior_usefulness_supported':useful,'metrics':[z for z in groups if z['arm']=='ensemble' and z['scope']=='all']},ensure_ascii=True),flush=True)

if __name__=='__main__':
 stage=sys.argv[1]
 if stage=='prepare':prepare()
 elif stage=='first':run(['DIAG10_000','DIAG10_001']);score('midpoint')
 elif stage=='rest':
  reg=check();run([r['key'] for r in reg['folds'] if r['key'] not in ['DIAG10_000','DIAG10_001']]);score('final')
 elif stage=='finalfit':__import__('final_train_v1').final_train()
 else:raise ValueError(stage)
