from pathlib import Path
import sys,csv,json,math,os
H=Path(__file__).resolve().parent;R=H.parents[3];L=R/'집/코덱스/local'/H.name
stage=sys.argv[1];report=json.loads((H/f'{stage}_score_v1.json').read_text(encoding='utf8'));reg=json.loads((H/'registration_v1.json').read_text(encoding='utf8'));allrows=[]
for file in sorted((L/'cv_predictions').glob('*_v1.csv')):
 rows=list(csv.DictReader(file.open(encoding='utf8')));key=file.name.removesuffix('_v1.csv');rec=next(f for f in reg['folds'] if f['key']==key);assert len(rows)==len(rec['query_indices'])
 ids={q['row_id'] for q in rows};expected=set(json.loads(file.with_suffix('.json').read_text(encoding='utf8'))['query_ids']);assert ids==expected
 prior=rec['train_classes_days']['1']/sum(rec['train_classes_days'].values())
 for q in rows:
  assert abs(float(q['prior'])-prior)<1e-14
  assert abs(sum(float(q[f'et_{s}']) for s in [8383,1919,7171])/3-float(q['ensemble']))<1e-14
 allrows.extend(rows)
def metrics(y,p,threshold):
 tp=sum(t==1 and v>=threshold for t,v in zip(y,p));fp=sum(t==0 and v>=threshold for t,v in zip(y,p));tn=sum(t==0 and v<threshold for t,v in zip(y,p));fn=sum(t==1 and v<threshold for t,v in zip(y,p));pos=sum(y);neg=len(y)-pos
 auc=sum(1 if a>b else .5 if a==b else 0 for a,t in zip(p,y) if t for b,u in zip(p,y) if not u)/(pos*neg) if pos and neg else None
 ap=None
 if pos:
  grouped={}
  for t,v in zip(y,p):grouped.setdefault(v,[]).append(t)
  seen=hits=0;ap=0.
  for v in sorted(grouped,reverse=True):
   ys=grouped[v];new=sum(ys);hits+=new;seen+=len(ys);ap+=(new/pos)*(hits/seen)
 brier=sum((v-t)**2 for t,v in zip(y,p))/len(y);loss=-sum(t*math.log(max(1e-7,min(1-1e-7,v)))+(1-t)*math.log(1-max(1e-7,min(1-1e-7,v))) for t,v in zip(y,p))/len(y)
 return dict(roc_auc=auc,average_precision=ap,brier=brier,logloss=loss,precision=tp/(tp+fp) if tp+fp else 0.,recall=tp/pos if pos else None,f2=5*tp/(5*tp+4*fn+fp) if 5*tp+4*fn+fp else 0.,tp=tp,fp=fp,tn=tn,fn=fn,days=len(y),high_days=pos,ordinary_days=neg,accuracy=(tp+tn)/len(y))
for item in report['groups']:
 rr=[q for q in allrows if q['validator']==item['validator'] and int(q['hour'])==item['hour'] and (item['scope']=='all' or int(q['day'])>=179)];y=[int(q['high']) for q in rr];p=[float(q[item['arm']]) for q in rr]
 for threshold,result in [(.5,item),(.2,item['threshold02'])]:
  ours=metrics(y,p,threshold)
  for k,a in ours.items():
   b=result[k];assert (a is None)==(b is None),(k,a,b)
   if a is not None:assert abs(a-b)<1e-12,(k,a,b)
# Reproduce stratified bootstrap with independently assembled squared-error sums.
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[name]='1'
sys.path.insert(0,str(R/'.analysis-tools/python'));handles=[]
if hasattr(os,'add_dll_directory'):
 for path in (R/'.analysis-tools/python').glob('**/.libs'):handles.append(os.add_dll_directory(str(path)))
import numpy as np
q=[t for t in allrows if t['validator']=='DIAG10' and int(t['hour'])==15];groups={}
for t in q:groups.setdefault((t['farm'],int(t['day'])//5),[]).append(t)
keys=sorted(groups)
for item in report['bootstrap']:
 arm=item['comparator'];diff=np.array([sum((float(t['ensemble'])-int(t['high']))**2-(float(t[arm])-int(t['high']))**2 for t in groups[k]) for k in keys]);counts=np.array([len(groups[k]) for k in keys]);rng=np.random.default_rng(202610095);d=[];pools=[[i for i,k in enumerate(keys) if k[0]==farm] for farm in ['F13','F47']]
 for _ in range(20):
  chosen=np.concatenate([rng.choice(pool,size=(1000,len(pool)),replace=True) for pool in pools],axis=1);d.extend((diff[chosen].sum(axis=1)/counts[chosen].sum(axis=1)).tolist())
 assert abs(np.mean(np.array(d)>=0)-item['p_worse'])<1e-14
 assert np.max(np.abs(np.quantile(d,[.025,.975])-item['delta_brier_ci95']))<1e-12
out={'status':'INDEPENDENT_CLASSIFICATION_SCORE_PASS','stage':stage,'prediction_rows':len(allrows),'metric_groups':len(report['groups']),'both_thresholds_rechecked':True,'AUC_AP_Brier_logloss_confusions_bootstrap_rechecked':True}
p=H/f'critic_{stage}_score_recheck_v1.json';assert not p.exists();p.write_text(json.dumps(out,indent=2),encoding='utf8');print(json.dumps(out))
