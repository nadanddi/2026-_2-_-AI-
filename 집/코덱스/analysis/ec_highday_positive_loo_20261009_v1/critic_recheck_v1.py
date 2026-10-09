from pathlib import Path
import csv,json,math,hashlib,sys
H=Path(__file__).resolve().parent;R=H.parents[3];OLD=R/'집/코덱스/analysis/ec_highday_classifier_20261009_v1';OLDL=R/'집/코덱스/local'/OLD.name;L=R/'집/코덱스/local'/H.name
stage=sys.argv[1];reg=json.loads((H/'registration_v1.json').read_text(encoding='utf8'));oldreg=json.loads((OLD/'registration_v1.json').read_text(encoding='utf8'));meta=list(csv.DictReader((OLDL/'metadata_public_v1.csv').open(encoding='utf8')));rowdays=[(q['farm'],int(q['day'])) for q in meta];lock=set(map(tuple,oldreg['locked_days']));pub=set(rowdays)
for path,digest in reg['source_pins'].items():assert hashlib.sha256((R/path).read_bytes()).hexdigest()==digest
assert len(reg['folds'])==26 and len({(q['farm'],q['day']) for q in reg['folds']})==26
for q in reg['folds']:
 farm,day=q['farm'],q['day'];forbidden={(farm,day+j) for j in [-1,0,1]}|{('F47' if farm=='F13' else 'F13',day+j) for j in range(-3,4)}|{(ff,dd+j) for ff,dd in lock for j in [-1,0,1]}
 assert set(map(tuple,q['forbidden_days']))==forbidden
 tr=[i for i,d in enumerate(rowdays) if d not in forbidden];va=[i for i,d in enumerate(rowdays) if d==(farm,day)];assert tr==q['train_indices'] and va==q['query_indices'] and len(va)==24
 positives={rowdays[i] for i in tr if int(meta[i]['high'])==1};assert len(positives)==q['train_high'] and len(pub-forbidden)==q['train_days'] and q['train_ordinary']==q['train_days']-q['train_high']
 assert all(int(meta[i]['high'])==1 for i in va)
data=[]
for p in sorted((L/'predictions').glob('*_v1.csv')):
 qs=list(csv.DictReader(p.open(encoding='utf8')));q=next(t for t in reg['folds'] if p.name==t['key']+'_v1.csv');assert len(qs)==24 and {t['row_id'] for t in qs}=={meta[i]['row_id'] for i in q['query_indices']}
 for t in qs:
  assert int(t['high'])==1 and abs(float(t['prior'])-q['train_high']/q['train_days'])<1e-14
  assert abs(float(t['ensemble'])-sum(float(t[f'et_{s}']) for s in [8383,1919,7171])/3)<1e-14
 data+=qs
baseline={}
for p in sorted((OLDL/'cv_predictions').glob('DIAG10*_v1.csv')):
 for q in csv.DictReader(p.open(encoding='utf8')):assert q['row_id'] not in baseline;baseline[q['row_id']]=float(q['ensemble'])
report=json.loads((H/f'{stage}_score_v1.json').read_text(encoding='utf8'))
def metric(p):
 return dict(days=len(p),caught05=sum(v>=.5 for v in p),missed05=sum(v<.5 for v in p),recall05=sum(v>=.5 for v in p)/len(p),caught02=sum(v>=.2 for v in p),mean_score=math.fsum(p)/len(p),conditional_positive_brier=math.fsum((v-1)**2 for v in p)/len(p),positive_logloss=-math.fsum(math.log(max(1e-7,min(1,v))) for v in p)/len(p))
def verify(a,b):
 for k,v in a.items():assert abs(v-b[k])<1e-12,(k,v,b[k])
for q in report['groups']:
 verify(metric([float(t[q['arm']]) for t in data if int(t['hour'])==q['hour']]),q)
qs=[t for t in data if int(t['hour'])==15];a=[float(t['ensemble']) for t in qs];b=[baseline[t['row_id']] for t in qs];verify(metric(a),report['paired_h15']['loo']);verify(metric(b),report['paired_h15']['diag']);assert sum(x>=.5 and y<.5 for x,y in zip(a,b))==report['paired_h15']['gained'];assert sum(x<.5 and y>=.5 for x,y in zip(a,b))==report['paired_h15']['lost'];assert abs(math.fsum((x-1)**2-(y-1)**2 for x,y in zip(a,b))/len(a)-report['paired_h15']['delta_positive_brier'])<1e-12
paired=list(csv.DictReader((L/f'{stage}_paired_h15_v1.csv').open(encoding='utf8')))
for t in paired:assert abs(float(t['diag_ensemble'])-baseline[t['row_id']])<1e-14
for segment in report['segments']:
 rows=[t for t in qs if t['farm']==segment['farm'] and (int(t['day'])>=179)==segment['pass2']];verify(metric([float(t['ensemble']) for t in rows]),segment);verify(metric([baseline[t['row_id']] for t in rows]),segment['diag'])
sys.path.insert(0,str(OLD));import runtime_v1
import numpy as np
blocks=sorted({(t['farm'],int(t['day'])//5) for t in qs});d=[];counts=[]
for k in blocks:
 rows=[t for t in qs if (t['farm'],int(t['day'])//5)==k];d.append(sum((float(t['ensemble'])-1)**2-(baseline[t['row_id']]-1)**2 for t in rows));counts.append(len(rows))
d=np.array(d);counts=np.array(counts);pools=[[i for i,k in enumerate(blocks) if k[0]==farm] for farm in ['F13','F47']];rng=np.random.default_rng(202610095);values=[]
for _ in range(20):
 choices=np.concatenate([rng.choice(pool,size=(1000,len(pool)),replace=True) for pool in pools],axis=1);values.extend((d[choices].sum(axis=1)/counts[choices].sum(axis=1)).tolist())
assert abs(float(np.mean(np.array(values)>=0))-report['paired_h15']['p_worse'])<1e-12
assert max(abs(np.quantile(values,[.025,.975])-report['paired_h15']['delta_brier_ci95']))<1e-12
out={'status':'INDEPENDENT_POSITIVE_LOO_SPLIT_SCORE_PASS','stage':stage,'all26_splits_rebuilt':True,'completed_days':len(qs),'completed_rows':len(data),'groups_checked':len(report['groups']),'metrics_paired_source_hash_bootstrap_verified':True,'no_negative_evaluation':True,'extra40_target_or_performance_loaded':False}
p=H/f'critic_{stage}_recheck_v1.json';assert not p.exists();p.write_text(json.dumps(out,indent=2),encoding='utf8');print(json.dumps(out))
