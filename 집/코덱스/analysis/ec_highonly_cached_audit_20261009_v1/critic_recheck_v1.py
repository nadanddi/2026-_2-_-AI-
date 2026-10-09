from pathlib import Path
import csv,json,math,hashlib,sys
H=Path(__file__).resolve().parent;R=H.parents[3];OLD=R/'집/코덱스/analysis/ec_highday_classifier_20261009_v1';OLDL=R/'집/코덱스/local'/OLD.name
stage=sys.argv[1];registered=json.loads((H/'registration_v1.json').read_text(encoding='utf8'))
for path,digest in registered['source_pins'].items():assert hashlib.sha256((R/path).read_bytes()).hexdigest()==digest
ref={}
for p in (R/'집/클로드/research/local/ct1_ckpt').glob('DIAG10_*.csv'):
 for q in csv.DictReader(p.open(encoding='utf8')):assert q['row_id'] not in ref;ref[q['row_id']]=q
meta=list(csv.DictReader((OLDL/'metadata_public_v1.csv').open(encoding='utf8')));oldreg=json.loads((OLD/'registration_v1.json').read_text(encoding='utf8'));labels={}
for q in ref.values():labels.setdefault((q['farm'],int(q['day'])),[]).append(float(q['sub_ec']))
means={k:math.fsum(v)/len(v) for k,v in labels.items()};high={k for k,v in means.items() if v>=1.2};assert len(high)==26
raw=[]
for p in (R/'집/클로드/research/local/mx1_ckpt').glob('DIAG10_*.csv'):
 for q in csv.DictReader(p.open(encoding='utf8')):
  r=ref[q['row_id']]
  assert (q['farm'],int(q['day']),int(q['hour']),float(q['sub_ec']),int(q['validation_fold']))==(r['farm'],int(r['day']),int(r['hour']),float(r['sub_ec']),int(r['validation_fold']))
  raw.append(q)
assert len(raw)==8640 and len({q['row_id'] for q in raw})==8640
constants={};summaries=[];rowdays=[(q['farm'],int(q['day'])) for q in meta];locked=set(map(tuple,oldreg['locked_days']))
for q in oldreg['folds']:
 if q['name']!='DIAG10':continue
 query=set(map(tuple,q['query_days']));forbidden={(farm,day+j) for farm,day in query for j in [-1,0,1]}|{('F47' if farm=='F13' else 'F13',day+j) for farm,day in query for j in range(-3,4)}|{(farm,day+j) for farm,day in locked for j in [-1,0,1]};train=[i for i,d in enumerate(rowdays) if d not in forbidden];assert train==q['train_indices']
 ids=[meta[i]['row_id'] for i in train if rowdays[i] in high];count=len(ids)//24;value=math.fsum(float(ref[k]['sub_ec']) for k in ids)/len(ids);constants[q['index']]=value
 qs=[r for r in raw if int(r['validation_fold'])==q['index']];assert {r['row_id'] for r in qs}=={meta[i]['row_id'] for i in q['query_indices']};assert all(int(r['n_high_train_days'])==count for r in qs) and count>=3
 summaries.append({'fold':q['index'],'train_high_days':count,'constant':value})
selected=[]
for q in raw:
 if (q['farm'],int(q['day'])) not in high or (stage=='midpoint' and int(q['validation_fold']) not in [0,1]):continue
 r=dict(q);r['high_constant']=constants[int(q['validation_fold'])]
 for arm in ['BASE','H']:r[arm+'_mean']=math.fsum(float(q[f'{arm}_{s}']) for s in [3737,5858,7979])/3
 selected.append(r)
def metrics(rr,arm):
 y=[float(q['sub_ec']) for q in rr];p=[float(q[arm]) for q in rr];assert all(math.isfinite(t) for t in y+p);n=len(y);e=[b-a for a,b in zip(y,p)];days={}
 for q in rr:days.setdefault((q['farm'],int(q['day'])),[]).append(q)
 assert all(len(qs)==24 for qs in days.values())
 de=[math.fsum(float(q[arm]) for q in qs)/24-math.fsum(float(q['sub_ec']) for q in qs)/24 for qs in days.values()]
 my,mp=math.fsum(y)/n,math.fsum(p)/n
 return dict(rows=n,days=len(days),rmse=math.sqrt(math.fsum(v*v for v in e)/n),mae=math.fsum(abs(v) for v in e)/n,bias=math.fsum(e)/n,actual_mean=my,prediction_mean=mp,actual_std=math.sqrt(math.fsum((v-my)**2 for v in y)/n),prediction_std=math.sqrt(math.fsum((v-mp)**2 for v in p)/n),daily_mean_rmse=math.sqrt(math.fsum(v*v for v in de)/len(de)),daily_mean_mae=math.fsum(abs(v) for v in de)/len(de),daily_mean_within01=sum(abs(v)<=.1 for v in de),daily_mean_within02=sum(abs(v)<=.2 for v in de),daily_mean_within03=sum(abs(v)<=.3 for v in de))
def verify(ours,reported):
 for k,v in ours.items():assert abs(v-reported[k])<1e-12,(k,v,reported[k])
report=json.loads((H/f'{stage}_score_v1.json').read_text(encoding='utf8'))
for q in report['groups']:
 rr=[r for r in selected if q['scope']=='all_high' or (int(r['day'])>=179)==(q['scope']=='pass2_high')];verify(metrics(rr,q['arm']),q)
for q in report['segments']:
 rr=[r for r in selected if r['farm']==q['farm'] and (int(r['day'])>=179)==q['pass2']];verify(metrics(rr,q['arm']),q)
for q in report['folds']:
 expected=next(r for r in summaries if r['fold']==q['fold']);assert expected['train_high_days']==q['train_high_days'] and abs(expected['constant']-q['constant'])<1e-12
out={'status':'INDEPENDENT_HIGH_ONLY_CACHED_AUDIT_PASS','stage':stage,'public_rows_checked':8640,'selected_rows':len(selected),'selected_days':len({(r['farm'],r['day']) for r in selected}),'groups_checked':len(report['groups']),'segments_checked':len(report['segments']),'all10_train_high_counts_constants_rebuilt':True,'same_key_fold_target_verified':True,'conditional_metrics_from_source_csv_rebuilt':True,'extra40_target_or_performance_loaded':False}
p=H/f'critic_{stage}_recheck_v1.json';assert not p.exists();p.write_text(json.dumps(out,indent=2),encoding='utf8');print(json.dumps(out))
