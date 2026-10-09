from pathlib import Path
import csv,json,hashlib,math,sys
H=Path(__file__).resolve().parent;R=H.parents[3];L=R/'집/코덱스/local'/H.name;OLDH=R/'집/코덱스/analysis/ec_highday_classifier_20261009_v1';OLDL=R/'집/코덱스/local'/OLDH.name
stage=sys.argv[1];reg=json.loads((H/'registration_v1.json').read_text(encoding='utf8'));oldreg=json.loads((OLDH/'registration_v1.json').read_text(encoding='utf8'));meta=list(csv.DictReader((OLDL/'metadata_public_v1.csv').open(encoding='utf8')));source={q['row_id']:q for q in meta};rowdays=[(q['farm'],int(q['day'])) for q in meta];lock=set(map(tuple,oldreg['locked_days']))
for path,digest in reg['source_pins'].items():assert hashlib.sha256((R/path).read_bytes()).hexdigest()==digest
assert len(reg['folds'])==27
for fold in reg['folds']:
 prior=next(q for q in oldreg['folds'] if q['key']==fold['key']);query=set(map(tuple,prior['query_days']));forbidden={(f,d+j) for f,d in query for j in [-1,0,1]}|{('F47' if f=='F13' else 'F13',d+j) for f,d in query for j in range(-3,4)}|{(f,d+j) for f,d in lock for j in [-1,0,1]};train=[i for i,d in enumerate(rowdays) if d not in forbidden and int(meta[i]['high'])==1];valid=[i for i,d in enumerate(rowdays) if d in query]
 assert train==fold['train_indices'] and valid==fold['query_indices'];assert len(train)//24==fold['train_high_days'] and len(valid)//24==fold['query_days'] and all(int(meta[i]['high'])==1 for i in train)
allrows=[];completed=0
for p in sorted((L/'predictions').glob('*_v1.csv')):
 rows=list(csv.DictReader(p.open(encoding='utf8')));fold=next(q for q in reg['folds'] if p.name==q['key']+'_v1.csv');receipt=json.loads(p.with_suffix('.json').read_text(encoding='utf8'));assert {q['row_id'] for q in rows}=={meta[i]['row_id'] for i in fold['query_indices']}
 baseline={q['row_id']:q for q in csv.DictReader((R/fold['baseline']).open(encoding='utf8'))}
 for q in rows:
  rid=q['row_id'];assert q['farm']==source[rid]['farm'] and int(q['day'])==int(source[rid]['day']) and int(q['hour'])==int(source[rid]['hour']) and int(q['high'])==int(source[rid]['high']);s=float(q['oneclass_score']);assert math.isfinite(s) and int(q['oneclass_high'])==int(s>0)
  assert abs(float(q['et_score'])-float(baseline[rid]['ensemble']))<1e-14 and int(q['et_high'])==int(float(q['et_score'])>=.5)
 assert receipt['train_ordinary_days']==0 and receipt['train_rows']==len(fold['train_indices']) and receipt['imputer_fit_rows']==receipt['scaler_fit_rows']==receipt['train_rows'] and receipt['train_query_overlap']==0 and receipt['fit_status']==0
 assert receipt['support_vectors']<=receipt['train_rows'] and 0<=receipt['training_inlier_fraction_all24h']<=1
 allrows+=rows;completed+=1
report=json.loads((H/f'{stage}_score_v1.json').read_text(encoding='utf8'))
def metrics(rows,arm):
 y=[int(q['high']) for q in rows];p=[int(q['oneclass_high']) if arm=='oneclass' else int(q['et_high']) if arm=='et' else int(arm=='always_high') for q in rows];tp=sum(a==1 and b==1 for a,b in zip(y,p));fn=sum(a==1 and b==0 for a,b in zip(y,p));fp=sum(a==0 and b==1 for a,b in zip(y,p));tn=sum(a==0 and b==0 for a,b in zip(y,p));pos=tp+fn;neg=fp+tn
 return dict(days=len(y),high_days=pos,ordinary_days=neg,tp=tp,fn=fn,fp=fp,tn=tn,recall=tp/pos if pos else None,ordinary_false_positive_rate=fp/neg if neg else None,precision=tp/(tp+fp) if tp+fp else 0.,balanced_accuracy=.5*(tp/pos+tn/neg) if pos and neg else None)
def verify(a,b):
 for k,v in a.items():
  assert (v is None)==(b[k] is None)
  if v is not None:assert abs(v-b[k])<1e-12,(k,v,b[k])
for q in report['groups']:
 rows=[r for r in allrows if r['validator']==q['validator'] and int(r['hour'])==q['hour'] and (q['scope']=='all' or int(r['day'])>=179)];verify(metrics(rows,q['arm']),q)
for q in report['segments']:
 rows=[r for r in allrows if r['validator']==q['validator'] and int(r['hour'])==15 and r['farm']==q['farm'] and (int(r['day'])>=179)==q['pass2']];verify(metrics(rows,'oneclass'),q)
assert (report['status']=='FULL_CV')==(completed==27)
result={'status':'INDEPENDENT_ONECLASS_VALIDATION_PASS','stage':stage,'all27_high_only_splits_rebuilt':True,'completed_folds':completed,'prediction_rows':len(allrows),'metric_groups_checked':len(report['groups']),'segments_checked':len(report['segments']),'native_positive_boundary_strict_greater_than_zero':True,'baseline_scores_targets_query_ids_verified':True,'preprocessor_fit_scope_source_and_receipt_review_not_independent_refit':True,'extra40_target_or_performance_loaded':False}
p=H/f'critic_{stage}_recheck_v1.json';assert not p.exists();p.write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps(result))
