from pathlib import Path
import csv,json,hashlib,sys,math
H=Path(__file__).resolve().parent;R=H.parents[3];L=R/'집/코덱스/local'/H.name;OLD=R/'집/코덱스/local/ec_highday_classifier_20261009_v1';LOO=R/'집/코덱스/analysis/ec_highday_positive_loo_20261009_v1'
stage=sys.argv[1];reg=json.loads((H/'registration_v1.json').read_text(encoding='utf8'));meta=list(csv.DictReader((OLD/'metadata_public_v1.csv').open(encoding='utf8')));loo=json.loads((LOO/'registration_v1.json').read_text(encoding='utf8'))
for path,digest in reg['source_pins'].items():assert hashlib.sha256((R/path).read_bytes()).hexdigest()==digest
for fold,summary in zip(loo['folds'],reg['all26_single_class_folds']):
 idx=[i for i in fold['train_indices'] if int(meta[i]['high'])==1];assert idx and all(int(meta[i]['high'])==1 for i in idx);assert len(idx)==summary['train_rows'] and len(idx)//24==summary['train_days']==fold['train_high'] and summary['classes']==[1]
first=loo['folds'][0];train={meta[i]['row_id'] for i in first['train_indices'] if int(meta[i]['high'])==1};positive={meta[i]['row_id'] for i in first['query_indices']};negative={t['row_id'] for t in meta if int(t['high'])==0};assert len(train)==480 and len(positive)==24 and len(negative)==8016 and train.isdisjoint(positive|negative)
results=[]
for seed in ([8383] if stage=='midpoint' else [8383,1919,7171]):
 rows=list(csv.DictReader((L/f'prediction_{seed}_v1.csv').open(encoding='utf8')));receipt=json.loads((H/f'score_{seed}_v1.json').read_text(encoding='utf8'));assert len(rows)==8040 and len({t['row_id'] for t in rows})==8040 and receipt['classes']==[1]
 assert {t['row_id'] for t in rows if t['scope']=='held_high'}==positive and {t['row_id'] for t in rows if t['scope']=='unseen_ordinary'}==negative
 assert all(float(t['score'])==1 and int(t['predicted'])==1 for t in rows)
 daily={}
 for t in rows:daily.setdefault((t['farm'],int(t['day']),t['scope']),[]).append(t);assert int(t['high'])==int(t['scope']=='held_high')
 assert len(daily)==335 and all(len(v)==24 for v in daily.values())
 assert receipt['train_days']==20 and receipt['held_high_days']==1 and receipt['ordinary_days']==334 and receipt['high_recall']==1 and receipt['ordinary_false_positive_rate']==1 and not receipt['model_saved'] and not receipt['extra40_rescored']
 results.append({'seed':seed,'train_positive_days':20,'held_positive_days':1,'ordinary_days':334,'high_recall':1,'ordinary_false_positive_rate':1,'min_score':1,'max_score':1})
result={'status':'INDEPENDENT_SINGLE_CLASS_CHECK_PASS','stage':stage,'all26_positive_only_subsets_verified':True,'actual_folds_fitted':1,'actual_seeds_checked':len(results),'predictions_per_seed':8040,'independent_daily_group_check':True,'results':results,'extra40_target_or_performance_loaded':False}
p=H/f'critic_{stage}_recheck_v1.json';assert not p.exists();p.write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps(result))
