from pathlib import Path
import csv,json,hashlib,math,sys
import runtime_v1
import numpy as np,joblib
H=Path(__file__).resolve().parent;R=H.parents[3];L=R/'집/코덱스/local'/H.name;M=L/'model_full_v1'
report=json.loads((H/'final_score_v1.json').read_text(encoding='utf8'));assert report['status']=='FULL_CV'
manifest=json.loads((M/'model_manifest_v1.json').read_text(encoding='utf8'));assert manifest['training_days']==400 and manifest['training_rows']==9600 and len(manifest['models'])==3 and manifest['ec_threshold']==1.2
for name,digest in manifest['code_hashes'].items():assert hashlib.sha256((H/name).read_bytes()).hexdigest()==digest
for name,digest in manifest['source_sha'].items():assert hashlib.sha256((R/name).read_bytes()).hexdigest()==digest
assert hashlib.sha256((H/'final_score_v1.json').read_bytes()).hexdigest()==manifest['public_cv_score_sha']
features={q['row_id']:q for q in csv.DictReader((L/'features_public_v1.csv').open(encoding='utf8'))};probe=list(csv.DictReader((M/'probe_input_v1.csv').open(encoding='utf8')));actual={q['row_id']:q for q in csv.DictReader((M/'probe_prediction_v1.csv').open(encoding='utf8'))}
# These 73 public feature values were independently rebuilt from raw input by critic_prepare_recheck_v1.py.
assert json.loads((H/'critic_prepare_recheck_v1.json').read_text(encoding='utf8'))['status']=='INDEPENDENT_FEATURE_LABEL_PURGE_PASS'
cols=manifest['feature_columns'];mat=np.array([[float(features[q['row_id']][k]) if features[q['row_id']][k] else float('nan') for k in cols] for q in probe]);preds=[]
for m in manifest['models']:
 p=M/m['file'];assert hashlib.sha256(p.read_bytes()).hexdigest()==m['sha'];pipeline=joblib.load(p);clf=pipeline.named_steps['clf'];assert clf.n_estimators==300 and clf.max_depth==10 and clf.min_samples_leaf==24 and clf.random_state==m['seed'];assert list(pipeline.classes_)==[0,1]
 v=pipeline.predict_proba(mat)[:,1];preds.append(v)
score=np.array([sum(float(v[i]) for v in preds)/3 for i in range(len(probe))]);expected=np.array([float(actual[q['row_id']]['high_ec_score']) for q in probe]);gap=float(max(abs(score-expected)));assert gap<1e-12
for i,q in enumerate(probe):assert int(actual[q['row_id']]['high_ec_class'])==int(score[i]>=.5)
# Integrity only: inspect all400 target means after FULL_CV; no consumed40 predictions or scores.
y=next(R/name for name in manifest['source_sha'] if name.endswith('train_y.csv'));means={}
for q in csv.DictReader(y.open(encoding='utf8')):
 farm,day,h=q['row_id'].split('_')
 if farm in ['F13','F47']:means.setdefault((farm,int(day)),[]).append(float(q['sub_ec']))
assert len(means)==400 and all(len(v)==24 for v in means.values());high=sum(math.fsum(v)/24>=1.2 for v in means.values());assert high==manifest['high_days']
result={'status':'INDEPENDENT_SAVED_CLASSIFIER_PASS','probe_rows':len(probe),'model_components':3,'independently_rebuilt_public_features_used':True,'probability_max_abs_diff':gap,'class_threshold_match':True,'code_source_model_sha_match':True,'full400_high_day_count_rechecked':high,'extra40_rescored':False}
p=H/'critic_final_model_recheck_v1.json';assert not p.exists();p.write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps(result))
