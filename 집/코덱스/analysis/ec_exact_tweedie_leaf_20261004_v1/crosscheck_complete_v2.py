"""Run ONLY after first/66-fit/root whole PASS. Public CSV/Decimal/manual boot.
No native fitting/prediction; plain-text checkpoint recipe audit only.
"""
from pathlib import Path
import csv,json,math,hashlib,sys,statistics,re
from decimal import Decimal,localcontext
from collections import defaultdict
H=Path(__file__).resolve().parent;ROOT=H.parents[3];OUT=ROOT/'집/코덱스/local'/H.name
sys.dont_write_bytecode=True
SEEDS=(7,101,2024);VALIDATORS=('DIAG10','A','B','EXT10','EXT12')
FOLDS=[(v,k) for v,n in zip(VALIDATORS,(10,5,5,1,1)) for k in range(n)]
ALPHA=.025/24;ATOL=1e-12
COLS=['row_id','farm','day','hour','y','baseline','candidate','new_lgb_raw','raw_et','raw_lgb','raw_mlp','old_pfn_raw','validator','fold','seed','clip_lo','clip_hi']
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def readj(p):
 def unique(items):
  d={}
  for k,v in items:assert k not in d;d[k]=v
  return d
 return json.loads(p.read_text(encoding='utf-8'),object_pairs_hook=unique,parse_constant=lambda x:(_ for _ in ()).throw(ValueError(x)))
def csvrows(p):
 with p.open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))
def near(a,b):assert math.isfinite(a) and math.isfinite(b) and abs(a-b)<=ATOL,(a,b)
def rms(rows,key):return math.sqrt(math.fsum((float(r[key])-float(r['y']))**2 for r in rows)/len(rows))
def decimal_rms(rows,key):
 with localcontext() as c:
  c.prec=45
  return (sum(((Decimal(r[key])-Decimal(r['y']))**2 for r in rows),Decimal(0))/Decimal(len(rows))).sqrt()
def qmanual(x,p):
 position=(len(x)-1)*p;i=math.floor(position);f=position-i
 return x[i] if i==len(x)-1 else x[i]*(1-f)+x[i+1]*f
def typed(s):
 if s in ['true','false']:return s=='true'
 try:return int(s)
 except ValueError:
  try:return float(s)
  except ValueError:return s
def checkpoint_recipe(path,features,seed,sourceparams):
 text=path.read_text(encoding='utf-8');header={}
 for line in text.splitlines():
  if line.startswith('Tree='):break
  if '=' in line:key,value=line.split('=',1);header[key]=value
 assert header['feature_names'].split()==features and len(features)==14
 assert header['objective']=='tweedie_exact_leaf rho:1.5 lambda_l2:1'
 trees=[int(x) for x in re.findall(r'^Tree=(\d+)$',text,re.M)];assert trees==list(range(len(trees))) and 0<len(trees)<=800
 params={}
 for key,value in re.findall(r'^\[([^:]+): (.*?)\]$',text,re.M):assert key not in params;params[key]=typed(value)
 aliases=dict(num_iterations='n_estimators',learning_rate='learning_rate',bagging_fraction='subsample',bagging_freq='subsample_freq',feature_fraction='colsample_bytree',lambda_l2='reg_lambda',lambda_l1='reg_alpha',tweedie_variance_power='tweedie_variance_power',num_leaves='num_leaves',min_data_in_leaf='min_child_samples',seed='random_state',num_threads='n_jobs',deterministic='deterministic',force_col_wise='force_col_wise',max_depth='max_depth',min_sum_hessian_in_leaf='min_child_weight',min_gain_to_split='min_split_gain',bin_construct_sample_cnt='subsample_for_bin',verbosity='verbose',boosting='boosting_type')
 for canonical,original in aliases.items():assert params[canonical]==sourceparams[original],(canonical,params.get(canonical),sourceparams[original])
 assert params['objective']=='tweedie_exact_leaf' and params['seed']==seed and params['num_iterations']==800
 return len(trees)
def main():
 # Completion gate BEFORE any candidate CSV/score read, model digest or metric.
 required=['first_fold_verification_v1.json','fit_audit_v1.json','compiled_newton_all66_v1.json','full_verification_v4.json','full_scores_v4.csv','full_segments_v4.csv']
 assert all((H/p).is_file() for p in required),'WAIT: first/fit/root whole not complete'
 whole=readj(H/'full_verification_v4.json');first=readj(H/'first_fold_verification_v1.json');fit=readj(H/'fit_audit_v1.json');controls=readj(H/'compiled_newton_all66_v1.json')
 assert whole['status']=='PASS' and whole['cells']==66 and whole['rows']==83160 and whole['score_cells']==15 and whole['family']==24 and whole['alpha']==ALPHA
 assert first['status']=='PASS' and first['cpp_audit']['status']=='PASS' and first['atol']==ATOL
 assert fit['status']=='PASS' and fit['score_count']==0 and len(fit['cells'])==66
 assert controls['status']=='PASS' and controls['score_count']==0 and len(controls['cells'])==66 and controls['atol']==ATOL
 assert not (H/'crosscheck_complete_result_v2.json').exists(),'Preserve previous result'
 receipt=readj(H/'registered_execution_v1.json');prep=readj(H/'preparation_v3.json')
 assert receipt['status']=='REGISTERED_BEFORE_ACTUAL_FIT' and receipt['fit_count']==receipt['predict_count']==receipt['score_count']==0
 for rel,digest in receipt['files'].items():assert sha(ROOT/rel)==digest

 repair=readj(H/'portability_repair_v1.json')
 old=(H/'verify_full_v3.py').read_text(encoding='utf-8');new=(H/'verify_full_v4.py').read_text(encoding='utf-8')
 changes=[("destinations=[H/'full_verification_v3.json',H/'full_scores_v3.csv',H/'full_segments_v3.csv']","destinations=[H/'full_verification_v4.json',H/'full_scores_v4.csv',H/'full_segments_v4.csv']"),("lightgbm.Booster(model_file=str(checkpoint))","lightgbm.Booster(model_str=checkpoint.read_text(encoding='utf-8'))"),("target=H/'synthetic_verify_full_v3.json'","target=H/'synthetic_verify_full_v4.json'")]
 expectedsource=old
 for before,after in changes:assert expectedsource.count(before)==1;expectedsource=expectedsource.replace(before,after)
 assert expectedsource==new
 import difflib
 assert repair['diff']==list(difflib.unified_diff(old.splitlines(),new.splitlines()))
 assert sha(H/'verify_full_v3.py')==repair['old_verifier_sha256']=='4093283b32509ea848d2ac43fed00e91ed191e500527ab35644cb5ffd991357e'
 assert sha(H/'verify_full_v4.py')==repair['new_verifier_sha256']=='5ea91e3b768c889a6d333ff486f14cb45d79c48d991643b8a498f828c1171f0a'
 assert repair['rules_changed'] is False and repair['retrain']==0 and repair['candidate_inputs_modified'] is False
 for file,key in [('run_v3.py','runner_sha256'),('preparation_v3.json','prep_sha256'),('preregistration_v1.md','prereg_sha256')]:assert sha(H/file)==repair[key]
 assert sha(H/'verify_full_v3.log')==repair['first_v3_failure_log_sha256']
 assert fit['preparation']==prep and prep['family']==24 and prep['alpha']==ALPHA and len(prep['manifest'])==22
 for name,field in [('verify_full_v4.py','verifier_sha256'),('run_v3.py','runner_sha256'),('preparation_v3.json','preparation_sha256'),('preregistration_v1.md','preregistration_sha256'),('fit_audit_v1.json','fit_audit_sha256'),('first_fold_verification_v1.json','first_audit_sha256'),('full_scores_v4.csv','scores_sha256'),('full_segments_v4.csv','segments_sha256')]:assert sha(H/name)==whole[field]
 assert sha(H/'compiled_newton_all66_v1.json')==fit['compiled_newton_all66_sha256']
 assert sha(OUT/'first_exact_cpp_trace_v1.ndjson')==first['cpp_trace_sha256']
 sys.path.insert(0,str(ROOT/'집/클로드/research'));import env
 import numpy as np
 publicpath=ROOT/'집/코덱스/local/ec_dc4_integration_20261002_v1/v2_integration_oof.csv'
 assert sha(publicpath)==prep['inputs']['public_oof']
 public=csvrows(publicpath);labels={};baseline={}
 for r in public:
  key=(r['validator'],int(r['validation_fold']),int(r['seed']),r['row_id']);assert key not in baseline;baseline[key]=float(r['season_v2'])
  if r['validator']=='DIAG10' and int(r['seed'])==7:
   assert r['row_id'] not in labels;labels[r['row_id']]=float(r['sub_ec'])
 assert len(labels)==8640
 allrows=[];metas=[];treecounts=[];prefixgap=0.;rawdelta_gap=0.
 modelcounts={(r['validator'],r['fold'],r['seed']):r['trees'] for r in whole['actual_tree_counts']};assert len(modelcounts)==66
 for j,(v,k) in enumerate(FOLDS):
  sig=prep['manifest'][j];assert (sig['validator'],sig['fold'])==(v,k) and sig==whole['manifest'][j]
  for rel,digest in sig['cache_hashes'].items():assert sha(ROOT/rel)==digest
  for seed in SEEDS:
   name=f'{v}_{k}_{seed}';meta=readj(OUT/(name+'.json'));path=OUT/(name+'_pred.csv');modelpath=OUT/(name+'_model.txt')
   expected=dict(sig,seed=seed,preparation_sha256=sha(H/'preparation_v3.json'),preregistration_sha256=sha(H/'preregistration_v1.md'))
   assert meta['status']=='PASS' and meta['signature']==expected and meta['csv_sha256']==sha(path) and meta['model_sha256']==sha(modelpath)
   assert meta['first_audit_sha256']==sha(H/'first_fold_verification_v1.json')
   count=checkpoint_recipe(modelpath,sig['new_lgb_features'],seed,prep['params'][str(seed)]);assert count==modelcounts[v,k,seed]
   treecounts.append(dict(validator=v,fold=k,seed=seed,trees=count))
   control=readj(OUT/(name+'_newton_control.json'));assert control==controls['cells'][j*3+SEEDS.index(seed)] and control['signature']==sig and control['seed']==seed and control['npz_sha256']==sha(OUT/(name+'_newton_control.npz'));assert 0<=control['max_error']<=ATOL
   rows=csvrows(path);assert list(rows[0])==COLS and len({r['row_id'] for r in rows})==len(rows)
   ids='\n'.join(r['row_id'] for r in rows);assert hashlib.sha256(ids.encode()).hexdigest()==sig['query_ids']
   groups=defaultdict(list)
   for r in rows:
    assert r['validator']==v and int(r['fold'])==k and int(r['seed'])==seed
    for col in COLS:
     if col not in ['row_id','farm','validator']:assert math.isfinite(float(r[col]))
    near(float(r['y']),labels[r['row_id']]);near(float(r['baseline']),baseline[v,k,seed,r['row_id']])
    near(float(r['clip_lo']),sig['bounds'][0]);near(float(r['clip_hi']),sig['bounds'][1]);groups[r['farm'],int(r['day'])].append(r)
   for key,g in groups.items():
    assert len(g)==24 and {int(r['hour']) for r in g}==set(range(24));history=[];basehistory=[]
    for r in sorted(g,key=lambda r:int(r['hour'])):
     et,new,old,mlp,pfn=[float(r[c]) for c in ['raw_et','new_lgb_raw','raw_lgb','raw_mlp','old_pfn_raw']]
     raw=math.fsum([.48*et,.24*new,.08*mlp,.2*pfn]);oldraw=math.fsum([.48*et,.24*old,.08*mlp,.2*pfn]);near(raw-oldraw,.24*(new-old));rawdelta_gap=max(rawdelta_gap,abs(raw-oldraw-.24*(new-old)))
     history.append(raw);basehistory.append(oldraw);lo,hi=sig['bounds'];assert lo<=hi
     for actual,value,hist in [(float(r['candidate']),raw,history),(float(r['baseline']),oldraw,basehistory)]:
      expectedfinal=min(hi,max(lo,.5*value+.5*math.fsum(hist)/len(hist)));near(actual,expectedfinal);prefixgap=max(prefixgap,abs(actual-expectedfinal))
   metas.append(meta);allrows.extend(rows)
 assert metas==fit['cells'] and len(allrows)==83160 and allrows==csvrows(OUT/'oof.csv') and sha(OUT/'oof.csv')==whole['aggregate_sha256']
 buckets=defaultdict(list)
 for r in allrows:buckets[r['validator'],int(r['seed'])].append(r)
 scores=csvrows(H/'full_scores_v4.csv');freshscores=[]
 assert len(scores)==15
 for s in scores:
  v,seed=s['validator'],int(s['seed']);rows=buckets[v,seed];rb,rc=rms(rows,'baseline'),rms(rows,'candidate');db,dc=decimal_rms(rows,'baseline'),decimal_rms(rows,'candidate');near(rb,float(db));near(rc,float(dc));assert (rc<rb)==(dc<db)
  near(rb,float(s['baseline']));near(rc,float(s['candidate']));near(100*(rc/rb-1),float(s['change_pct']));assert len(rows)==int(s['n'])
  freshscores.append(dict(validator=v,seed=seed,n=len(rows),baseline=rb,candidate=rc,decimal_candidate=str(dc),change_pct=100*(rc/rb-1),improved=rc<rb))
 boots={};daygroups={};highsets={}
 for seed in SEEDS:
  days=defaultdict(list)
  for r in buckets['DIAG10',seed]:days[r['farm'],int(r['day'])].append(r)
  assert len(days)==360;daygroups[seed]=days;highsets[seed]={key for key,g in days.items() if math.fsum(float(r['y']) for r in g)/len(g)>=1};assert len(highsets[seed])==31
  rng=np.random.default_rng(20261003+seed);blocks={};draws={}
  for farm in ['F13','F47']:
   ds=sorted(day for f,day in days if f==farm)
   blocks[farm]=[(math.fsum((float(r['candidate'])-float(r['y']))**2-(float(r['baseline'])-float(r['y']))**2 for day in ds[i:i+5] for r in days[farm,day]),sum(len(days[farm,day]) for day in ds[i:i+5])) for i in range(0,len(ds),5)]
   draws[farm]=rng.integers(len(blocks[farm]),size=(20000,len(blocks[farm])))
  samples=[]
  for i in range(20000):
   values=[];n=0
   for farm in ['F13','F47']:
    for ix in draws[farm][i]:value,count=blocks[farm][ix];values.append(value);n+=count
   samples.append(math.fsum(values)/n)
  ordered=sorted(samples);pw=sum(x>=0 for x in samples)/20000;ci=[qmanual(ordered,p) for p in [ALPHA,1-ALPHA]];ci95=[qmanual(ordered,p) for p in [.025,.975]];rootb=whole['bootstrap'][str(seed)]
  assert pw==rootb['p_worse']
  for a,b in zip(ci+ci95,rootb['ci_adjusted']+rootb['ci95']):near(a,b)
  gate=pw<ALPHA and ci[1]<0;assert gate==(rootb['p_worse']<ALPHA and rootb['ci_adjusted'][1]<0);boots[str(seed)]=dict(p_worse=pw,ci_adjusted=ci,ci95=ci95,pass_gate=gate)
 segments=[]
 for s in csvrows(H/'full_segments_v4.csv'):
  seed=int(s['seed']);name=s['segment'];predicates={'high':lambda r:(r['farm'],int(r['day'])) in highsets[seed],'ordinary':lambda r:(r['farm'],int(r['day'])) not in highsets[seed],'late':lambda r:int(r['day'])>=179,'F13':lambda r:r['farm']=='F13','F47':lambda r:r['farm']=='F47','hour0':lambda r:int(r['hour'])==0}
  rows=[r for r in buckets['DIAG10',seed] if predicates[name](r)];rb,rc=rms(rows,'baseline'),rms(rows,'candidate');bb=math.fsum(float(r['baseline'])-float(r['y']) for r in rows)/len(rows);cb=math.fsum(float(r['candidate'])-float(r['y']) for r in rows)/len(rows)
  for a,key in [(rb,'baseline'),(rc,'candidate'),(bb,'baseline_bias'),(cb,'candidate_bias')]:near(a,float(s[key]))
  assert len(rows)==int(s['n']) and len({(r['farm'],r['day']) for r in rows})==int(s['days'])
  segments.append(dict(seed=seed,segment=name,n=len(rows),days=int(s['days']),baseline=rb,candidate=rc,change_pct=100*(rc/rb-1),baseline_bias=bb,candidate_bias=cb))
 passed=all(s['improved'] for s in freshscores) and all(b['pass_gate'] for b in boots.values());assert passed==whole['public_pass']
 result=dict(status='PASS_INDEPENDENT_COMPLETE_PUBLIC_CHECK',rows=83160,cells=66,aggregate_exact_string_rows=True,scalar_prefix_maxdiff=prefixgap,raw_delta_maxdiff=rawdelta_gap,scores=freshscores,bootstrap=boots,segments=segments,actual_tree_counts=treecounts,registered_files=len(receipt['files']),public_pass=passed,precision_decimal=45,new_native_fit=0,new_native_predict=0,raw_ec_reads=0,test_reads=0,EL1_reads=0,source_sha256=sha(Path(__file__)),root_whole_sha256=sha(H/'full_verification_v4.json'),limitations=['Plain checkpoint text recipe/count verification; independent prediction not performed','First actual trace arithmetic/membership replay is parent whole evidence','No optimizer or native score-cache replay','Repeated public validation; no new untouched holdout','Registration timing uses parent main evidence'])
 with (H/'crosscheck_complete_result_v2.json').open('x',encoding='utf-8') as out:json.dump(result,out,ensure_ascii=False,indent=2)
 print(result['status'],passed)
if __name__=='__main__':main()
