"""Public completed CSV audit; stdlib/Decimal/independent bootstrap. No models."""
from pathlib import Path
import csv,json,math,hashlib,sys,statistics
from decimal import Decimal,localcontext
from collections import defaultdict
H=Path(__file__).resolve().parent;ROOT=H.parents[3];OUT=ROOT/'집/코덱스/local'/H.name
sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT/'집/클로드/research'));import env
import numpy as np
MODES=('FINAL_LOSS','RAW_LOSS');SEEDS=(7,101,2024)
FOLDS=[(v,k) for v,n in [('DIAG10',10),('A',5),('B',5),('EXT10',1),('EXT12',1)] for k in range(n)]
ALPHA=.025/24;checks=0;inputs={}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):
 inputs[str(p.relative_to(ROOT))]=sha(p)
 with p.open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))
def near(a,b):
 global checks
 assert math.isfinite(a) and math.isfinite(b) and abs(a-b)<=1e-12,(a,b)
 checks+=1
def rms(rows,col):return math.sqrt(math.fsum((float(r[col])-float(r['y']))**2 for r in rows)/len(rows))
def decimal_rms(rows,col):
 with localcontext() as ctx:
  ctx.prec=45
  s=sum(((Decimal(r[col])-Decimal(r['y']))**2 for r in rows),Decimal(0))
  return (s/Decimal(len(rows))).sqrt()
def quantile(values,p):
 a=(len(values)-1)*p;i=math.floor(a);f=a-i
 return values[i] if i==len(values)-1 else values[i]*(1-f)+values[i+1]*f
whole=json.loads((H/'full_verification_v3.json').read_text(encoding='utf-8'))
inputs['whole']=sha(H/'full_verification_v3.json')
assert whole['cells']==132 and whole['rows']==166320
allrows=[];cells={};prefixgap=0.
for mode in MODES:
 mode_rows=[]
 for v,k in FOLDS:
  for seed in SEEDS:
   rows=read(OUT/f'{mode}_{v}_{k}_{seed}_pred.csv');cells[mode,v,k,seed]=rows
   assert len({r['row_id'] for r in rows})==len(rows)
   assert all(r['mode']==mode and r['validator']==v and int(r['fold'])==k and int(r['seed'])==seed for r in rows)
   groups=defaultdict(list)
   for i,r in enumerate(rows):groups[r['farm'],int(r['day'])].append((int(r['hour']),i,r))
   for key,g in groups.items():
    assert {h for h,i,r in g}==set(range(24)) and len(g)==24
    history=[];basehist=[]
    for h,i,r in sorted(g):
     lo,hi=float(r['clip_lo']),float(r['clip_hi']);assert lo<=hi
     raw=math.fsum([float(r['base_raw']),float(r['delta'])]);history.append(raw);basehist.append(float(r['base_raw']))
     got=min(hi,max(lo,.5*raw+.5*math.fsum(history)/len(history)))
     baseline=min(hi,max(lo,.5*float(r['base_raw'])+.5*math.fsum(basehist)/len(basehist)))
     prefixgap=max(prefixgap,abs(got-float(r['candidate'])),abs(baseline-float(r['baseline'])))
     near(got,float(r['candidate']));near(baseline,float(r['baseline']))
   mode_rows.extend(rows)
 assert mode_rows==read(OUT/f'oof_{mode}.csv') and len(mode_rows)==83160
 allrows.extend(mode_rows)
assert allrows==read(OUT/'oof.csv') and len(allrows)==166320
scores=read(H/'full_scores_v3.csv');freshscores=[];buckets=defaultdict(list)
for r in allrows:buckets[r['mode'],r['validator'],int(r['seed'])].append(r)
for s in scores:
 mode,v,seed=s['mode'],s['validator'],int(s['seed']);rows=buckets[mode,v,seed]
 rb,rc=rms(rows,'baseline'),rms(rows,'candidate');db,dc=decimal_rms(rows,'baseline'),decimal_rms(rows,'candidate')
 near(rb,float(db));near(rc,float(dc));near(rb,float(s['baseline']));near(rc,float(s['candidate']));assert (rc<rb)==(dc<db)
 pct=100*(rc/rb-1);near(pct,float(s['change_pct']))
 assert len(rows)==int(s['n'])
 freshscores.append(dict(mode=mode,validator=v,seed=seed,n=len(rows),baseline=rb,candidate=rc,change_pct=pct,decimal_candidate=str(dc),improved=rc<rb))
segments=read(H/'full_segments_v3.csv');freshsegments=[];boots={};dailycache={}
for mode in MODES:
 boots[mode]={}
 for seed in SEEDS:
  rows=buckets[mode,'DIAG10',seed];days=defaultdict(list)
  for r in rows:days[r['farm'],int(r['day'])].append(r)
  high={key for key,rs in days.items() if math.fsum(float(r['y']) for r in rs)/len(rs)>=1}
  assert len(days)==360 and len(high)==31
  dailycache[mode,seed]=(days,high)
  rng=np.random.default_rng(20261003+seed);blocks={};draws={}
  for farm in ('F13','F47'):
   ordered=sorted(day for f,day in days if f==farm)
   blocks[farm]=[(math.fsum((float(r['candidate'])-float(r['y']))**2-(float(r['baseline'])-float(r['y']))**2 for day in ordered[i:i+5] for r in days[farm,day]),sum(len(days[farm,day]) for day in ordered[i:i+5])) for i in range(0,len(ordered),5)]
   draws[farm]=rng.integers(len(blocks[farm]),size=(20000,len(blocks[farm])))
  samples=[]
  for i in range(20000):
   sums=[];n=0
   for farm in ('F13','F47'):
    for index in draws[farm][i]:value,count=blocks[farm][index];sums.append(value);n+=count
   samples.append(math.fsum(sums)/n)
  sorted_s=sorted(samples);ci=[quantile(sorted_s,p) for p in [ALPHA,1-ALPHA]];ci95=[quantile(sorted_s,p) for p in [.025,.975]]
  pw=sum(s>=0 for s in samples)/len(samples);record=whole['bootstrap'][mode][str(seed)]
  assert pw==record['p_worse']
  for a,b in zip(ci+ci95,record['ci_adjusted']+record['ci95']):near(a,b)
  gate=pw<ALPHA and ci[1]<0;assert gate==record['pass_gate']
  boots[mode][str(seed)]=dict(p_worse=pw,ci_adjusted=ci,ci95=ci95,pass_gate=gate)
for s in segments:
 mode,seed,name=s['mode'],int(s['seed']),s['segment'];rows=buckets[mode,'DIAG10',seed];days,high=dailycache[mode,seed]
 predicates={'high':lambda r:(r['farm'],int(r['day'])) in high,'ordinary':lambda r:(r['farm'],int(r['day'])) not in high,'late':lambda r:int(r['day'])>=179,'F13':lambda r:r['farm']=='F13','F47':lambda r:r['farm']=='F47','hour0':lambda r:int(r['hour'])==0}
 rs=[r for r in rows if predicates[name](r)];rb,rc=rms(rs,'baseline'),rms(rs,'candidate')
 bb=math.fsum(float(r['baseline'])-float(r['y']) for r in rs)/len(rs);cb=math.fsum(float(r['candidate'])-float(r['y']) for r in rs)/len(rs)
 for val,col in [(rb,'baseline'),(rc,'candidate'),(bb,'baseline_bias'),(cb,'candidate_bias')]:near(val,float(s[col]))
 assert len(rs)==int(s['n']) and len({(r['farm'],r['day']) for r in rs})==int(s['days'])
 freshsegments.append(dict(mode=mode,seed=seed,segment=name,n=len(rs),days=int(s['days']),baseline=rb,candidate=rc,change_pct=100*(rc/rb-1),baseline_bias=bb,candidate_bias=cb,sse_change=math.fsum((float(r['candidate'])-float(r['y']))**2-(float(r['baseline'])-float(r['y']))**2 for r in rs)))
training=read(H/'training_and_outer_loss_v3.csv');trainstats=[]
assert len(training)==132
for mode in MODES:
 selected=[r for r in training if r['mode']==mode];drop=[];finalgaps=[]
 for r in selected:
  v,k,seed=r['validator'],int(r['fold']),int(r['seed']);meta=json.loads((OUT/f'{mode}_{v}_{k}_{seed}.json').read_text(encoding='utf-8'))
  t=meta['train'];li,lf=float(r['train_loss_initial']),float(r['train_loss_final'])
  near(li,t['loss_initial']);near(lf,t['loss_final']);assert int(r['train_rows'])==t['training_rows'] and t['epochs']==400
  outer=rms(cells[mode,v,k,seed],'candidate');near(outer,float(r['outer_rmse']))
  drop.append(100*(lf/li-1));finalgaps.append(outer-math.sqrt(lf))
 trainstats.append(dict(mode=mode,cells=len(selected),decreased=sum(float(r['train_loss_final'])<float(r['train_loss_initial']) for r in selected),loss_change_pct_min=min(drop),loss_change_pct_median=statistics.median(drop),loss_change_pct_max=max(drop),outer_minus_sqrt_train_loss_min=min(finalgaps),outer_minus_sqrt_train_loss_median=statistics.median(finalgaps),outer_minus_sqrt_train_loss_max=max(finalgaps),warning='RAW training raw-output MSE and outer clipped MSE differ; FINAL inner/outer populations and clip bounds differ. These gaps do not prove overfitting.'))
matched=read(H/'matched_loss_comparison_v3.csv');comparisons=[]
for r in matched:
 v,seed=r['validator'],int(r['seed']);a=rms(buckets['FINAL_LOSS',v,seed],'candidate');b=rms(buckets['RAW_LOSS',v,seed],'candidate');pct=100*(a/b-1)
 near(a,float(r['final_rmse']));near(b,float(r['raw_loss_rmse']));near(pct,float(r['change_pct']));comparisons.append(dict(validator=v,seed=seed,final_vs_raw_pct=pct,final_better=a<b))
decisions={mode:all(s['improved'] for s in freshscores if s['mode']==mode) and all(b['pass_gate'] for b in boots[mode].values()) for mode in MODES}
assert decisions==whole['candidate_pass']
result=dict(status='PASS_INDEPENDENT_PUBLIC_CSV_DECIMAL_BOOTSTRAP',rows=166320,cells=132,aggregate_exact_string_rows=True,scalar_prefix_maxdiff=prefixgap,checks=checks,precision_decimal=45,scores=freshscores,bootstrap=boots,segments=freshsegments,training=trainstats,matched=comparisons,candidate_pass=decisions,input_sha256=inputs,new_native_fit=0,new_native_predict=0,raw_ec_reads=0,test_reads=0,EL1_reads=0)
with (H/'crosscheck_complete_loss_result_v1.json').open('x',encoding='utf-8') as out:json.dump(result,out,ensure_ascii=False,indent=2)
print('PASS_INDEPENDENT',len(allrows),checks,decisions)
