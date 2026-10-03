from pathlib import Path
import sys,csv,math,json
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;R=H.parents[3]
sys.path.insert(0,str(R/'집/클로드/research'));import env
import numpy as np,pandas as pd
from scipy.stats import false_discovery_control
checks=0
def close(a,b):
 global checks
 if pd.isna(a) and pd.isna(b):checks+=1;return
 assert math.isclose(float(a),float(b),rel_tol=2e-9,abs_tol=2e-9),(a,b)
 checks+=1
def rows(path):
 with open(path,encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def number(x):return float(x) if x else float('nan')
def avg(v):v=[x for x in v if math.isfinite(x)];return math.fsum(v)/len(v) if v else float('nan')
raw={r['row_id']:r for r in rows(Path(env.DATA)/'train_X.csv')}
groups={}
for tar,path in [('TEMP',R/'집/코덱스/local/temp_tk_season_20261003_v1/TK1_predictions.csv'),('EC',R/'집/코덱스/local/ec_dc4_integration_20261002_v1/v2_integration_oof.csv')]:
 for z in rows(path):
  if z['validator']!='DIAG10':continue
  if tar=='TEMP' and not(z['member']=='W30G' and z['base_seed']=='7' and z['context']=='1-8'):continue
  if tar=='EC' and z['seed']!='7':continue
  key=(tar,z['farm'],int(z['day']));groups.setdefault(key,[]).append(z)
daily=pd.read_csv(H/'final_daily_cases_v2.csv',float_precision='round_trip').set_index(['target','farm','day']);fail={};verified={}
for k,rs in groups.items():
 tar,f,day=k;rs.sort(key=lambda r:int(r['hour']));ys=[number(r['sub_temp' if tar=='TEMP' else 'sub_ec']) for r in rs];ps=[number(r['prediction' if tar=='TEMP' else 'season_v2']) for r in rs];air=[number(raw[r['row_id']]['in_temp']) for r in rs];error=[p-y for p,y in zip(ps,ys)]
 y=avg(ys);p=avg(ps);bias=avg(error);sse=math.fsum(e*e for e in error);gap=y-avg(air) if tar=='TEMP' else float('nan');rmse=math.sqrt(sse/len(rs));nvalid=sum(math.isfinite(v) for v in air);event=(nvalid==24 and abs(gap)>=2) if tar=='TEMP' else y>=1;failure=event and rmse>(.5 if tar=='TEMP' else .1)
 fail[k]=failure;verified[k]=dict(sse=sse,bias=bias,n=len(rs),gap=gap,event=event,rmse=rmse)
 z=daily.loc[k]
 for c,v in [('truth',y),('prediction',p),('bias',bias),('sse',sse),('gap',gap),('rmse',rmse),('air_n',nvalid),('level_sse_fraction',len(rs)*bias*bias/sse)]:close(z[c],v)
 assert bool(z.failure)==failure and bool(z.event)==event;checks+=2
 for phase,hs in [('night',range(7)),('day',range(7,17)),('evening',range(17,24))]:close(z[phase+'_bias'],avg([error[i] for i in hs]));close(z[phase+'_sse_fraction'],math.fsum(error[i]**2 for i in hs)/sse)
 # Independent selected input summaries and relations, without exploratory helpers.
 for c in ['in_temp','in_hum','in_co2','out_temp','out_rad','act_vent','act_shade','act_thermal']:
  vals=[number(raw[r['row_id']][c]) for r in rs];good=[v for v in vals if math.isfinite(v)];mu=avg(good);sd=math.sqrt(avg([(v-mu)**2 for v in good]))
  close(z[c+'__mean'],mu);close(z[c+'__std'],sd);close(z[c+'__h0'],vals[0]);close(z[c+'__daynight'],avg(vals[9:17])-avg(vals[:7]))
summary=pd.read_csv(H/'cohort_summary.csv',float_precision='round_trip')
for r in summary.itertuples():
 q=daily.loc[r.target];z=q if r.group=='all' else q[q[r.group]]
 close(r.n,len(z));close(r.sse_fraction,math.fsum(z.sse)/math.fsum(q.sse));close(r.rmse,math.sqrt(math.fsum(z.sse)/math.fsum(z.n)))
s=pd.read_csv(H/'conditions_with_date_control_v2.csv',float_precision='round_trip')
bhs=false_discovery_control(s.p_permutation.to_numpy());bhfine=false_discovery_control(s.p_fine_date.to_numpy())
for r in s.itertuples():
 q=daily.loc[r.target];v=q[r.feature];valid=np.isfinite(v);a=(v>=r.cut) if r.operator=='>=' else v<=r.cut;f=q.failure&valid
 close(r.failure_n,int(f.sum()));close(r.failure_yes,int((a&f).sum()));close(r.failure_rate,float(a[f].mean()))
 close(r.q_BH,bhs[r.Index]);close(r.q_fine_date_BH,bhfine[r.Index])
 for lab in ['core_failure','strict1','strict2']:
  f=q[lab]&valid;close(getattr(r,lab+'_n'),int(f.sum()));close(getattr(r,lab+'_rate'),float(a[f].mean()) if f.any() else float('nan'))
# Training references must be training-only and all nearest rows actual public days.
cov=pd.read_csv(H/'actual_training_coverage_v2.csv',float_precision='round_trip');nei=pd.read_csv(H/'actual_training_neighbors_v2.csv',float_precision='round_trip')
assert not ((nei.farm==nei.neighbor_farm)&(nei.day==nei.neighbor_day)).any();checks+=1
for r in cov.itertuples():
 ns=nei[(nei.target==r.target)&(nei.farm==r.farm)&(nei.day==r.day)].sort_values('distance').head(5);close(r.nearest_distance,ns.distance.min())
 if r.target=='TEMP':close(r.top5_severe_same_sign,sum(abs(n.neighbor_gap)>=2 and np.sign(n.neighbor_gap)==np.sign(r.query_gap) for n in ns.itertuples()))
 else:close(r.top5_high,int((ns.neighbor_truth>=1).sum()));close(r.top5_truth_gap,r.query_truth-avg(ns.neighbor_truth))
claims={}
for tar in ['TEMP','EC']:
 q=daily.loc[tar];f=q[q.failure];core=q[q.core_failure];claims[tar]=dict(failure_days=len(f),core_days=len(core),failure_sse_fraction=math.fsum(f.sse)/math.fsum(q.sse),core_sse_fraction=math.fsum(core.sse)/math.fsum(q.sse),failure_level_sse_fraction=math.fsum(f.n*f.bias**2)/math.fsum(f.sse),core_level_sse_fraction=math.fsum(core.n*core.bias**2)/math.fsum(core.sse))
p=H/'verification_v4.json';assert not p.exists();p.write_text(json.dumps(dict(status='PASS',checks=checks,claims=claims,independent_methods=['raw CSV DictReader and math.fsum daily outcomes','manual input means/variance/phases','SciPy independent BH correction','all condition membership/counts','nearest coverage cross-check'],limitations=['not proof of causal mechanisms','nearest-distance full split reproduction is separate coverage audit','permutations assume exchangeability; adjacent/weather duplicate observations remain dependent']),ensure_ascii=False,indent=2),encoding='utf-8');print('PASS',checks,claims)
