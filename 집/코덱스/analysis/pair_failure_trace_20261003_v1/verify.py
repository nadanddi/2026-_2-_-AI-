import csv,json,math
from diagnose import *
def mean(a):return math.fsum(a)/len(a)
def close(a,b):assert math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-10),(a,b)
def main():
 result=json.loads((H/'result.json').read_text(encoding='utf-8'));cov=json.loads((H/'coverage.json').read_text(encoding='utf-8'))
 source=R/'공용/대회자료/정형데이터/참가자_배포'
 with (source/'train_X.csv').open(encoding='utf-8-sig',newline='') as f:x={q['row_id']:q for q in csv.DictReader(f) if q['row_id'][:3] in ['F13','F47']}
 # Read only the temperature label field. EC is never consumed.
 with (source/'train_y.csv').open(encoding='utf-8-sig',newline='') as f:y={q['row_id']:float(q['sub_temp']) for q in csv.DictReader(f) if q['row_id'] in x}
 with (R/'집/코덱스/local/temp_tk_season_20261003_v1/TK1_predictions.csv').open(encoding='utf-8-sig',newline='') as f:p={(q['row_id'],q['member']):float(q['prediction']) for q in csv.DictReader(f) if q['validator']=='DIAG10' and q['base_seed']=='7' and q['context']=='1-8'}
 n=0
 for s in result['stats']:
  ids=[f"{s['farm']}_{s['day']:03d}_{h:02d}" for h in range(24)]
  if 'segment' in s:
   ids=[i for i in ids if (int(i[-2:])<=6 if s['segment']=='밤0–6' else 7<=int(i[-2:])<=16 if s['segment']=='낮7–16' else int(i[-2:])>=17)]
  air=[float(x[i]['in_temp']) for i in ids];truth=[y[i] for i in ids];pred=[p[i,'W30G'] for i in ids];e=[v-u for v,u in zip(pred,truth)];sse=math.fsum(v*v for v in e);bias=mean(e)
  for c,v in dict(air=mean(air),truth=mean(truth),pred=mean(pred),bias=bias,rmse=math.sqrt(sse/len(ids))).items():close(s[c],v);n+=1
  if 'segment' not in s:
   close(s['level_fraction'],24*bias*bias/sse);close(s['centered_rmse'],math.sqrt(mean([(v-bias)**2 for v in e])))
   oracle=[];above=0
   for i in ids:
    experts=[p[i,m] for m in ['BASE','CODEX','PFN']];above+=y[i]>max(experts);oracle.append(min(max(y[i],min(experts)),max(experts))-y[i])
   assert above==s['above_experts_hours'];assert sum(v<0 for v in e)==s['under_hours'];close(s['oracle_rmse'],math.sqrt(mean([v*v for v in oracle])));close(s['oracle_remaining_sse'],math.fsum(v*v for v in oracle)/sse);n+=6
 keys=sorted({(s[:3],int(s[4:7])) for s in np.load(R/'집/코덱스/local/statistical_experiments_20261003_v1/T_DIAG10_5_cpu.npz')['outer_train_id']})
 days=sorted({(i[:3],int(i[4:7])) for i in x});features={};gaps={}
 for k in days:
  ids=[f'{k[0]}_{k[1]:03d}_{h:02d}' for h in range(24)]
  features[k]=[mean([float(x[i][c]) for i in ids if x[i][c]!='']) for c in RAW]
  vals=[float(x[i]['in_temp']) for i in ids if x[i]['in_temp']!='']
  if len(vals)==24:gaps[k]=mean([y[i] for i in ids])-mean(vals)
 scales=[]
 for j in range(len(RAW)):
  values=[features[k][j] for k in keys];m=mean(values);sd=math.sqrt(mean([(v-m)**2 for v in values]));scales.append(sd if sd else 1)
 for f,day in CASES:
  candidates=[k for k in keys if k[0]==f and (k[1]>=179)==(day>=179)]
  distances={k:math.sqrt(mean([((features[k][j]-features[f,day][j])/scales[j])**2 for j in range(len(RAW))])) for k in candidates}
  nearest=sorted(distances,key=distances.get)[:5];record=[q for q in result['neighbors'] if q['case_farm']==f]
  assert [(q['neighbor_farm'],q['neighbor_day']) for q in record]==nearest
  for q in record:close(q['distance'],distances[q['neighbor_farm'],q['neighbor_day']]);n+=1
 for row in cov['coverage']:
  ks=[k for k in gaps if k[0]==row['farm'] and (k[1]>=179)==row['late'] and (row['scope']=='all' or ((k in keys)==(row['scope']=='train')))]
  assert len(ks)==row['n'];assert [k[1] for k in ks if gaps[k]>=2]==row['warm_days'];n+=2
 for q in result['weather_train']:
  ids=[f"{q['case_farm']}_{q['case_day']:03d}_{h:02d}" for h in range(24)];other=[f"{q['train_farm']}_{q['train_day']:03d}_{h:02d}" for h in range(24)]
  assert (q['train_farm'],q['train_day']) in keys
  assert all(x[i][c]==x[j][c] for i,j in zip(ids,other) for c in W);n+=96
 proof=dict(status='PASS',scalar_checks=n,method='raw CSV dictionaries/math.fsum; manual distance/ranking and outer train coverage',limitations=['Two selected days: no causal identification','Raw day aggregates diagnostic only; not causal prediction features','Same weather does not prove date/location identity','Oracle uses true labels and is not achievable validation score','No new training or EC labels/lock access'])
 (H/'verification.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2),encoding='utf-8');print(proof)
if __name__=='__main__':main()
