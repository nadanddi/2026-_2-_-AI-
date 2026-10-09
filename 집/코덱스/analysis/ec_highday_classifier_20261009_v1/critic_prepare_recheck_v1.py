from pathlib import Path
import csv,json,math,hashlib
H=Path(__file__).resolve().parent;R=H.parents[3];L=R/'집/코덱스/local'/H.name
reg=json.loads((H/'registration_v1.json').read_text(encoding='utf8'));raw=list(csv.DictReader((L/'raw_public_v1.csv').open(encoding='utf8')));ff={q['row_id']:q for q in csv.DictReader((L/'features_public_v1.csv').open(encoding='utf8'))};meta={q['row_id']:q for q in csv.DictReader((L/'metadata_public_v1.csv').open(encoding='utf8'))}
assert len(raw)==len(ff)==len(meta)==8640
labels={}
for p in (R/'집/클로드/research/local/ct1_ckpt').glob('DIAG10_*.csv'):
 for q in csv.DictReader(p.open(encoding='utf8')):assert q['row_id'] not in labels;labels[q['row_id']]=float(q['sub_ec'])
groups={}
for q in raw:
 farm,day,h=q['row_id'].split('_');groups.setdefault((farm,int(day)),[]).append(q)
assert len(groups)==360
rawcols=['out_temp','out_hum','out_rad','out_wspd','in_temp','in_hum','in_co2','act_vent','act_shade','act_thermal','act_heating','act_circfan','act_co2','act_fog']
def number(x):return float(x) if x else float('nan')
def run(v,predicate):
 n=0
 for t in v:
  if not math.isfinite(t):n=None
  elif predicate(t):n=n+1 if n is not None else None
  else:n=0
 return float('nan') if n is None else n
maxgap=0.;cells=0;highdays=0
for (farm,day),qs in groups.items():
 qs.sort(key=lambda q:int(q['row_id'].split('_')[2]));assert [int(q['row_id'].split('_')[2]) for q in qs]==list(range(24))
 daily=math.fsum(labels[q['row_id']] for q in qs)/24;highdays+=daily>=1.2
 for j,q in enumerate(qs):
  rid=q['row_id'];z={};vals={c:[number(t[c]) for t in qs[:j+1]] for c in rawcols}
  assert abs(float(meta[rid]['daily_ec'])-daily)<1e-12 and int(meta[rid]['high'])==int(daily>=1.2)
  for c,v in vals.items():
   observed=[t for t in v if math.isfinite(t)];z[c+'__current']=v[-1];z[c+'__h0']=v[0];z[c+'__mean']=sum(observed)/len(observed) if observed else math.nan;z[c+'__zero_share']=sum(t==0 for t in observed)/len(observed) if observed else math.nan
  vent=vals['act_vent'];heat=vals['act_heating'];co2=vals['act_co2'];thermal=vals['act_thermal'];shade=vals['act_shade'];ventok=all(map(math.isfinite,vent));curtainok=all(map(math.isfinite,thermal+shade));opened=[i for i,t in enumerate(vent) if t>0];changes=[i for i in range(1,j+1) if ((thermal[i]>0)!=(thermal[i-1]>0)) or ((shade[i]>0)!=(shade[i-1]>0))] if curtainok else []
  switch=lambda v:sum((a>0)!=(b>0) for a,b in zip(v,v[1:])) if all(map(math.isfinite,v)) else math.nan
  z.update(hour=j,hr_sin=math.sin(j/24*2*math.pi),hr_cos=math.cos(j/24*2*math.pi),farm47=int(farm=='F47'),seal_run=run(vent,lambda t:t==0),vent_open_hours=sum(t>0 for t in vent) if ventok else math.nan,first_open_hour=opened[0] if opened and all(map(math.isfinite,vent[:opened[0]+1])) else 24 if ventok else math.nan,thermal_switches=switch(thermal),shade_switches=switch(shade),since_curtain_change=j-changes[-1] if changes else j+1 if curtainok else math.nan,heat_run=run(heat,lambda t:t>0),co2_hours=sum(t>0 for t in co2) if all(map(math.isfinite,co2)) else math.nan,vent_max=max(t for t in vent if math.isfinite(t)) if any(map(math.isfinite,vent)) else math.nan,daynight_available=int(j>=9))
  for c in ['act_circfan','act_shade','act_thermal']:
   v=vals[c][:min(j,15)+1];z[c+'__daynight']=sum(v[9:])/len(v[9:])-sum(v[:9])/9 if j>=9 and all(map(math.isfinite,v)) else math.nan
  assert set(z)==set(reg['feature_columns']) and len(z)==73
  for k,v in z.items():
   w=number(ff[rid][k]);assert math.isnan(v)==math.isnan(w),(rid,k,v,w)
   if math.isfinite(v):gap=abs(v-w);assert gap<1e-10,(rid,k,v,w);maxgap=max(maxgap,gap)
   cells+=1
pub=set(groups);lock={tuple(d) for d in reg['locked_days']};rowdays=[(q['farm'],int(q['day'])) for q in meta.values()]
# Indices refer to original metadata file order, retained by csv.DictReader.
for f in reg['folds']:
 q={tuple(d) for d in f['query_days']};forbidden={(farm,d+j) for farm,d in q for j in [-1,0,1]}|{('F47' if farm=='F13' else 'F13',d+j) for farm,d in q for j in range(-3,4)}|{(farm,d+j) for farm,d in lock for j in [-1,0,1]}
 assert set(map(tuple,f['forbidden_days']))==forbidden
 assert f['train_indices']==[i for i,d in enumerate(rowdays) if d not in forbidden]
 assert f['query_indices']==[i for i,d in enumerate(rowdays) if d in q]
 assert set(map(tuple,f['train_days']))==pub-forbidden
out={'status':'INDEPENDENT_FEATURE_LABEL_PURGE_PASS','feature_columns':73,'cells':cells,'max_abs_gap':maxgap,'public_days':360,'high_days':highdays,'folds':len(reg['folds']),'official_extra40_y_loaded':False}
p=H/'critic_prepare_recheck_v1.json';assert not p.exists();p.write_text(json.dumps(out,indent=2),encoding='utf8');print(json.dumps(out))
