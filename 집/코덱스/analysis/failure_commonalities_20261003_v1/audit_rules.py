from analyze import *
import csv,math
from patterns_v2 import RULES
def main():
 d=pd.read_csv(H/'final_daily_cases_v2.csv',float_precision='round_trip');fc=pd.read_csv(H/'fixed_rule_counts_v3.csv',float_precision='round_trip');di={};checks=0
 with open(Path(env.DATA)/'train_X.csv',encoding='utf-8-sig',newline='') as f:
  for r in csv.DictReader(f):
   if r['row_id'][:3] not in ['F13','F47']:continue
   key=(r['row_id'][:3],int(r['row_id'][4:7]));di.setdefault(key,[]).append(r)
 def number(v):return float(v) if v else float('nan')
 def av(v):v=[x for x in v if math.isfinite(x)];return math.fsum(v)/len(v) if v else float('nan')
 prev={};derived={}
 for k,rs in sorted(di.items()):
  rs.sort(key=lambda r:r['row_id']);a={c:[number(r[c]) for r in rs] for c in RAW};z={}
  for c in ['in_temp','in_hum','in_co2','out_temp','out_rad','act_vent','act_shade','act_thermal','act_heating','act_circfan','act_co2']:
   v=a[c];z[c+'__mean']=av(v);z[c+'__h0']=v[0];z[c+'__night']=av(v[:7]);z[c+'__evening']=av(v[17:]);z[c+'__daynight']=av(v[9:17])-av(v[:7]);good=[x for x in v if math.isfinite(x)];z[c+'__zero_fraction']=sum(x==0 for x in good)/len(good) if good else float('nan')
   dif=[abs(b-a) for a,b in zip(v,v[1:]) if math.isfinite(a) and math.isfinite(b)];z[c+'__jump_max']=max(dif) if dif else float('nan')
  if k[0] in prev:pk=prev[k[0]];z['previous_record_gap']=k[1]-pk[1];z['out_rad__mean__previous']=derived[pk]['out_rad__mean']
  else:z['previous_record_gap']=z['out_rad__mean__previous']=float('nan')
  aa=a['in_temp'];bb=a['out_temp'];mu=av(aa);mb=av(bb);valid=[(x,y) for x,y in zip(aa,bb) if math.isfinite(x) and math.isfinite(y)]
  if len(valid)>=4:
   mx=av([x for x,y in valid]);my=av([y for x,y in valid]);den=math.sqrt(math.fsum((x-mx)**2 for x,y in valid)*math.fsum((y-my)**2 for x,y in valid));z['in_temp__corr__out_temp']=math.fsum((x-mx)*(y-my) for x,y in valid)/den if den else float('nan')
  else:z['in_temp__corr__out_temp']=float('nan')
  derived[k]=z;prev[k[0]]=k
 for r in fc.itertuples():
  if r.feature in ['exact_weather_twins','level_sse_fraction','all_under_fraction','all_over_fraction']:continue
  q=d[d.target==r.target];q=q if r.group=='all' else q[q[r.group]];vals=[derived[(z.farm,z.day)][r.feature] for z in q.itertuples()];good=[v for v in vals if math.isfinite(v)]
  def ok(v):return v<=r.cut if r.operator=='<=' else v>=r.cut if r.operator=='>=' else v>r.cut
  yes=sum(ok(v) for v in good);assert yes==r.yes,(r.rule,r.target,r.group,yes,r.yes);assert len(good)==r.available;checks+=2
 save('raw_rule_audit.json',dict(status='PASS',checks=checks,method='Raw CSV dictionaries, math.fsum, explicit phase means/jumps/zero counts/Pearson correlation; no exploratory feature helpers'))
 print('PASS',checks)
if __name__=='__main__':main()
