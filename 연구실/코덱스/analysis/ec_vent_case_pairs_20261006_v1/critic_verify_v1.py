"""Independent case-pair arithmetic using csv/math.fsum, no training."""
from pathlib import Path
import csv,json,math
from collections import defaultdict
H=Path(__file__).resolve().parent;S=H.parent/'ec_vent_gap_20261006_v1';O=H/'results_v1'
def read(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def avg(v):return math.fsum(v)/len(v)
def same(a,b):assert math.isclose(a,b,rel_tol=1e-11,abs_tol=1e-11),(a,b)
d=read(S/'results_v2/ordinary_days.csv');rp=read(S/'results_v2/row_predictions.csv')
cols=['in_temp_mean','in_hum_mean','in_co2_mean','act_vent_mean','act_thermal_mean','act_shade_mean','act_heating_mean','act_circfan_mean','act_co2_mean','act_fog_mean']
for r in d:
    for k in cols+['ymean','pmean','bias','mse','day','phase','fold','ventzero']:r[k]=float(r[k])
    r['closed']=r['closed']=='True';r['fanlow']=r['act_circfan_mean']<10
    r['success']=r['mse']<=.1**2;r['severe']=abs(r['bias'])>=.2
index={(r['farm'],int(r['day'])):r for r in d}
def median(v):
    s=sorted(v);n=len(s);return s[n//2] if n%2 else (s[n//2-1]+s[n//2])/2
scale={};med={}
for col in cols:
    vals=[r[col] for r in d if math.isfinite(r[col])];m=avg(vals)
    scale[col]=math.sqrt(math.fsum((v-m)**2 for v in vals)/(len(vals)-1)) or 1
    med[col]=median(vals)
def value(r,col):return r[col] if math.isfinite(r[col]) else med[col]
def dist(a,b):return math.sqrt(avg([((value(a,k)-value(b,k))/scale[k])**2 for k in cols]))
summary=[]
for t in read(O/'group_success.csv'):
    name=t['group'];q=[r for r in d if name=='ordinary' or name=='closed' and r['closed'] or name=='closed_fanlow' and r['closed'] and r['fanlow'] or name=='closed_fanhigh' and r['closed'] and not r['fanlow'] or name=='rest' and not r['closed']]
    result=dict(group=name,days=len(q),success=sum(r['success'] for r in q),severe=sum(r['severe'] for r in q),rmse=math.sqrt(avg([r['mse'] for r in q])))
    for k in ['days','success','severe','rmse']:same(result[k],float(t[k]))
    summary.append(result)
pairs=read(O/'pairs.csv');coverage=read(O/'pair_coverage.csv');expected={}
for t in coverage:
    f=index[(t['farm'],int(t['day']))];assert f['closed'] and f['fanlow'] and f['severe']
    q=[s for s in d if s['farm']==f['farm'] and s['phase']==f['phase'] and s['closed'] and s['fanlow'] and s['success'] and abs(s['in_temp_mean']-f['in_temp_mean'])<=2 and abs(s['in_hum_mean']-f['in_hum_mean'])<=10]
    if t['mode']=='input_y':q=[s for s in q if abs(s['ymean']-f['ymean'])<=.1]
    if t['mode']=='same_fold':q=[s for s in q if s['fold']==f['fold']]
    same(len(q),int(t['success_candidates']))
    expected[(f['farm'],int(f['day']),t['mode'])]=sorted([(dist(f,s),int(s['day'])) for s in q])[:3]
for t in pairs:
    f=index[(t['farm'],int(t['failed_day']))];s=index[(t['farm'],int(t['success_day']))]
    q=expected[(t['farm'],int(t['failed_day']),t['mode'])]
    v,day=q[int(t['rank'])-1];assert day==int(s['day']);same(v,float(t['distance']))
    for label,r in [('failed',f),('success',s)]:
        for k,source in [('y','ymean'),('p','pmean'),('bias','bias'),('fold','fold')]:same(r[source],float(t[label+'_'+k]))
assert len(pairs)==sum(len(v) for v in expected.values())
byday=defaultdict(list)
for r in rp:byday[(r['farm'],int(r['day']))].append(r)
weights={'et':.48,'lgb':.24,'mlp':.08,'pfn':.20};maxgap=0
for t in read(O/'day_member_scores.csv'):
    q=byday[(t['farm'],int(t['day']))];y=avg([float(r['y']) for r in q]);p=avg([float(r['baseline']) for r in q]);terms=[]
    for n,w in weights.items():
        val=avg([float(r[n+'_smooth']) for r in q]);same(val,float(t[n]));term=w*(val-y);terms.append(term);same(term,float(t[n+'_weighted_bias']))
    gap=abs(math.fsum(terms)-(p-y));maxgap=max(gap,maxgap);same(math.fsum(terms),p-y)
    same(y,float(t['ymean']));same(p,float(t['pmean']));same(p-y,float(t['bias']))
out=dict(status='PASS',method='csv/math.fsum; prior independently verified public OOF-derived rows',checks=['group success/severe counts','all candidate coverage','all top3 distance rankings and calipers','input_y and same_fold separation','all329day weighted member bias identities'],groups=summary,pairs=len(pairs),coverage_rows=len(coverage),max_bias_identity_error=maxgap,matching_scale='sample ddof1 std over ordinary329 days; global-median missing fill')
with (H/'critic_verification_v1.json').open('x',encoding='utf-8') as f:json.dump(out,f,ensure_ascii=False,indent=2)
print(json.dumps(out,ensure_ascii=False,indent=2))
