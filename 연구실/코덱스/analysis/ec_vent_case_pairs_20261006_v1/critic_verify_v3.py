from pathlib import Path
import csv,math,json
from collections import defaultdict
H=Path(__file__).resolve().parent;ROOT=H.parents[3];S=H.parent/'ec_vent_gap_20261006_v1';O=H/'results_v4'
def read(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def mean(v):
    v=[x for x in v if math.isfinite(x)];return math.fsum(v)/len(v)
def sd(v):
    m=mean(v);return math.sqrt(math.fsum((x-m)**2 for x in v)/(len(v)-1)) or 1
def same(a,b):assert math.isclose(a,b,rel_tol=1e-11,abs_tol=1e-11),(a,b)
d=read(S/'results_v2/ordinary_days.csv');days={(r['farm'],int(r['day'])):r for r in d};rp=read(S/'results_v2/row_predictions.csv');ids={r['row_id'] for r in rp};p0={(r['farm'],int(r['day'])):r for r in rp if r['hour']=='0'}
cols=['in_temp','in_hum','in_co2','act_vent','act_thermal','act_shade','act_heating','act_circfan','act_co2','act_fog'];scale={c:sd([float(r[c+'_mean']) for r in d if math.isfinite(float(r[c+'_mean']))]) for c in cols}
raw=defaultdict(list)
for r in read(ROOT/'공용/대회자료/정형데이터/참가자_배포/train_X.csv'):
    if r['row_id'] in ids:
        key=(r['row_id'][:3],int(r['row_id'][4:7]));r['hour']=int(r['row_id'][8:10])
        for c in cols:r[c]=float(r[c]) if r[c] else float('nan')
        raw[key].append(r)
def dist(a,b):return math.sqrt(mean([((a[c]-b[c])/scale[c])**2 for c in cols]))
fixed=read(O/'fixed_scale_prefix_similarity.csv')
for r in fixed:
    h=int(r['hour']);a={c:mean([v[c] for v in raw[(r['farm'],int(r['failed_day']))] if v['hour']<=h]) for c in cols};b={c:mean([v[c] for v in raw[(r['farm'],int(r['success_day']))] if v['hour']<=h]) for c in cols};same(dist(a,b),float(r['distance_same_fixed_scale']))
h0={k:next(r for r in q if r['hour']==0) for k,q in raw.items()}
fails={k:r for k,r in days.items() if r['closed']=='True' and float(r['act_circfan_mean'])<10 and abs(float(r['bias']))>=.2};expected={}
for key,f in fails.items():
    q=[]
    for k,s in days.items():
        if k[0]==key[0] and s['phase']==f['phase'] and s['closed']=='True' and float(s['act_circfan_mean'])<10 and float(s['mse'])<=.01 and abs(float(p0[k]['baseline'])-float(p0[k]['y']))<=.1:q.append((dist(h0[key],h0[k]),k))
    expected[key]=sorted(q)[:3]
analog=read(O/'h0_success_analogies.csv')
for r in analog:
    k=(r['farm'],int(r['failed_day']));s=(r['farm'],int(r['success_day']));distance,key=expected[k][int(r['rank'])-1];assert key==s;same(distance,float(r['distance']))
    for side,target in [('failed',k),('success',s)]:
        same(float(r[side+'_y0']),float(p0[target]['y']));same(float(r[side+'_p0']),float(p0[target]['baseline']));same(float(r[side+'_day_rmse']),math.sqrt(float(days[target]['mse'])))
        for c in cols:same(float(r[side+'_'+c]),h0[target][c])
assert len(analog)==sum(len(q) for q in expected.values())==28
same23=[]
old=read(H/'results_v1/pairs.csv')
for r in fixed:
    if r['hour']=='23':
        t=next(t for t in old if t['farm']==r['farm'] and t['failed_day']==r['failed_day'] and t['success_day']==r['success_day'] and t['rank']=='1');same(float(r['distance_same_fixed_scale']),float(t['distance']));same23.append(r)
out=dict(status='PASS',fixed_distance_rows=len(fixed),h0_analogies=len(analog),severe_daily_targets=len(fails),checks=['all48 common-scale prefix distances','all12 h23 distances equal original daily distances','all28 h0 candidates qualifications/input values/y/p and top3 rankings'],warnings=['h0 failure label uses severe daily bias, not h0 error','h0 successful pool uses future full-day label and h0 outcome','new h0 ranking has no temp/humidity caliper, samefold or y similarity control'])
with (H/'critic_verification_v3.json').open('x',encoding='utf-8') as f:json.dump(out,f,ensure_ascii=False,indent=2)
print(json.dumps(out,ensure_ascii=False,indent=2))
