from pathlib import Path
import csv,math,json,sys
from collections import defaultdict
H=Path(__file__).resolve().parent;ROOT=H.parents[3];S=H.parent/'ec_vent_gap_20261006_v1';O=H/'results_v2'
def read(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def mean(v):
    v=[x for x in v if math.isfinite(x)];return math.fsum(v)/len(v) if v else float('nan')
def sd(v):
    v=[x for x in v if math.isfinite(x)];m=mean(v);s=math.sqrt(math.fsum((x-m)**2 for x in v)/(len(v)-1));return s or 1
def median(v):
    v=sorted(x for x in v if math.isfinite(x));n=len(v);return v[n//2] if n%2 else mean(v[n//2-1:n//2+1])
def same(a,b):assert math.isclose(a,b,abs_tol=1e-11,rel_tol=1e-11),(a,b)
def number(x):return float(x) if x else float('nan')
all_d=read(S/'results_v2/days.csv');days={(r['farm'],int(r['day'])):r for r in all_d}
rp=read(S/'results_v2/row_predictions.csv');pred={r['row_id']:r for r in rp}
rawpath=ROOT/'공용/대회자료/정형데이터/참가자_배포/train_X.csv'
cols=['in_temp','in_hum','in_co2','act_vent','act_thermal','act_shade','act_heating','act_circfan','act_co2','act_fog']
raw=defaultdict(list)
for r in read(rawpath):
    key=(r['row_id'][:3],int(r['row_id'][4:7]))
    if key in days:
        r['hour']=int(r['row_id'][8:10])
        for c in cols:r[c]=number(r[c])
        raw[key].append(r)
assert len(raw)==360 and sum(len(q) for q in raw.values())==8640
profiles={}
for h in [0,6,12,23]:profiles[h]={key:{c:mean([r[c] for r in q if r['hour']<=h]) for c in cols} for key,q in raw.items()}
rows=read(O/'pair_prefix_all.csv')
for r in rows:
    key=(r['farm'],int(r['day']));h=int(r['hour']);q=raw[key];current=next(u for u in q if u['hour']==h);prefix=[u for u in q if u['hour']<=h]
    cp=pred[current['row_id']]
    same(float(r['true_y']),float(cp['y']));same(float(r['prediction']),float(cp['baseline']))
    for label,name in [('prefix_y','y'),('prefix_prediction','baseline')]:same(float(r[label]),mean([float(pred[u['row_id']][name]) for u in prefix]))
    for c in cols:same(float(r[c]),current[c]);same(float(r[c+'_prefix']),profiles[h][key][c])
    for c in ['et','lgb','mlp','pfn']:same(float(r[c]),float(cp[c+'_smooth']));same(float(r[c+'_prefix']),mean([float(pred[u['row_id']][c+'_smooth']) for u in prefix]))
distances=read(O/'prefix_similarity.csv')
for r in distances:
    h=int(r['hour']);a=profiles[h][(r['farm'],int(r['failed_day']))];b=profiles[h][(r['farm'],int(r['success_day']))]
    scales={c:sd([u[c] for u in profiles[h].values()]) for c in cols};distance=math.sqrt(mean([((a[c]-b[c])/scales[c])**2 for c in cols]));same(distance,float(r['prefix_distance']))
sys.dont_write_bytecode=True;sys.path.insert(0,str(ROOT/'집/클로드/research'));import env
import numpy as np
trainids={}
for fold in range(10):
    with np.load(ROOT/'연구실/코덱스/local/ec_vent_gap_20261006_v1/inputs/components'/f'DIAG10_{fold}_r3_7.npz',allow_pickle=False) as z:trainids[fold]={(i[:3],int(i[4:7])) for i in z['train_row_id'].astype(str)}
for r in read(O/'fold_training_context.csv'):
    fold=int(r['fold']);keys=[k for k in trainids[fold] if k[0]==r['farm']];q=[days[k] for k in keys]
    same(len(q),int(r['train_days']));same(sum(float(u['ymean'])>=1 for u in q),int(r['train_high']));same(mean([float(u['ymean']) for u in q]),float(r['train_mean_y']))
for r in read(O/'prefix_training_neighbors.csv'):
    fold=int(r['fold']);h=int(r['hour']);key=(r['farm'],int(r['day']));keys=sorted(k for k in trainids[fold] if k[0]==r['farm']);assert key not in keys
    vals={k:profiles[h][k] for k in keys};scales={c:sd([v[c] for v in vals.values()]) for c in cols};med={c:median([v[c] for v in vals.values()]) for c in cols}
    def value(v,c):return v[c] if math.isfinite(v[c]) else med[c]
    query=profiles[h][key];ranks=sorted((math.sqrt(mean([((value(v,c)-value(query,c))/scales[c])**2 for c in cols])),k) for k,v in vals.items())
    top=ranks[:5];same(top[0][0],float(r['nearest_distance']));same(mean([float(days[k]['ymean']) for _,k in top]),float(r['neighbor_y']));same(mean([float(float(days[k]['ymean'])>=1) for _,k in top]),float(r['neighbor_high']))
    assert [k[1] for _,k in top]==list(map(int,r['neighbor_days'].split(';')))
out=dict(status='PASS',score_trace_rows=len(rows),targets=len({(r['farm'],r['day']) for r in rows}),pair_prefix_distances=len(distances),checks=['72 current/prefix predictions and inputs','360-public-day hour-specific prefix scales/distances','samefarm actual-fold training contexts','all36 nearest5 prefix neighbors and daily-y summaries'],scale_warning='day distance: ordinary329 full-day std; prefix distance: public360 hour-specific std; numerical ratios across these metrics are not comparable',neighbor_label_warning='neighbor_y/high are full-day target summaries, not targets at query hour')
with (H/'critic_verification_v2.json').open('x',encoding='utf-8') as f:json.dump(out,f,ensure_ascii=False,indent=2)
print(json.dumps(out,ensure_ascii=False,indent=2))
