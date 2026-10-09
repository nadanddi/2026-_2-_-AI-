from pathlib import Path
import csv,math,json,hashlib,random,sys
H=Path(__file__).resolve().parent;ROOT=H.parents[3];CACHE=ROOT/'집/클로드/research/local/ct1_ckpt'
SEEDS=(2121,4343,6565);ARMS=('REF','THS')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
files=sorted(CACHE.glob('*.csv'));assert len(files)==76
pins={str(p.relative_to(ROOT)):sha(p) for p in files}
pins['집/클로드/research/ct1_thermal_schedule_features_v1.py']=sha(ROOT/'집/클로드/research/ct1_thermal_schedule_features_v1.py')
reg=H/'registration_v1.json'
if not reg.exists():
 reg.write_text(json.dumps(dict(inputs=pins,plan_sha=sha(H/'PLAN_v1.md'),script_sha=sha(Path(__file__))),ensure_ascii=False,indent=2),encoding='utf8')
else:
 assert json.loads(reg.read_text(encoding='utf8'))['inputs']==pins
rows=[]
for p in files:
 with p.open(encoding='utf8',newline='') as f:rr=list(csv.DictReader(f))
 assert len({r['row_id'] for r in rr})==len(rr)
 days={}
 for r in rr:
  r['day']=int(r['day']);r['hour']=int(r['hour']);r['y']=float(r['sub_ec']);r['fold']=int(r['validation_fold'])
  assert math.isfinite(r['y'])
  days.setdefault((r['farm'],r['day']),[]).append(r['hour'])
  for a in ARMS:
   ps=[]
   for s in SEEDS:
    vals=[float(r[f'{a}_{k}_{s}']) for k in ['et','lgb','mlp']];assert all(math.isfinite(x) for x in vals)
    pred=min(float(r['hi']),max(float(r['lo']),math.fsum(w*x for w,x in zip((.6,.3,.1),vals))))
    r[f'{a}_{s}']=pred;ps.append(pred)
   r[a]=math.fsum(ps)/3
 rows.extend(rr)
 assert all(sorted(hs)==list(range(24)) for hs in days.values())
for v in ['DIAG10','EL1','P2LOO']:
 rr=[r for r in rows if r['validator']==v];assert len({r['row_id'] for r in rr})==len(rr)
def stats(rr):
 assert rr
 occ={}
 for r in rr:occ.setdefault((r['farm'],r['day']),set()).add(r['fold'])
 out={'rows':len(rr),'days':len(occ),'fold_occurrences_per_day_range':[min(map(len,occ.values())),max(map(len,occ.values()))]}
 for a in ARMS:
  err=[r[a]-r['y'] for r in rr];score=math.sqrt(math.fsum(e*e for e in err)/len(rr))
  other=math.sqrt(sum((r[a]-r['y'])**2 for r in rr)/len(rr));assert abs(score-other)<1e-12
  out[a]={'rmse':score,'bias':math.fsum(err)/len(rr),'seed_rmse':[math.sqrt(math.fsum((r[f'{a}_{s}']-r['y'])**2 for r in rr)/len(rr)) for s in SEEDS]}
 out['pct']=100*(out['THS']['rmse']/out['REF']['rmse']-1)
 out['seed_better']=sum(b<a for a,b in zip(out['REF']['seed_rmse'],out['THS']['seed_rmse']))
 return out
def bootstrap(rr,seed):
 groups={}
 for r in rr:
  b=groups.setdefault((r['farm'],r['day']//5),[0.,0.,0]);b[0]+=(r['REF']-r['y'])**2;b[1]+=(r['THS']-r['y'])**2;b[2]+=1
 keys=sorted(groups);farmidx={f:[i for i,k in enumerate(keys) if k[0]==f] for f in ['F13','F47']}
 rng=random.Random(seed);deltas=[]
 for _ in range(20000):
  ix=[rng.choice(ids) for ids in farmidx.values() for __ in ids];a=math.fsum(groups[keys[i]][0] for i in ix);b=math.fsum(groups[keys[i]][1] for i in ix);n=sum(groups[keys[i]][2] for i in ix)
  deltas.append(math.sqrt(b/n)-math.sqrt(a/n))
 deltas.sort();return dict(blocks=len(keys),p_worse=sum(x>=0 for x in deltas)/len(deltas),delta_rmse_ci95=[deltas[499],deltas[19499]])
stage=sys.argv[1];assert stage in ('midpoint','final')
D=[r for r in rows if r['validator']=='DIAG10'];whole=stats(D)
assert abs(whole['REF']['rmse']-.20459306704101943)<1e-12 and abs(whole['THS']['rmse']-.19286410493483824)<1e-12
D2=[r for r in D if r['day']>=179];daily={}
for r in D2:daily.setdefault((r['farm'],r['day']),[]).append(r)
gain={k:math.fsum((r['REF']-r['y'])**2-(r['THS']-r['y'])**2 for r in rr) for k,rr in daily.items()}
top=sorted(gain,key=lambda k:(-gain[k],k))[:5];net=math.fsum(gain.values())
primary={};cells=[]
for name,fn,seed in [('early',lambda r:r['hour']<=8,202610091),('late',lambda r:r['hour']>=9,202610092)]:
 rr=[r for r in D2 if fn(r)];z=stats(rr);z['bootstrap']=bootstrap(rr,seed);z['farms']={f:stats([r for r in rr if r['farm']==f]) for f in ['F13','F47']};z['without_top5']=stats([r for r in rr if (r['farm'],r['day']) not in top]);primary[name]=z
 cells.extend(x['seed_better']==3 for x in z['farms'].values())
summary=dict(stage=stage,baseline=whole,primary=primary,diagnostic_supported=all(cells) and all(z['bootstrap']['p_worse']<.0125 for z in primary.values()),top5=[{'farm':f,'day':d,'gain_sse':gain[(f,d)]} for f,d in top],day_gain_counts={'positive':sum(x>0 for x in gain.values()),'negative':sum(x<0 for x in gain.values()),'zero':sum(x==0 for x in gain.values())},net_gain_sse=net,top5_net_gain_share=(math.fsum(gain[k] for k in top)/net if net>0 else None),rows_total=len(rows),adopted=False)
if stage=='final':
 summary['segments']=[]
 summary['target_groups']=[]
 for v in ['DIAG10','A','B','EL1','P2LOO']:
  vr=[r for r in rows if r['validator']==v];dm={}
  for r in vr:dm.setdefault((r['farm'],r['day'],r['fold']),[]).append(r['y'])
  high={k for k,yy in dm.items() if math.fsum(yy)/len(yy)>=1.2}
  for label in ['high','normal']:
   rr=[r for r in vr if ((r['farm'],r['day'],r['fold']) in high)==(label=='high')]
   if rr:summary['target_groups'].append(dict(validator=v,target_group=label,**stats(rr)))
 for v in ['DIAG10','A','B','EL1','P2LOO']:
  for f in ['F13','F47']:
   for p2 in [False,True]:
    for hours,fn in [('early',lambda r:r['hour']<=8),('late',lambda r:r['hour']>=9)]:
     rr=[r for r in rows if r['validator']==v and r['farm']==f and (r['day']>=179)==p2 and fn(r)]
     if rr:summary['segments'].append(dict(validator=v,farm=f,pass2=p2,hours=hours,**stats(rr)))
 summary['daily_gain']=[dict(farm=f,day=d,gain_sse=gain[(f,d)],**stats(rr)) for (f,d),rr in sorted(daily.items())]
p=H/f'{stage}_v1.json';assert not p.exists();p.write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps({k:v for k,v in summary.items() if k not in ['segments','daily_gain']},ensure_ascii=True))
