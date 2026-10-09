from pathlib import Path
import csv,json,math,hashlib
H=Path(__file__).resolve().parent;R=H.parents[3]
channels=['act_vent','act_circfan','act_heating','act_shade','act_co2','act_fog','act_thermal'];cuts=[8,15,23]
reg=json.loads((H/'registration_v1.json').read_text(encoding='utf8'));truth={};source={}
for rel,sha in reg['inputs'].items():
 p=R/rel;assert hashlib.sha256(p.read_bytes()).hexdigest()==sha
 if 'DIAG10_' in p.name:
  for q in csv.DictReader(p.open(encoding='utf8',newline='')):
   assert q['row_id'] not in truth
   preds=[]
   for s in [2121,4343,6565]:
    v=.6*float(q[f'REF_et_{s}'])+.3*float(q[f'REF_lgb_{s}'])+.1*float(q[f'REF_mlp_{s}']);preds.append(max(float(q['lo']),min(float(q['hi']),v)))
   truth[q['row_id']]=(float(q['sub_ec']),sum(preds)/3)
 xp=next(R/k for k in reg['inputs'] if k.endswith('train_X.csv'))
for q in csv.DictReader(xp.open(encoding='utf8',newline='')):
 if q['row_id'] in truth:source[q['row_id']]=q
assert len(source)==len(truth)==8640
features=list(csv.DictReader((H/'daily_features_v1.csv').open(encoding='utf8',newline='')));assert len(features)==360
maxdiff=0.;checked=0
for z in features:
 ids=[f"{z['farm']}_{int(z['day']):03d}_{h:02d}" for h in range(24)]
 # IDs can use different zero-padding; resolve by parsed farm/day/hour.
 matching={int(k.rsplit('_',1)[1]):k for k in truth if k.split('_')[0]==z['farm'] and int(k.split('_')[1])==int(z['day'])};assert sorted(matching)==list(range(24));ids=[matching[h] for h in range(24)]
 ec=sum(truth[rid][0] for rid in ids)/24;pred=sum(truth[rid][1] for rid in ids)/24
 for key,v in [('ec',ec),('pred',pred),('residual',ec-pred)]:
  delta=abs(v-float(z[key]));assert delta<1e-12;maxdiff=max(maxdiff,delta);checked+=1
 for c in channels:
  vv=[float(source[rid][c]) for rid in ids]
  for h in cuts:
   a=vv[:h+1];d={'mean':sum(a)/len(a),'active':sum(v>.1 for v in a),'full':sum(v>=99.9 for v in a),'switches':sum(abs(v-u)>.1 for u,v in zip(a,a[1:]))}
   if h>=15:d['daynight']=sum(vv[9:16])/7-sum(vv[:9])/9
   for key,v in d.items():
    delta=abs(v-float(z[f'{c}__h{h}__{key}']));assert delta<1e-10;maxdiff=max(maxdiff,delta);checked+=1
out={'status':'INDEPENDENT_INPUT_FEATURE_TARGET_PASS','rows':8640,'days':360,'feature_count':98,'cells_rechecked':checked,'max_abs_diff':maxdiff,'source':'registered public OOF and train_X only','official_train_y_loaded':False,'test_values_loaded':False}
p=H/'critic_midpoint_recheck_v1.json';assert not p.exists();p.write_text(json.dumps(out,indent=2),encoding='utf8');print(json.dumps(out))
