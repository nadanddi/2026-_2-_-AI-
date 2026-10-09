from pathlib import Path
import csv,json,math,random,hashlib,sys
H=Path(__file__).resolve().parent;R=H.parents[3]
seeds=[2121,4343,6565];data=[]
files=sorted((R/'집/클로드/research/local/ct1_ckpt').glob('*.csv'))
assert len(files)==76
registered=json.loads((H/'registration_v1.json').read_text(encoding='utf8'))
for file in files:
 assert hashlib.sha256(file.read_bytes()).hexdigest()==registered['inputs'][str(file.relative_to(R))]
 records=list(csv.DictReader(file.open(encoding='utf8',newline='')))
 for q in records:
  lo,hi=float(q['lo']),float(q['hi']);assert math.isfinite(lo) and math.isfinite(hi) and lo<=hi
  q['day']=int(q['day']);q['hour']=int(q['hour']);q['fold']=int(q['validation_fold']);q['truth']=float(q['sub_ec'])
  for arm in ['REF','THS']:
   ps=[]
   for s in seeds:
    p=.6*float(q[f'{arm}_et_{s}'])+.3*float(q[f'{arm}_lgb_{s}'])+.1*float(q[f'{arm}_mlp_{s}'])
    ps.append(max(lo,min(hi,p)))
   q[arm]=sum(ps)/3;q[arm+'_seeds']=ps
  data.append(q)
# Repeated fold records must have identical row labels and farm/day/hour keys.
identity={}
for q in data:
 key=q['row_id'];value=(q['farm'],q['day'],q['hour'],q['truth'])
 assert key not in identity or identity[key]==value
 identity[key]=value

def verify_stats(z,rr):
 assert z['rows']==len(rr) and z['days']==len({(q['farm'],q['day']) for q in rr})
 for arm in ['REF','THS']:
  rm=math.sqrt(sum((q[arm]-q['truth'])**2 for q in rr)/len(rr));assert abs(rm-z[arm]['rmse'])<1e-12
  for j in range(3):
   rm=math.sqrt(sum((q[arm+'_seeds'][j]-q['truth'])**2 for q in rr)/len(rr));assert abs(rm-z[arm]['seed_rmse'][j])<1e-12

stage=sys.argv[1];z=json.loads((H/f'{stage}_v1.json').read_text(encoding='utf8'));d=[q for q in data if q['validator']=='DIAG10'];verify_stats(z['baseline'],d)
d2=[q for q in d if q['day']>=179];assert len(d2)==46*24
byday={}
for q in d2:byday.setdefault((q['farm'],q['day']),[]).append(q)
gains={k:sum((q['REF']-q['truth'])**2-(q['THS']-q['truth'])**2 for q in rr) for k,rr in byday.items()}
top=sorted(gains,key=lambda k:(-gains[k],k))[:5]
assert top==[(q['farm'],q['day']) for q in z['top5']]
assert abs(sum(gains.values())-z['net_gain_sse'])<1e-10
for name,seed in [('early',202610091),('late',202610092)]:
 rr=[q for q in d2 if (q['hour']<=8)==(name=='early')];a=z['primary'][name];verify_stats(a,rr)
 for f in ['F13','F47']:verify_stats(a['farms'][f],[q for q in rr if q['farm']==f])
 verify_stats(a['without_top5'],[q for q in rr if (q['farm'],q['day']) not in top])
 # Construct block table independently from individual rows; replicate fixed RNG draws.
 blocks=sorted({(q['farm'],q['day']//5) for q in rr});vals=[]
 for k in blocks:
  qs=[q for q in rr if (q['farm'],q['day']//5)==k]
  vals.append((sum((q['REF']-q['truth'])**2 for q in qs),sum((q['THS']-q['truth'])**2 for q in qs),len(qs)))
 pools=[[i for i,k in enumerate(blocks) if k[0]==f] for f in ['F13','F47']];rnd=random.Random(seed);bs=[]
 for _ in range(20000):
  chosen=[rnd.choice(pool) for pool in pools for _ in range(len(pool))]
  x=sum(vals[i][0] for i in chosen);y=sum(vals[i][1] for i in chosen);n=sum(vals[i][2] for i in chosen)
  bs.append(math.sqrt(y/n)-math.sqrt(x/n))
 bs.sort();assert sum(v>=0 for v in bs)/20000==a['bootstrap']['p_worse']
 assert abs(bs[499]-a['bootstrap']['delta_rmse_ci95'][0])<1e-12
 assert abs(bs[19499]-a['bootstrap']['delta_rmse_ci95'][1])<1e-12
if stage=='final':
 for item in z['segments']:
  rr=[q for q in data if q['validator']==item['validator'] and q['farm']==item['farm'] and (q['day']>=179)==item['pass2'] and (q['hour']<=8)==(item['hours']=='early')];verify_stats(item,rr)
 for item in z['target_groups']:
  vr=[q for q in data if q['validator']==item['validator']];means={}
  for q in vr:means.setdefault((q['farm'],q['day'],q['fold']),[]).append(q['truth'])
  high={k for k,yy in means.items() if sum(yy)/len(yy)>=1.2}
  rr=[q for q in vr if ((q['farm'],q['day'],q['fold']) in high)==(item['target_group']=='high')];verify_stats(item,rr)
result={'status':'INDEPENDENT_ARITHMETIC_PASS','stage':stage,'cache_files':len(files),'cache_rows':len(data),'unique_rows':len(identity),'pass2_days':len(byday),'finite_bounds':True,'repeated_identity_consistent':True,'scores_seeds_segments_bootstrap_rechecked':True}
out=H/f'critic_independent_{stage}_v1.json';assert not out.exists();out.write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps(result))
