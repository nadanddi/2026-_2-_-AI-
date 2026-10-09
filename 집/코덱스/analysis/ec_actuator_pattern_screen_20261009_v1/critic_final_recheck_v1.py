from pathlib import Path
import csv,json,math,sys,os
H=Path(__file__).resolve().parent;R=H.parents[3]
for s in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[s]='1'
sys.path.insert(0,str(R/'.analysis-tools/python'));handles=[]
if hasattr(os,'add_dll_directory'):
 for d in (R/'.analysis-tools/python').glob('**/.libs'):handles.append(os.add_dll_directory(str(d)))
import numpy as np
rows=list(csv.DictReader((H/'daily_features_v1.csv').open(encoding='utf8')));z=json.loads((H/'final_v1.json').read_text(encoding='utf8'))
farms=[r['farm'] for r in rows];days=[int(r['day']) for r in rows]
def ranks(v):
 order=sorted(range(len(v)),key=lambda j:v[j]);out=[0.]*len(v);a=0
 while a<len(v):
  b=a+1
  while b<len(v) and v[order[b]]==v[order[a]]:b+=1
  for i in order[a:b]:out[i]=(a+b+1)/2
  a=b
 return out

def center(v,ids,width=20):
 a=ranks(v);g={}
 for j,i in enumerate(ids):g.setdefault((farms[i],days[i]//width),[]).append(j)
 for js in g.values():
  m=sum(a[j] for j in js)/len(js)
  for j in js:a[j]-=m
 return np.array(a),g

def rho(feature,target,ids,width=20):
 ids=[i for i in ids if math.isfinite(float(rows[i][feature])) and math.isfinite(float(rows[i][target]))]
 if len(ids)<4:return None
 a,_=center([float(rows[i][feature]) for i in ids],ids,width);b,_=center([float(rows[i][target]) for i in ids],ids,width)
 den=math.sqrt(sum(a*a)*sum(b*b));return float(sum(a*b)/den) if den>0 else None

def equal(a,b):
 assert (a is None)==(b is None)
 if a is not None:assert abs(a-b)<1e-12,(a,b)
one=[i for i,d in enumerate(days) if d<179];two=[i for i,d in enumerate(days) if d>=179];cache={};pvals=[]
for q in z['results']:
 feature,target=q['feature'],q['target'];v=rho(feature,target,one);equal(v,q['rho']);equal(rho(feature,target,two),q['rho_pass2']);equal(rho(feature,target,one,40),q['rho_width40'])
 for f in ['F13','F47']:
  equal(rho(feature,target,[i for i in one if farms[i]==f]),q['farm_discovery'][f]);equal(rho(feature,target,[i for i in two if farms[i]==f]),q['farm_pass2'][f])
 p=1.;ext=None
 if v is not None:
  a,g=center([float(rows[i][feature]) for i in one],one);b,_=center([float(rows[i][target]) for i in one],one)
  if target not in cache:
   rng=np.random.default_rng(202610093);matrix=np.tile(b,(4000,1))
   for js in g.values():
    subset=b[js]
    for j in range(4000):matrix[j,js]=rng.permutation(subset)
   cache[target]=matrix
  null=(cache[target]@a)/(math.sqrt(sum(a*a))*math.sqrt(sum(b*b)));ext=sum(abs(t)>=abs(v)-1e-14 for t in null);p=(ext+1)/4001
 assert ext==q['extreme_permutations'],(feature,target,ext,q['extreme_permutations']);equal(p,q['p']);pvals.append(p)
order=sorted(range(196),key=lambda i:pvals[i]);adjusted=[1.]*196;running=1.
for j in range(195,-1,-1):
 i=order[j];running=min(running,pvals[i]*196/(j+1));adjusted[i]=running
for i,q in enumerate(z['results']):
 equal(adjusted[i],q['q']);r=q['rho'];direction=lambda t:r is not None and t is not None and r*t>0
 disc=q['q']<.05 and r is not None and abs(r)>=.2 and all(direction(t) for t in q['farm_discovery'].values())
 trans=disc and direction(q['rho_pass2']) and abs(q['rho_pass2'])>=.15 and all(direction(t) and abs(t)>=.15 for t in q['farm_pass2'].values()) and min(q['n_pass2'].values())>=10
 assert disc==q['discovery_supported'] and trans==q['transfer_supported']
summary={'status':'INDEPENDENT_FINAL_STATISTICS_PASS','results_checked':196,'permutations_each':4000,'rho_p_q_farm_direction_selection_all_rechecked':True,'selected_residual':z['selected_residual'],'selected_ec':z['selected_ec'],'redundant_daynight_h15_h23':[c for c in ['act_vent','act_circfan','act_heating','act_shade','act_co2','act_fog','act_thermal'] if all(float(q[c+'__h15__daynight'])==float(q[c+'__h23__daynight']) for q in rows)],'discovery_layers':len({(farms[i],days[i]//20) for i in one}),'pass2_layers':len({(farms[i],days[i]//20) for i in two})}
p=H/'critic_final_recheck_v1.json';assert not p.exists();p.write_text(json.dumps(summary,indent=2),encoding='utf8');print(json.dumps(summary))
