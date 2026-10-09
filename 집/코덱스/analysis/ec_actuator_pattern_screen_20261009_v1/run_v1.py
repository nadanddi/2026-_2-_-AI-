from pathlib import Path
import sys,os,csv,json,math,hashlib
sys.dont_write_bytecode=True
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[name]='1'
H=Path(__file__).resolve().parent;R=H.parents[3];sys.path.insert(0,str(R/'.analysis-tools/python'))
DLL=[]
if hasattr(os,'add_dll_directory'):
 for p in (R/'.analysis-tools/python').glob('**/.libs'):
  DLL.append(os.add_dll_directory(str(p)))
import numpy as np
CHANNELS=['act_vent','act_circfan','act_heating','act_shade','act_co2','act_fog','act_thermal'];CUTS=[8,15,23]
SHA=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
XPATH=R/'공용/대회자료/정형데이터/참가자_배포/train_X.csv';OOFS=sorted((R/'집/클로드/research/local/ct1_ckpt').glob('DIAG10_*.csv'));assert len(OOFS)==10
stage=sys.argv[1]
FEATURES=[f'{c}__h{h}__{k}' for c in CHANNELS for h in CUTS for k in ['mean','active','full','switches']+(['daynight'] if h>=15 else [])];assert len(FEATURES)==98
pins={str(p.relative_to(R)):SHA(p) for p in [XPATH,*OOFS]};reg=H/'registration_v1.json'
if stage=='prepare':
 assert not reg.exists();reg.write_text(json.dumps({'inputs':pins,'script_sha':SHA(Path(__file__)),'plan_sha':SHA(H/'PLAN_v1.md')},ensure_ascii=False,indent=2),encoding='utf8')
 labels={};oofdays={}
 for p in OOFS:
  with p.open(encoding='utf8',newline='') as f:
   for r in csv.DictReader(f):
    assert r['row_id'] not in labels
    ps=[min(float(r['hi']),max(float(r['lo']),math.fsum(w*float(r[f'REF_{k}_{s}']) for k,w in [('et',.6),('lgb',.3),('mlp',.1)]))) for s in (2121,4343,6565)]
    labels[r['row_id']]=(float(r['sub_ec']),math.fsum(ps)/3)
 with XPATH.open(encoding='utf8',newline='') as f:
  rd=csv.DictReader(f);header=rd.fieldnames;assert all(c in header for c in CHANNELS)
  x={r['row_id']:r for r in rd if r['row_id'] in labels}
 assert len(x)==len(labels)==8640
 for rid in sorted(labels):
  farm,ds,hs=rid.split('_');d,h=int(ds),int(hs);oofdays.setdefault((farm,d),[]).append((h,rid))
 assert len(oofdays)==360
 out=[]
 for (farm,d),pairs in sorted(oofdays.items()):
  pairs.sort();assert [h for h,_ in pairs]==list(range(24));rids=[rid for _,rid in pairs]
  z={'farm':farm,'day':d,'ec':math.fsum(labels[rid][0] for rid in rids)/24,'pred':math.fsum(labels[rid][1] for rid in rids)/24};z['residual']=z['ec']-z['pred']
  for c in CHANNELS:
   vals=np.array([float(x[rid][c]) if x[rid][c] else np.nan for rid in rids]);assert not np.isinf(vals).any()
   for h in CUTS:
    a=vals[:h+1];base=f'{c}__h{h}__';ok=np.isfinite(a).all()
    z.update({base+'mean':float(a.mean()) if ok else math.nan,base+'active':float((a>.1).sum()) if ok else math.nan,base+'full':float((a>=99.9).sum()) if ok else math.nan,base+'switches':float((np.abs(np.diff(a))>.1).sum()) if ok else math.nan})
    if h>=15:z[base+'daynight']=float(vals[9:16].mean()-vals[:9].mean()) if np.isfinite(vals[:16]).all() else math.nan
  out.append(z)
 with (H/'daily_features_v1.csv').open('w',encoding='utf8',newline='') as f:w=csv.DictWriter(f,fieldnames=['farm','day','ec','pred','residual']+FEATURES);w.writeheader();w.writerows(out)
 pr=dict(rows=8640,days=360,pass1=sum(z['day']<179 for z in out),pass2=sum(z['day']>=179 for z in out),missing_features={k:sum(not math.isfinite(z[k]) for z in out) for k in FEATURES},source_columns=header,feature_count=98,target_y_source='CT1 DIAG10 public OOF only',official_train_y_loaded=False,official_test_values_loaded=False,profile_columns={c:{'missing':sum(not x[rid][c] for rid in x),'min':min(float(x[rid][c]) for rid in x if x[rid][c]),'max':max(float(x[rid][c]) for rid in x if x[rid][c])} for c in CHANNELS},feature_sha=SHA(H/'daily_features_v1.csv'))
 (H/'profile_v1.json').write_text(json.dumps(pr,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps({k:v for k,v in pr.items() if k not in ['missing_features','source_columns']},ensure_ascii=True));sys.exit()
saved=json.loads(reg.read_text(encoding='utf8'));assert saved['inputs']==pins and saved['script_sha']==SHA(Path(__file__)) and saved['plan_sha']==SHA(H/'PLAN_v1.md')
with (H/'daily_features_v1.csv').open(encoding='utf8',newline='') as f:rows=list(csv.DictReader(f))
F=np.array([r['farm'] for r in rows]);D=np.array([int(r['day']) for r in rows]);X=np.array([[float(r[k]) for k in FEATURES] for r in rows]);Y={t:np.array([float(r[t]) for r in rows]) for t in ['ec','residual']}
assert SHA(H/'daily_features_v1.csv')==json.loads((H/'profile_v1.json').read_text(encoding='utf8'))['feature_sha']
def rank(v):
 order=np.argsort(v,kind='stable');r=np.empty(len(v));i=0
 while i<len(v):
  j=i+1
  while j<len(v) and v[order[j]]==v[order[i]]:j+=1
  r[order[i:j]]=(i+1+j)/2;i=j
 return r
def centered(v,f,d,width):
 a=rank(v);groups={}
 for i,k in enumerate(zip(f,d//width)):groups.setdefault(k,[]).append(i)
 for ix in groups.values():a[ix]-=a[ix].mean()
 return a,groups
def corr(x,y,mask,width=20):
 m=mask&np.isfinite(x)&np.isfinite(y);n=int(m.sum())
 if n<4:return None
 a,_=centered(x[m],F[m],D[m],width);b,_=centered(y[m],F[m],D[m],width);den=np.linalg.norm(a)*np.linalg.norm(b)
 return float(a@b/den) if den>0 else None
def pcorr_thermal(i,target,mask):
 h=FEATURES[i].split('__')[1];j=FEATURES.index(f'act_thermal__{h}__mean');x,y,c=X[:,i],Y[target],X[:,j];m=mask&np.isfinite(x)&np.isfinite(y)&np.isfinite(c)
 a,_=centered(x[m],F[m],D[m],20);b,_=centered(y[m],F[m],D[m],20);t,_=centered(c[m],F[m],D[m],20);den=t@t
 if den>0:a=a-t*(a@t/den);b=b-t*(b@t/den)
 den=np.linalg.norm(a)*np.linalg.norm(b);return float(a@b/den) if den>1e-12 else None
mask1=D<179;mask2=D>=179;results=[];permcaches={}
for target,y in Y.items():
 for i,k in enumerate(FEATURES):
  m=mask1&np.isfinite(X[:,i])&np.isfinite(y);n=int(m.sum());counts={f:int((m&(F==f)).sum()) for f in ['F13','F47']};rho=corr(X[:,i],y,m)
  p=1.
  if n>=50 and min(counts.values())>=20 and rho is not None:
   key=(target,tuple(np.flatnonzero(m)));a,groups=centered(X[m,i],F[m],D[m],20);b,_=centered(y[m],F[m],D[m],20);an=np.linalg.norm(a);bn=np.linalg.norm(b)
   if key not in permcaches:
    rng=np.random.default_rng(202610093);P=np.tile(b,(4000,1))
    for ids in groups.values():
     vv=b[ids]
     for q in range(4000):P[q,ids]=rng.permutation(vv)
    permcaches[key]=P/bn
   null=permcaches[key]@(a/an);p=float((int((np.abs(null)>=abs(rho)-1e-14).sum())+1)/4001)
  z=dict(feature=k,target=target,n_discovery=n,rho=rho,p=p,rho_width40=corr(X[:,i],y,m,40),farm_discovery={f:corr(X[:,i],y,mask1&(F==f)) for f in ['F13','F47']},rho_pass2=corr(X[:,i],y,mask2),farm_pass2={f:corr(X[:,i],y,mask2&(F==f)) for f in ['F13','F47']})
  results.append(z)
order=sorted(range(len(results)),key=lambda i:results[i]['p']);last=1.
for j in range(len(order)-1,-1,-1):
 i=order[j];last=min(last,results[i]['p']*len(results)/(j+1));results[i]['q']=last
worst2=set(np.argsort(np.abs(Y['residual'][mask2]))[-2:]);ids=np.flatnonzero(mask2);stress=mask2.copy();stress[ids[list(worst2)]]=False
for z in results:
 i=FEATURES.index(z['feature']);r=z['rho'];direction=lambda x:x is not None and r is not None and x*r>0
 z['discovery_supported']=z['q']<.05 and r is not None and abs(r)>=.20 and all(direction(x) for x in z['farm_discovery'].values())
 z['transfer_supported']=z['discovery_supported'] and direction(z['rho_pass2']) and abs(z['rho_pass2'])>=.15 and all(direction(x) and abs(x)>=.15 for x in z['farm_pass2'].values())
 if z['discovery_supported']:
  z['pass2_drop_top2_residual_rho']=corr(X[:,i],Y[z['target']],stress);z['discovery_thermal_partial_rho']=pcorr_thermal(i,z['target'],mask1);z['pass2_thermal_partial_rho']=pcorr_thermal(i,z['target'],mask2)
p=H/'final_v1.json';assert not p.exists();p.write_text(json.dumps(dict(results=results,hypotheses=196,permutations=4000,selected_residual=[z['feature'] for z in results if z['target']=='residual' and z['transfer_supported']],selected_ec=[z['feature'] for z in results if z['target']=='ec' and z['transfer_supported']],stress_removed=[{'farm':str(F[i]),'day':int(D[i])} for i in ids[list(worst2)]],feature_sha=SHA(H/'daily_features_v1.csv'),adopted=False),ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps({'residual_candidates':[z for z in results if z['target']=='residual' and z['transfer_supported']],'ec_candidates':[z for z in results if z['target']=='ec' and z['transfer_supported']],'discovery_counts':{t:sum(z['discovery_supported'] for z in results if z['target']==t) for t in Y}},ensure_ascii=True))
