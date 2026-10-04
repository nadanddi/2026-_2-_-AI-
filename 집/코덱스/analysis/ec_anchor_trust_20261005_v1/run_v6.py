"""Predeclared families25/26; public labels only; immutable outputs; one mode at a time."""
from pathlib import Path
import sys,json,hashlib,math,argparse,platform
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;ROOT=H.parents[3]
sys.path.insert(0,str(ROOT/'집/클로드/research'));import env
sys.path.insert(0,str(H.parent/'statistical_experiments_20261003_v1'));import support as S
import numpy as np,pandas as pd,sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression,Ridge
from threadpoolctl import threadpool_limits
OUT=ROOT/'집/코덱스/local'/H.name
INNER=ROOT/'집/코덱스/local/ec_matched_inner_calibration_20261003_v1'
PREV=H.parent/'ec_final_output_loss_20261004_v1/preparation_v3.json'
SEEDS=[7,101,2024];KEYS=[(v,k) for v,n in [('DIAG10',10),('A',5),('B',5),('EXT10',1),('EXT12',1)] for k in range(n)]
CC=['in_temp','in_hum','in_co2','act_heating','act_vent','act_thermal','act_shade','act_circfan','act_fog','act_co2','out_temp','out_hum','out_rad','out_wspd']
CFG=dict(families={'GATE':25,'RIDGE':26},alpha=.025/26,seeds=SEEDS,validators=[v for v,n in [('DIAG10',10),('A',5),('B',5),('EXT10',1),('EXT12',1)]],strict15=True,pass2_only=True,signature='14 current-record available prefix means; missing imputed reference-hour median only; reference-only standardization per hour',reference='samefarm samepass strictly earlier public training day; outer/inner validation +/-1 purged',anchor='mean of 3 nearest signature reference days, stable day tie-break',gate='cost-sensitive LogisticRegression C=1 max_iter=3000 tol=1e-8; probability>=.5',cost='abs(SSE_base-SSE_anchor), normalized to mean1 on nonzero eligible inner rows',proposed_delta='.5*clip(anchor-prefix_v2,-.6,.6)',ridge='Ridge alpha=100 target hourly residual, .2*clip(predicted residual,-.3,.3)',features='current baseline, prefix baseline, anchor, anchor-minus-prefix, d1,d2,top3label std, log1p(n), 14 prefix means, hour/23',baseline='original .8R3+.2PFN4bag -> shrink .5 -> inner-a/outer-tr bounds',output='post-final correction plus baseline, train-bounds clip; pass1/noanchors unchanged',inner='existing matched heldout b subset, base trained only a; anchor references a; meta preprocessing fit eligible b only',selection='fixed coefficients/model/threshold; no validation tuning',raw_ec_reads=0,test_value_reads=0,EL1_rescore=0)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,o):
 with Path(p).open('x',encoding='utf-8') as f:json.dump(o,f,ensure_ascii=False,indent=2)
def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def ids(a):return hashlib.sha256('\n'.join(map(str,a)).encode()).hexdigest()
def ar(a):
 a=np.asarray(a);return hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes()).hexdigest()
def close(a,b):
 a,b=np.asarray(a,float),np.asarray(b,float);assert a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all();e=float(np.max(abs(a-b)));assert e<=1e-12;return e
def npz(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def prefix(q,x):
 out=np.empty_like(x,dtype=float)
 for _,g in q.groupby(['farm','day'],sort=False):
  g=g.sort_values('hour');assert np.array_equal(g.hour,np.arange(24));ix=g.index.to_numpy();out[ix]=np.cumsum(x[ix],axis=0)/np.arange(1,25).reshape(-1,*([1]*(x.ndim-1)))
 return out
def prefix_missing(q,x):
 out=np.empty_like(x,dtype=float)
 for _,g in q.groupby(['farm','day'],sort=False):
  ix=g.sort_values('hour').index.to_numpy(); vv=x[ix]; count=np.cumsum(np.isfinite(vv),axis=0); out[ix]=np.divide(np.nancumsum(vv,axis=0),count,out=np.full_like(vv,np.nan),where=count>0)
 return out
def signatures(raw):
 a=raw.copy().sort_values(['farm','day','hour']).reset_index(drop=True);x=a[CC].to_numpy(float);assert not np.isinf(x).any(); return pd.DataFrame(prefix_missing(a,x),index=a.row_id,columns=CC)
def neighbors_slow(ref,q,sig):
 # Only reference input rows fit every standardization. Query never updates state.
 daily=ref.groupby(['farm','day']).sub_ec.mean();A=sig.reindex(ref.row_id).to_numpy();Q=sig.reindex(q.row_id).to_numpy();out=np.zeros((len(q),6));sources=[]; Qfilled=np.empty_like(Q)
 for h in range(24):
  rr=ref[ref.hour==h];av=A[ref.hour.to_numpy()==h];med=np.nanmedian(av,axis=0);med=np.nan_to_num(med);av=np.where(np.isnan(av),med,av);mu=av.mean(0);sd=av.std(0);sd[sd==0]=1
  qq=q[q.hour==h];qv=Q[q.hour.to_numpy()==h];qv=np.where(np.isnan(qv),med,qv);Qfilled[q.hour.to_numpy()==h]=qv
  for j,(ii,r) in enumerate(qq.iterrows()):
   allowed=(rr.farm==r.farm)&((rr.day>=179)==(r.day>=179))&(rr.day<r.day)
   c=rr[allowed];a=av[allowed.to_numpy()];n=len(c)
   if not n:sources.append({'row_id':r.row_id,'anchors':[]});continue
   dist=np.sqrt(np.mean(((a-qv[j])/sd)**2,axis=1));order=np.lexsort((c.day.to_numpy(),dist));z=order[:3];keys=list(zip(c.farm.iloc[z],c.day.iloc[z]));vals=np.array([daily[k] for k in keys]);ds=dist[order]
   out[ii]=[vals.mean(),ds[0],ds[min(1,n-1)],vals.std(),np.log1p(n),1];sources.append({'row_id':r.row_id,'anchors':[int(d) for f,d in keys]})
 return out,Qfilled,sources
def neighbors(ref,q,sig):
 daily=ref.groupby(['farm','day']).sub_ec.mean(); A=sig.reindex(ref.row_id).to_numpy();Q=sig.reindex(q.row_id).to_numpy();out=np.zeros((len(q),6));Qfilled=np.empty_like(Q);sources=[]
 for h in range(24):
  rr=ref[ref.hour==h];av=A[ref.hour.to_numpy()==h];med=np.nanmedian(av,axis=0);med=np.nan_to_num(med);av=np.where(np.isnan(av),med,av);sd=av.std(0);sd[sd==0]=1
  qi=np.where(q.hour.to_numpy()==h)[0];qv=Q[qi];qv=np.where(np.isnan(qv),med,qv);Qfilled[qi]=qv
  for f in ['F13','F47']:
   for late in [False,True]:
    ai=np.where(((rr.farm==f)&((rr.day>=179)==late)).to_numpy())[0];jj=np.where(((q.iloc[qi].farm==f)&((q.iloc[qi].day>=179)==late)).to_numpy())[0]
    if not len(jj):continue
    if not len(ai):sources.extend({'row_id':q.row_id.iloc[qi[j]],'anchors':[]} for j in jj);continue
    ai=ai[np.argsort(rr.day.to_numpy()[ai],kind='stable')];days=rr.day.to_numpy()[ai];eligible=days[None,:]<q.day.to_numpy()[qi[jj],None];dist=np.sqrt(np.mean(((av[ai][None,:,:]-qv[jj][:,None,:])/sd)**2,axis=2));dist[~eligible]=np.inf;order=np.argsort(dist,axis=1,kind='stable');labels=np.array([daily[f,int(d)] for d in days])
    for i,j in enumerate(jj):
     ii=qi[j];n=int(eligible[i].sum())
     if not n:sources.append({'row_id':q.row_id.iloc[ii],'anchors':[]});continue
     z=order[i,:min(3,n)];vals=labels[z];ds=dist[i,order[i]];out[ii]=[vals.mean(),ds[0],ds[min(1,n-1)],vals.std(),np.log1p(n),1];sources.append({'row_id':q.row_id.iloc[ii],'anchors':[int(d) for d in days[z]]})
 return out,Qfilled,sources

def design(q,base,N,Q):
 pm=prefix(q,np.asarray(base));x=np.column_stack([base,pm,N[:,0],N[:,0]-pm,N[:,1:5],Q,q.hour.to_numpy()/23]);ok=(q.day.to_numpy()>=179)&(N[:,5]>0);delta=np.where(ok,.5*np.clip(N[:,0]-pm,-.6,.6),0.);assert x.shape[1]==23 and np.isfinite(x).all();return x,ok,delta
def preparation():
 assert sha(Path(S.__file__))=='82074e7a04ea681bc94e94e1c90354a743609e0f008b1a378a5a9a23575debed'
 assert sha(PREV)=='df4ff2c3b7617676c61e4cee4b8e9690f9312a7cca448c0caec189cb5051b860'; old=load(PREV);assert old['status']=='PASS_PREPARATION_MODEL_FIT0_PREDICT0_SCORE0'
 lab,core,wv,folds,outer=S.loadec();assert [(v,k) for v,k,tm,vm in folds]==KEYS and len(lab)==8640
 assert sha(Path(env.DATA)/'train_X.csv')==old['inputs']['train_X'] and sha(ROOT/'집/코덱스/local/ec_dc4_integration_20261002_v1/v2_integration_oof.csv')==old['inputs']['public_oof']
 assert sha(Path(core.__file__))==old['dependencies']['core']
 raw=pd.read_csv(Path(env.DATA)/'train_X.csv',usecols=['row_id']+CC);raw=raw[raw.row_id.isin(lab.row_id)].copy();raw['farm']=raw.row_id.str[:3];raw['day']=raw.row_id.str[4:7].astype(int);raw['hour']=raw.row_id.str[8:10].astype(int);assert len(raw)==8640
 sig=signatures(raw); causal=[]
 for farm in ['F13','F47']:
  day=int(raw.loc[raw.farm==farm,'day'].min());g=raw[(raw.farm==farm)&(raw.day==day)].copy()
  for h in [0,6,12]:
   cut=g[g.hour<=h];v=signatures(cut).to_numpy();full=sig.reindex(cut.row_id).to_numpy();assert np.array_equal(np.isnan(v),np.isnan(full));close(v[np.isfinite(v)],full[np.isfinite(full)])
   altered=g.copy();altered.loc[altered.hour>h,CC]=12345.;v2=signatures(altered).reindex(cut.row_id).to_numpy();assert np.array_equal(np.isnan(v2),np.isnan(full));close(v2[np.isfinite(v2)],full[np.isfinite(full)]);causal.append(dict(farm=farm,day=day,hour=h,error=0.))
 idx=lab.set_index('row_id');records=[];refs={} 
 for v,k,tm,vm in folds:
  prior=next(m for m in old['manifest'] if m['validator']==v and m['fold']==k)
  for path,digest in prior['cache_hashes'].items():assert sha(ROOT/path)==digest,path
  tr=lab[tm].reset_index(drop=True);q=lab[vm].reset_index(drop=True);z=npz(INNER/f'{v}_{k}_cpu.npz');a=idx.reindex(z['inner_train_id']).reset_index();b=idx.reindex(z['row_id']).reset_index()
  assert ids(a.row_id)==prior['inner_a_ids'] and ids(b.row_id)==prior['inner_b_ids'] and ids(tr.row_id)==prior['outer_train_ids'] and ids(q.row_id)==prior['outer_query_ids']
  for r,t in [(a,b),(tr,q)]:
   forbidden={(f,int(d)+j) for f,d in t[['farm','day']].itertuples(index=False,name=None) for j in [-1,0,1]};assert not set(r[['farm','day']].itertuples(index=False,name=None))&forbidden
  assert a.row_id.is_unique and b.row_id.is_unique; assert set(a.row_id)|set(b.row_id)<=set(tr.row_id) and not (set(a.row_id)|set(b.row_id))&set(q.row_id)
  close([z['lo'],z['hi']],[a.sub_ec.min(),a.sub_ec.max()]);bag=[]
  for s in [1,2,3,4]:
   p=npz(INNER/f'{v}_{k}_pfn_{s}.npz');assert np.array_equal(p['row_id'],b.row_id) and set(p['context_row_id'])<=set(a.row_id);bag.append(p['prediction'])
  bag=np.mean(bag,axis=0);NB,QB,SB=neighbors(a,b,sig);NQ,QQ,SQ=neighbors(tr,q,sig)
  if v=='DIAG10' and k==0:
   toy=q.iloc[:24].reset_index(drop=True);nf,qf,sf=neighbors(tr,toy,sig);ns,qs,ss=neighbors_slow(tr,toy,sig);close(nf,ns);close(qf,qs);assert sorted(sf,key=lambda z:z['row_id'])==sorted(ss,key=lambda z:z['row_id'])
  bases={};inner={};bounds=(float(tr.sub_ec.min()),float(tr.sub_ec.max()))
  for s in SEEDS:
   rawbase=.8*z[f'r3_{s}']+.2*bag;inner[s]=np.clip(.5*rawbase+.5*prefix(b,rawbase),z['lo'],z['hi'])
   bases[s]=outer[(outer.validator==v)&(outer.validation_fold==k)&(outer.seed==s)].set_index('row_id').season_v2.reindex(q.row_id).to_numpy();bc=npz(ROOT/f'집/코덱스/local/ec_tabpfn35_20261003_v1/{v}_{k}_baseline.npz');assert np.array_equal(bc['row_id'],q.row_id);close(bases[s],bc[f'baseline_{s}'])
  record=dict(validator=v,fold=k,prior_cache_hashes=prior['cache_hashes'],a_ids=ids(a.row_id),b_ids=ids(b.row_id),tr_ids=ids(tr.row_id),q_ids=ids(q.row_id),inner_neighbors=ar(NB),outer_neighbors=ar(NQ),inner_signature=ar(QB),outer_signature=ar(QQ),bounds=list(bounds))
  records.append(record);refs[v,k]=dict(a=a,b=b,tr=tr,q=q,NB=NB,NQ=NQ,QB=QB,QQ=QQ,SB=SB,SQ=SQ,inner=inner,baseline=bases,bounds=bounds,innerbounds=(float(z['lo']),float(z['hi'])),signature=record)
  print('PREP',v,k,flush=True)
 prep=dict(causal_prefix=causal,status='PASS_PREPARATION_FIT0_SCORE0',config=CFG,source=sha(Path(__file__)),prior=sha(PREV),inputs=old['inputs'],manifest=records,runtime=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,sklearn=sklearn.__version__))
 return refs,prep
def fit_predict(mode,ref,seed):
 b,q=ref['b'],ref['q'];xb,okb,db=design(b,ref['inner'][seed],ref['NB'],ref['QB']);xq,okq,dq=design(q,ref['baseline'][seed],ref['NQ'],ref['QQ']);ib=ref['inner'][seed];ob=ref['baseline'][seed];lo,hi=ref['innerbounds'];yb=b.sub_ec.to_numpy();trial=np.clip(ib+db,lo,hi);cost=(ib-yb)**2-(trial-yb)**2
 if mode=='GATE':eligible=okb&(abs(cost)>1e-14)
 else:eligible=okb
 scale=StandardScaler();xx=scale.fit_transform(xb[eligible]);test=scale.transform(xq);assert len(xx)>0
 if mode=='GATE':
  label=(cost[eligible]>0).astype(int);w=abs(cost[eligible]);w=w/w.mean()
  if len(np.unique(label))<2:prob=np.full(len(q),float(label[0]));model=None
  else:model=LogisticRegression(C=1,max_iter=3000,tol=1e-8,random_state=seed).fit(xx,label,sample_weight=w);prob=model.predict_proba(test)[:,1];assert model.n_iter_[0]<3000
  delta=np.where(okq&(prob>=.5),dq,0.)
 else:
  model=Ridge(alpha=100).fit(xx,(yb-ib)[eligible]);prob=model.predict(test);delta=np.where(okq,.2*np.clip(prob,-.3,.3),0.)
 candidate=np.clip(ob+delta,*ref['bounds']);assert np.array_equal(candidate[~okq],ob[~okq])
 fit=dict(eligible_ids=ids(b.loc[eligible,'row_id']),features_sha=ar(xb[eligible]),cost_sha=ar(cost[eligible]),mode=mode,seed=seed,n=int(eligible.sum()),positive=int((cost[eligible]>0).sum()),mean=scale.mean_.tolist(),scale=scale.scale_.tolist(),coef=model.coef_.tolist() if model is not None else None,intercept=np.asarray(model.intercept_).tolist() if model is not None else None,model_none=model is None)
 return candidate,delta,prob,fit
def replay(mode,x,fit):
 xx=(x-np.asarray(fit['mean']))/fit['scale']
 if fit['model_none']:return np.full(len(x),float(fit['positive']==fit['n']))
 lin=xx@np.asarray(fit['coef']).reshape(-1)+float(np.asarray(fit['intercept']).reshape(-1)[0])
 if mode=='RIDGE':return lin
 z=np.empty_like(lin);pos=lin>=0;z[pos]=1/(1+np.exp(-lin[pos]));e=np.exp(lin[~pos]);z[~pos]=e/(1+e);return z
def actual(mode):
 refs,prep=preparation(); reg=load(H/'registration_v1.json'); assert reg['run_sha']==sha(Path(__file__)) and reg['prep_sha']==sha(H/'preparation_v6.json') and reg['prereg_sha']==sha(H/'preregistration_v1.md'); assert load(H/'preparation_v6.json')==prep and (H/'preregistration_v1.md').exists();OUT.mkdir(parents=True,exist_ok=True);dest=OUT/mode;dest.mkdir(exist_ok=False);manifest=[]
 for (v,k),ref in refs.items():
  for seed in SEEDS:
   with threadpool_limits(limits=2):p,d,z,fit=fit_predict(mode,ref,seed)
   q=ref['q'];baseline=ref['baseline'][seed]
   if v=='DIAG10' and k==0 and seed==7:
    with threadpool_limits(limits=2):p2,d2,z2,fit2=fit_predict(mode,ref,seed)
    close(p,p2);close(d,d2);close(z,z2);assert fit==fit2
    x,ok,dq=design(ref['q'],ref['baseline'][seed],ref['NQ'],ref['QQ']);replayed=replay(mode,x,fit)
    close(z,replayed);close(replayed,replay(mode,x[::-1],fit)[::-1]);close(replayed,np.array([replay(mode,x[i:i+1],fit)[0] for i in range(len(x))]));changed=x[:8].copy();changed[1:]+=10000.;close(replayed[:1],replay(mode,changed,fit)[:1])
    scalar=np.array([min(ref['bounds'][1],max(ref['bounds'][0],float(a)+float(b))) for a,b in zip(ref['baseline'][seed],d)]);close(p,scalar) 
    save(H/f'first_{mode}_v1.json',dict(status='PASS',repeat_error=close(p,p2),serialized_error=close(z,replayed),reverse_error=close(replayed,replay(mode,x[::-1],fit)[::-1]),single_error=close(replayed,np.array([replay(mode,x[i:i+1],fit)[0] for i in range(len(x))])),other_query_error=close(replayed[:1],replay(mode,changed,fit)[:1]),scalar_error=close(p,scalar),causal_prefix=prep['causal_prefix'],fit=fit,source=sha(Path(__file__))))
   out=q[['row_id','farm','day','hour']].copy();out['y']=q.sub_ec;out['baseline']=baseline;out['candidate']=p;out['delta']=d;out['confidence']=z;out['anchor']=ref['NQ'][:,0];out['d1']=ref['NQ'][:,1];out['d2']=ref['NQ'][:,2];out['validator']=v;out['fold']=k;out['seed']=seed;out['clip_lo']=ref['bounds'][0];out['clip_hi']=ref['bounds'][1];path=dest/f'{v}_{k}_{seed}.csv';out.to_csv(path,index=False)
   save(dest/f'{v}_{k}_{seed}_fit.json',fit);manifest.append(dict(validator=v,fold=k,seed=seed,csv_sha=sha(path),fit_sha=sha(dest/f'{v}_{k}_{seed}_fit.json')))
  print(mode,v,k,'COMPLETE',flush=True)
 assert len(manifest)==66;pd.concat([pd.read_csv(dest/f'{v}_{k}_{s}.csv',float_precision='round_trip') for v,k in KEYS for s in SEEDS]).to_csv(dest/'oof.csv',index=False)
 save(H/f'fit_{mode}_v1.json',dict(status='PASS_FIT66_SCORE0',source=sha(Path(__file__)),preparation=sha(H/'preparation_v6.json'),manifest=manifest,aggregate=sha(dest/'oof.csv')))
 print(mode,'ALL66_COMPLETE',flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');ap.add_argument('--mode',choices=['GATE','RIDGE']);args=ap.parse_args()
 if args.prepare:refs,p=preparation();save(H/'preparation_v6.json',p)
 else:assert args.mode;actual(args.mode)


