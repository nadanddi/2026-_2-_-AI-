from pathlib import Path
import sys,json,math,argparse,gc
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;ROOT=H.parents[3]
sys.path.insert(0,str(ROOT/'집/클로드/research'));import env
sys.path.insert(0,str(ROOT/'집/코덱스/analysis/statistical_experiments_20261003_v1'));import support as S
import numpy as np,pandas as pd
from sklearn.base import clone
from threadpoolctl import threadpool_limits
INNER=ROOT/'집/코덱스/local/ec_matched_inner_calibration_20261003_v1'
IH=ROOT/'집/코덱스/analysis/ec_matched_inner_calibration_20261003_v1'
ORIGINAL=ROOT/'집/코덱스/local/ec_dc4_integration_20261002_v1'
OUT=ROOT/'집/코덱스/local/ec_nested_high_specialist_20261004_v1'
EXPECTED='b1041713e093ca9b8f17437e05b194f284a7e44e8a681a06ce033bc6fc8a7e40'
LAMBDA=.01

def arrayfile(path):
 with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}

def prefix(p,d):
 x=d[['farm','day','hour']].reset_index(drop=True).copy();x['p']=p
 x=x.sort_values(['farm','day','hour']);v=x.groupby(['farm','day']).p.transform(lambda z:z.expanding().mean())
 result=np.empty(len(x));result[x.index]=v;return result

def full_inner(v,k,tr,idx):
 z=arrayfile(INNER/f'{v}_{k}_cpu.npz')
 old=arrayfile(S.OUT/f'E_{v}_{k}_cpu.npz')
 assert idx.index.is_unique and len(z['inner_train_id'])>0 and len(z['row_id'])>0
 assert len(np.unique(z['inner_train_id']))==len(z['inner_train_id']) and len(np.unique(z['row_id']))==len(z['row_id'])
 assert np.array_equal(z['inner_train_id'],old['inner_train_id']) and np.array_equal(z['row_id'],old['row_id'])
 assert np.array_equal(old['outer_train_id'],tr.row_id)
 im,iv=S.inner(tr)
 for f in ['F13','F47']:
  if ((tr.farm==f)&(tr.day>=179)).any() and not ((tr.farm==f)&(tr.day>=179)&im).any():iv &= ~((tr.farm==f)&(tr.day>=179)).to_numpy()
 selected=set(tr.loc[iv,['farm','day']].itertuples(index=False,name=None))
 banned={(f,int(day)+j) for f,day in selected for j in [-1,0,1]}
 im=np.asarray([(f,int(day)) not in banned for f,day in zip(tr.farm,tr.day)])
 assert np.array_equal(z['inner_train_id'],tr.loc[im,'row_id']) and np.array_equal(z['row_id'],tr.loc[iv,'row_id'])
 a=idx.reindex(z['inner_train_id']).reset_index();b=idx.reindex(z['row_id']).reset_index()
 assert a.sub_ec.notna().all() and b.sub_ec.notna().all()
 assert set(a.row_id).isdisjoint(b.row_id) and set(a.row_id)|set(b.row_id)<=set(tr.row_id)
 banned={(f,int(day)+j) for f,day in b[['farm','day']].drop_duplicates().itertuples(index=False,name=None) for j in [-1,0,1]}
 assert not set(a[['farm','day']].itertuples(index=False,name=None))&banned
 assert abs(float(z['lo'])-a.sub_ec.min())<1e-12 and abs(float(z['hi'])-a.sub_ec.max())<1e-12
 preds=[]
 for seed in [1,2,3,4]:
  pp=arrayfile(INNER/f'{v}_{k}_pfn_{seed}.npz');ix=np.random.default_rng(seed).choice(len(a),min(2000,len(a)),replace=False)
  assert np.array_equal(pp['row_id'],b.row_id) and np.array_equal(pp['context_row_id'],a.row_id.iloc[ix])
  assert pp['prediction'].shape==(len(b),) and np.isfinite(pp['prediction']).all();preds.append(pp['prediction'])
 for seed in [7,101,2024]:assert z[f'r3_{seed}'].shape==(len(b),) and np.isfinite(z[f'r3_{seed}']).all()
 return a,b,z,np.mean(preds,axis=0)

def original_r3(v,k,seed,tr,va):
 path=ORIGINAL/f'{v}_{k}_r3_{seed}.npz';meta=json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
 assert S.sha(path)==meta['prediction_sha256'];z=arrayfile(path)
 assert np.array_equal(z['train_row_id'],tr.row_id) and np.array_equal(z['row_id'],va.row_id)
 assert np.max(abs(z['sub_ec']-va.sub_ec.to_numpy()))<1e-12
 assert np.array_equal(z['raw_r3'],.6*z['raw_et']+.3*z['raw_lgb']+.1*z['raw_mlp'])
 assert np.max(abs(np.asarray(meta['provenance']['train_target_bounds'])-[tr.sub_ec.min(),tr.sub_ec.max()]))<1e-12
 return z['raw_r3']

def preflight(lab,folds,preflight_outer):
 assert S.sha(IH/'run.py')==EXPECTED and (INNER/'source_sha.txt').read_text()==EXPECTED
 paths=[INNER/f'{v}_{k}_cpu.npz' for v,k,_,_ in folds]+[INNER/f'{v}_{k}_pfn_{s}.npz' for v,k,_,_ in folds for s in [1,2,3,4]]
 paths += [IH/'fit_audit_v1.json',INNER/'oof.csv']
 missing=[str(p.relative_to(ROOT)) for p in paths if not p.exists()]
 record=dict(status='WAIT_FOR_COMPLETE_INNER' if missing else 'PASS',missing=missing,required_cpu=22,required_pfn=88,source_sha256=S.sha(Path(__file__)),matched_source_sha256=EXPECTED)
 if missing:
  idx=lab.set_index('row_id');checked=[]
  for v,k,tm,vm in folds:
   needed=[INNER/f'{v}_{k}_cpu.npz']+[INNER/f'{v}_{k}_pfn_{s}.npz' for s in [1,2,3,4]]
   if not all(p.exists() for p in needed):continue
   tr,va=lab[tm].reset_index(drop=True),lab[vm].reset_index(drop=True)
   a,b,z,bag=full_inner(v,k,tr,idx)
   for s in [7,101,2024]:original_r3(v,k,s,tr,va)
   checked.append(dict(validator=v,fold=k,n_inner_train=len(a),n_inner_query=len(b)))
  record['ready_fold_id_audit']=checked;return record
 fit=json.loads((IH/'fit_audit_v1.json').read_text(encoding='utf-8'));assert fit['status']=='PASS' and fit['checks']==22*3*24
 for fn in ['cpu_reproduction_v1.json','pfn_reproduction_v1.json']:assert json.loads((IH/fn).read_text(encoding='utf-8'))['status']=='PASS'
 od=pd.read_csv(INNER/'oof.csv',float_precision='round_trip');assert len(od)==83160 and not od.duplicated(['validator','fold','seed','row_id']).any()
 expected={(v,k,s,r) for v,k,tm,vm in folds for s in [7,101,2024] for r in lab.loc[vm,'row_id']}
 observed=set(zip(od.validator,od.fold,od.seed,od.row_id));assert observed==expected
 idx=lab.set_index('row_id');checked=[]
 for v,k,tm,vm in folds:
  tr,va=lab[tm].reset_index(drop=True),lab[vm].reset_index(drop=True)
  a,b,z,bag=full_inner(v,k,tr,idx)
  for s in [7,101,2024]:
   original_r3(v,k,s,tr,va)
   g=od[(od.validator==v)&(od.fold==k)&(od.seed==s)].set_index('row_id').reindex(va.row_id)
   assert g[['y','baseline','candidate']].notna().all().all() and np.isfinite(g[['y','baseline','candidate']]).all().all()
   assert np.max(abs(g.y.to_numpy()-va.sub_ec.to_numpy()))<1e-12
   ref=preflight_outer[(preflight_outer.validator==v)&(preflight_outer.validation_fold==k)&(preflight_outer.seed==s)].set_index('row_id').season_v2.reindex(va.row_id).to_numpy()
   assert np.array_equal(g.baseline.to_numpy(),ref)
  checked.append(dict(validator=v,fold=k,inner_train_days=len(a[['farm','day']].drop_duplicates()),inner_query_days=len(b[['farm','day']].drop_duplicates())))
 record['checked']=checked;return record

def specialist(core,tr,q,cols,seed,fallback):
 hi=tr.groupby(['farm','day']).sub_ec.transform('mean')>=.8;hs=tr[hi]
 nd=len(hs[['farm','day']].drop_duplicates())
 info=dict(n_days=nd,n_rows=len(hs))
 import hashlib
 info['train_row_sha256']=hashlib.sha256('\n'.join(hs.row_id).encode()).hexdigest()
 if nd<2:return fallback.copy(),None,hs,dict(info,fallback=True,bounds=None)
 model=core.et(seed);model.steps[-1][1].n_jobs=2
 with threadpool_limits(limits=2):model.fit(hs[cols],hs.sub_ec)
 model.steps[-1][1].n_jobs=1;raw=model.predict(q[cols]);sp=np.clip(core.shrink(raw,q),hs.sub_ec.min(),hs.sub_ec.max())
 return sp,model,hs,dict(info,fallback=False,bounds=[float(hs.sub_ec.min()),float(hs.sub_ec.max())])

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');args=parser.parse_args()
 lab,core,wv,folds,outer=S.loadec();assert len(folds)==22
 check=preflight(lab,folds,outer)
 if args.prepare:
  p=H/'preparation_v2.json';assert not p.exists();p.write_text(json.dumps(check,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(check,ensure_ascii=False,indent=2));return
 assert check['status']=='PASS',check
 OUT.mkdir(parents=True,exist_ok=True);sha=S.sha(Path(__file__));p=OUT/'source_sha.txt'
 if p.exists():assert p.read_text()==sha
 else:p.write_text(sha)
 cols=[c for c in core.FULL if c!='day']+['season'];idx=lab.set_index('row_id')
 hm=pd.concat([c[c.validator!='EL1'] for c in pd.read_csv(ROOT/'집/클로드/research/local/ec3_HM1_all.csv',chunksize=2048)],ignore_index=True).set_index(['validator','validation_fold','row_id'])
 audits=[]
 for v,k,tm,vm in folds:
  a,b,z,bag=full_inner(v,k,lab[tm],idx);a,b=S.seasonal(a,b,wv);a,b=a.reset_index(drop=True),b.reset_index(drop=True)
  tr,q=S.seasonal(lab[tm],lab[vm],wv);tr,q=tr.reset_index(drop=True),q.reset_index(drop=True)
  for seed in [7,101,2024]:
   dest=OUT/f'{v}_{k}_{seed}.csv'
   if dest.exists():continue
   r3=np.clip(core.shrink(z[f'r3_{seed}'],b),a.sub_ec.min(),a.sub_ec.max())
   base=np.clip(core.shrink(.8*z[f'r3_{seed}']+.2*bag,b),a.sub_ec.min(),a.sub_ec.max())
   sp,im,ihs,ia=specialist(core,a,b,cols,seed,r3);flag=prefix(r3,b)>=.9;direction=.8*flag*(sp-r3)
   e=base-b.sub_ec.to_numpy();num=math.fsum(float(x)*float(y) for x,y in zip(e,direction))/len(b);den=math.fsum(float(x)**2 for x in direction)/len(b)+LAMBDA
   w=float(np.clip(-num/den,0,.5));wn=float(np.clip(-np.mean(e*direction)/(np.mean(direction**2)+LAMBDA),0,.5));assert abs(w-wn)<1e-12
   grad=2*(num+den*w);assert (w==0 and grad>=-1e-12) or (w==.5 and grad<=1e-12) or abs(grad)<1e-12
   rr=np.clip(core.shrink(original_r3(v,k,seed,tr,q),q),tr.sub_ec.min(),tr.sub_ec.max())
   reference=outer[(outer.validator==v)&(outer.validation_fold==k)&(outer.seed==seed)].set_index('row_id').season_v2.reindex(q.row_id).to_numpy()
   osp,om,ohs,oa=specialist(core,tr,q,cols,seed,rr)
   old=hm.loc[[(v,k,x) for x in q.row_id]];r3gap=float(np.max(abs(rr-old[f'r3s_{seed}'].to_numpy())));assert r3gap<1e-9
   if not oa['fallback']:
    gap=float(np.max(abs(osp-old[f'sp_{seed}'].to_numpy())));assert gap<1e-9 and int(old.n_spec_days.iloc[0])==oa['n_days']
   else:gap=None
   assert np.isfinite(reference).all()
   oflag=prefix(rr,q)>=.9;od=.8*oflag*(osp-rr);pre=reference+w*od;pred=np.clip(pre,tr.sub_ec.min(),tr.sub_ec.max())
   if v=='DIAG10' and k==0 and seed==7:
    for model,train,query in [(im,ihs,b),(om,ohs,q)]:
     if model is None:continue
     fresh=clone(model);fresh.steps[-1][1].n_jobs=2
     with threadpool_limits(limits=2):fresh.fit(train[cols],train.sub_ec)
     fresh.steps[-1][1].n_jobs=1;original=model.predict(query[cols])
     assert np.max(abs(fresh.predict(query[cols])-original))<1e-12 and np.max(abs(model.predict(query.iloc[:8][cols])-original[:8]))<1e-12
     for f in ['F13','F47']:
      for h in [0,6,12]:
       keep=(query.farm==f)&(query.hour<=h);mut=~keep;changed=query.copy();changed.loc[mut,cols]=changed.loc[mut,cols]*17+1000
       nr=model.predict(changed[cols]);assert np.max(abs(nr[keep]-original[keep]))<1e-12
       assert np.max(abs(core.shrink(nr,query)[keep]-core.shrink(original,query)[keep]))<1e-12
     del fresh
    for frame,values in [(b,r3),(q,rr)]:
     pp=prefix(values,frame)
     for _,g in frame.assign(p=values).groupby(['farm','day']):
      hist=[]
      for j,row in g.sort_values('hour').iterrows():hist.append(float(row.p));assert abs(pp[j]-math.fsum(hist)/len(hist))<1e-12
    (H/'first_fold_verification_v1.json').write_text(json.dumps(dict(status='PASS',weight=w,gradient=grad,outer_hm1_maxdiff=gap,outer_r3s_maxdiff=r3gap,replay_batch_prefix_future_checks='PASS'),indent=2),encoding='utf-8')
   d=q[['row_id','farm','day','hour']].copy();d['y']=q.sub_ec;d['baseline']=reference;d['candidate']=pred;d['r3s']=rr;d['specialist']=osp;d['flag']=oflag;d['direction']=od;d['weight']=w;d['validator']=v;d['fold']=k;d['seed']=seed
   assert np.isfinite(d[['y','baseline','candidate','direction','weight']]).all().all();d.to_csv(dest,index=False)
   audits.append(dict(validator=v,fold=k,seed=seed,weight=w,n_inner_query_rows=len(b),n_inner_query_days=len(b[['farm','day']].drop_duplicates()),n_inner_flag_rows=int(flag.sum()),n_outer_flag_rows=int(oflag.sum()),inner=ia,outer=oa,outer_hm1_maxdiff=gap,inner_baseline_mean=float(base.mean()),inner_specialist_mean=float(sp.mean()),outer_baseline_mean=float(reference.mean()),outer_specialist_mean=float(osp.mean()),clipped_rows=int(np.count_nonzero(pre!=pred))))
   print(v,k,seed,'done weight',w,flush=True);del im,om;gc.collect()
 (H/'fit_audit_v1.json').write_text(json.dumps(dict(status='PASS',cells=audits),ensure_ascii=False,indent=2),encoding='utf-8');print('ALL_FITS_COMPLETE',flush=True)

if __name__=='__main__':main()
