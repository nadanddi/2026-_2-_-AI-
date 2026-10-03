from pathlib import Path
import sys,json,gc
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;ROOT=H.parents[3]
sys.path.insert(0,str(ROOT/'집/클로드/research'));import env
sys.path.insert(0,str(ROOT/'집/코덱스/analysis/statistical_experiments_20261003_v1'));import support as S
import numpy as np,pandas as pd
from sklearn.base import clone
from threadpoolctl import threadpool_limits
OUT=ROOT/'집/코덱스/local/ec_log_partition_mean_20261004_v1'
def baseline(tr):
 maps={}
 for f,g in tr.groupby('farm'):
  d=g.groupby('day').agg(s=('season','first'),y=('sub_ec','mean')).sort_values('s')
  b=d.y.rolling(21,center=True,min_periods=5).median().bfill().ffill();maps[f]=(d.s.to_numpy(),b.to_numpy())
 return lambda fr: np.asarray([np.interp(x,*maps[f]) for f,x in zip(fr.farm,fr.season)])
def leaf_arrays(m,xtr,ratio):
 counts=[];means=[]
 for tree in m.steps[-1][1].estimators_:
  leaf=tree.apply(xtr);n=np.bincount(leaf,minlength=tree.tree_.node_count);sm=np.bincount(leaf,weights=ratio,minlength=len(n))
  zz=np.bincount(leaf,weights=np.log(ratio),minlength=len(n));good=n>0
  assert np.max(np.abs(zz[good]/n[good]-tree.tree_.value[good,0,0]))<1e-10
  value=np.divide(sm,n,out=np.zeros_like(sm),where=good);counts.append(n);means.append(value)
 return counts,means
def predict(m,fr,cols,bq,counts,means):
 x=m.steps[0][1].transform(fr[cols]);p=np.zeros(len(fr))
 for tree,n,val in zip(m.steps[-1][1].estimators_,counts,means):
  leaf=tree.apply(x);assert (n[leaf]>0).all();p+=val[leaf]
 p=bq*p/len(means);old=bq*np.exp(m.predict(fr[cols]));assert (p>=old-1e-10).all();return p,old
def main():
 OUT.mkdir(parents=True,exist_ok=True);sha=S.sha(Path(__file__));sp=OUT/'source_sha.txt'
 if sp.exists():assert sp.read_text()==sha
 else:sp.write_text(sha)
 lab,core,wv,folds,outer=S.loadec();cols=[c for c in core.FULL if c!='day']+['season'];assert len(cols)==38
 cache=pd.read_csv(ROOT/'집/클로드/research/local/ec3_DI1_all.csv',float_precision='round_trip').set_index(['validator','validation_fold','row_id'])
 lcache=pd.read_csv(ROOT/'집/클로드/research/local/ec3_LR1_all.csv',float_precision='round_trip');lcache=lcache[lcache.validator!='EL1'].set_index(['validator','validation_fold','row_id'])
 audits=[]
 for v,k,tm,vm in folds:
  tr,va=S.seasonal(lab[tm],lab[vm],wv);tr=tr.reset_index(drop=True);va=va.reset_index(drop=True)
  b=baseline(tr);btr,bq=b(tr),b(va);ratio=tr.sub_ec.to_numpy()/btr
  assert np.isfinite(ratio).all() and (ratio>0).all() and (bq>0).all()
  for seed in [7,101,2024]:
   path=OUT/f'{v}_{k}_{seed}.csv'
   if path.exists():continue
   m=core.et(seed);m.steps[-1][1].n_jobs=2
   with threadpool_limits(limits=2):m.fit(tr[cols],np.log(ratio))
   m.steps[-1][1].n_jobs=1;xtr=m.steps[0][1].transform(tr[cols]);counts,means=leaf_arrays(m,xtr,ratio);raw,old_log=predict(m,va,cols,bq,counts,means)
   old=cache.loc[[(v,k,r) for r in va.row_id],f'etS_{seed}'].to_numpy();ref=outer[(outer.validator==v)&(outer.validation_fold==k)&(outer.seed==seed)].set_index('row_id').season_v2.reindex(va.row_id).to_numpy()
   lr=lcache.loc[[(v,k,r) for r in va.row_id],f'etL_{seed}'].to_numpy();lr_diff=float(np.max(np.abs(np.clip(core.shrink(old_log,va),tr.sub_ec.min(),tr.sub_ec.max())-lr)));assert lr_diff<1e-9
   if v=='DIAG10' and k==0 and seed==7:
    original=core.et(seed);original.steps[-1][1].n_jobs=2
    with threadpool_limits(limits=2):op=core.predict_model(original,tr,va,cols)
    old_diff=float(np.max(np.abs(core.shrink(op,va)-old)));assert old_diff<1e-9
    fresh=clone(m);fresh.steps[-1][1].n_jobs=2
    with threadpool_limits(limits=2):fresh.fit(tr[cols],np.log(ratio))
    fresh.steps[-1][1].n_jobs=1;cc,vv=leaf_arrays(fresh,fresh.steps[0][1].transform(tr[cols]),ratio);rr,_=predict(fresh,va,cols,bq,cc,vv);replay=float(np.max(np.abs(rr-raw)));assert replay<1e-12
    small,_=predict(m,va.iloc[:8],cols,bq[:8],counts,means);batch=float(np.max(np.abs(small-raw[:8])));assert batch<1e-12
    causal=[]
    for farm in ['F13','F47']:
     for hour in [0,6,12]:
      mask=(va.farm==farm)&(va.hour<=hour);mutate=(va.farm!=farm)|((va.farm==farm)&(va.hour>hour));changed=va.copy();changed.loc[mutate,cols]=changed.loc[mutate,cols]*17+1000
      nr,_=predict(m,changed,cols,bq,counts,means);assert np.max(np.abs(nr[mask]-raw[mask]))<1e-12;assert np.max(np.abs(core.shrink(nr,va)[mask]-core.shrink(raw,va)[mask]))<1e-12;causal.append(dict(farm=farm,hour=hour,n=int(mask.sum()),status='PASS'))
    assert abs(np.mean([1.,3.])-2.)<1e-12 and abs(np.exp(np.mean(np.log([1.,3.])))-np.sqrt(3))<1e-12
    (H/'first_fold_verification_v1.json').write_text(json.dumps(dict(status='PASS',original_et_maxdiff=old_diff,original_LR1_maxdiff=lr_diff,replay_maxdiff=replay,batch_maxdiff=batch,jensen_min_increase=float((raw-old_log).min()),causal=causal),indent=2),encoding='utf-8');del original,fresh;gc.collect()
   pred=np.clip(ref+.48*(core.shrink(raw,va)-old),tr.sub_ec.min(),tr.sub_ec.max())
   d=va[['row_id','farm','day','hour']].copy();d['y']=va.sub_ec;d['baseline']=ref;d['candidate']=pred;d['new_et_raw']=raw;d['old_log_raw']=old_log;d['seasonal_baseline']=bq;d['old_et_shrunk']=old;d['validator']=v;d['fold']=k;d['seed']=seed
   assert np.isfinite(d[['y','baseline','candidate']]).all().all();d.to_csv(path,index=False);audits.append(dict(validator=v,fold=k,seed=seed,original_LR1_maxdiff=lr_diff,jensen_min_increase=float((raw-old_log).min())));print(v,k,seed,'done',flush=True);del m,counts,means;gc.collect()
 (H/'fit_audit_v1.json').write_text(json.dumps(dict(status='PASS',cells=audits),indent=2),encoding='utf-8');print('ALL_FITS_COMPLETE',flush=True)
if __name__=='__main__':main()
