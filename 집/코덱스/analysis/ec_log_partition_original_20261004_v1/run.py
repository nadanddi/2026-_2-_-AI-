from pathlib import Path
import sys,json,math,importlib.util,gc
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;ROOT=H.parents[3]
sys.path.insert(0,str(ROOT/'집/클로드/research'));import env
sp=importlib.util.spec_from_file_location('original_leaf_log_reference',H.parent/'ec_log_partition_mean_20261004_v1/run.py')
L=importlib.util.module_from_spec(sp);sp.loader.exec_module(L)
import numpy as np,pandas as pd
from sklearn.base import clone
from threadpoolctl import threadpool_limits
OUT=ROOT/'집/코덱스/local/ec_log_partition_original_20261004_v1'

def values(model,xt,y):
 counts=[];means=[]
 for tree in model.steps[-1][1].estimators_:
  leaf=tree.apply(xt);n=np.bincount(leaf,minlength=tree.tree_.node_count);sm=np.bincount(leaf,weights=y,minlength=len(n));good=n>0
  val=np.divide(sm,n,out=np.zeros_like(sm),where=good)
  assert np.max(abs(np.bincount(leaf,weights=y-val[leaf],minlength=len(n))[good]))<1e-9
  counts.append(n);means.append(val)
 return counts,means

def predict(model,q,cols,counts,means):
 x=model.steps[0][1].transform(q[cols]);pred=np.zeros(len(q))
 for tree,n,mean in zip(model.steps[-1][1].estimators_,counts,means):
  leaf=tree.apply(x);assert (n[leaf]>0).all();pred+=mean[leaf]
 return pred/len(means)

def main():
 OUT.mkdir(parents=True,exist_ok=True);sha=L.S.sha(H/'run.py');p=OUT/'source_sha.txt'
 if p.exists():assert p.read_text()==sha
 else:p.write_text(sha)
 lab,core,wv,folds,outer=L.S.loadec();cols=[c for c in core.FULL if c!='day']+['season'];assert len(cols)==38
 oldcache=pd.read_csv(ROOT/'집/클로드/research/local/ec3_DI1_all.csv',float_precision='round_trip').set_index(['validator','validation_fold','row_id'])
 audit=[]
 for v,k,tm,vm in folds:
  tr,q=L.S.seasonal(lab[tm],lab[vm],wv);tr,q=tr.reset_index(drop=True),q.reset_index(drop=True)
  b=L.baseline(tr);bt,bq=b(tr),b(q);ratio=tr.sub_ec.to_numpy()/bt
  assert np.isfinite(ratio).all() and (ratio>0).all()
  for seed in [7,101,2024]:
   dest=OUT/f'{v}_{k}_{seed}.csv'
   if dest.exists():continue
   model=core.et(seed);model.steps[-1][1].n_jobs=2;assert not model.steps[-1][1].bootstrap
   with threadpool_limits(limits=2):model.fit(tr[cols],np.log(ratio))
   model.steps[-1][1].n_jobs=1;xt=model.steps[0][1].transform(tr[cols]);counts,means=values(model,xt,tr.sub_ec.to_numpy())
   raw=predict(model,q,cols,counts,means);assert (raw>=tr.sub_ec.min()-1e-12).all() and (raw<=tr.sub_ec.max()+1e-12).all()
   cc,rr=L.leaf_arrays(model,xt,ratio);repro,_=L.predict(model,q,cols,bq,cc,rr)
   previous=pd.read_csv(L.OUT/f'{v}_{k}_{seed}.csv',float_precision='round_trip').set_index('row_id').reindex(q.row_id)
   priorgap=float(np.max(abs(repro-previous.new_et_raw.to_numpy())));assert priorgap<1e-9
   old=oldcache.loc[[(v,k,x) for x in q.row_id],f'etS_{seed}'].to_numpy()
   ref=outer[(outer.validator==v)&(outer.validation_fold==k)&(outer.seed==seed)].set_index('row_id').season_v2.reindex(q.row_id).to_numpy()
   if v=='DIAG10' and k==0 and seed==7:
    orig=core.et(seed);orig.steps[-1][1].n_jobs=2
    with threadpool_limits(limits=2):op=core.predict_model(orig,tr,q,cols)
    oldgap=float(np.max(abs(core.shrink(op,q)-old)));assert oldgap<1e-9
    manual=[]
    for tree,n,mean in list(zip(model.steps[-1][1].estimators_,counts,means))[:3]:
     leaf=tree.apply(xt);j=tree.apply(model.steps[0][1].transform(q.iloc[:1][cols]))[0];ix=np.flatnonzero(leaf==j)
     m=math.fsum(float(tr.sub_ec.iloc[t]) for t in ix)/len(ix);assert abs(m-mean[j])<1e-12;manual.append(dict(n=len(ix),mean=m))
    fresh=clone(model);fresh.steps[-1][1].n_jobs=2
    with threadpool_limits(limits=2):fresh.fit(tr[cols],np.log(ratio))
    fresh.steps[-1][1].n_jobs=1;fc,fm=values(fresh,fresh.steps[0][1].transform(tr[cols]),tr.sub_ec.to_numpy())
    replay=float(np.max(abs(predict(fresh,q,cols,fc,fm)-raw)));batch=float(np.max(abs(predict(model,q.iloc[:8],cols,counts,means)-raw[:8])));assert replay<1e-12 and batch<1e-12
    causal=[]
    for f in ['F13','F47']:
     for hour in [0,6,12]:
      keep=(q.farm==f)&(q.hour<=hour);changed=q.copy();changed.loc[~keep,cols]=changed.loc[~keep,cols]*17+1000
      nr=predict(model,changed,cols,counts,means);assert np.max(abs(nr[keep]-raw[keep]))<1e-12;assert np.max(abs(core.shrink(nr,q)[keep]-core.shrink(raw,q)[keep]))<1e-12
      causal.append(dict(farm=f,hour=hour,status='PASS'))
    (H/'first_fold_verification_v1.json').write_text(json.dumps(dict(status='PASS',original_et_maxdiff=oldgap,prior_partition_maxdiff=priorgap,manual_leaf_checks=manual,replay_maxdiff=replay,batch_maxdiff=batch,causal=causal),indent=2),encoding='utf-8')
    del orig,fresh;gc.collect()
   pred=np.clip(ref+.48*(core.shrink(raw,q)-old),tr.sub_ec.min(),tr.sub_ec.max())
   d=q[['row_id','farm','day','hour']].copy();d['y']=q.sub_ec;d['baseline']=ref;d['candidate']=pred;d['new_et_raw']=raw;d['old_et_shrunk']=old;d['prior_ratio_raw']=repro;d['validator']=v;d['fold']=k;d['seed']=seed
   assert np.isfinite(d[['y','baseline','candidate']]).all().all();d.to_csv(dest,index=False)
   audit.append(dict(validator=v,fold=k,seed=seed,prior_partition_maxdiff=priorgap,n_train=len(tr),n_valid=len(q)));print(v,k,seed,'done',flush=True)
   del model,counts,means,cc,rr;gc.collect()
 (H/'fit_audit_v1.json').write_text(json.dumps(dict(status='PASS',cells=audit,reference_source_sha256=L.S.sha(L.H/'run.py')),indent=2),encoding='utf-8');print('ALL_FITS_COMPLETE',flush=True)

if __name__=='__main__':main()
