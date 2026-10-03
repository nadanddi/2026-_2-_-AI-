from pathlib import Path
import sys,json,hashlib,math
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;ROOT=H.parents[3]
sys.path.insert(0,str(ROOT/'집/클로드/research'));import env
sys.path.insert(0,str(ROOT/'집/코덱스/analysis/statistical_experiments_20261003_v1'));import support as S
import numpy as np,pandas as pd
from sklearn.base import clone
from threadpoolctl import threadpool_limits
OUT=ROOT/'집/코덱스/local/ec_joint_level_target_20261003_v1'
def main():
 OUT.mkdir(parents=True,exist_ok=True);sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();p=OUT/'source_sha.txt'
 if p.exists():assert p.read_text()==sha
 else:p.write_text(sha)
 lab,core,wv,folds,outer=S.loadec();cols=[c for c in core.FULL if c!='day']+['season']
 oldcache=pd.read_csv(ROOT/'집/클로드/research/local/ec3_DI1_all.csv',float_precision='round_trip').set_index(['validator','validation_fold','row_id']);audits=[]
 for v,k,tm,vm in folds:
  tr,va=S.seasonal(lab[tm],lab[vm],wv);tr=tr.reset_index(drop=True);va=va.reset_index(drop=True)
  assert set(tr.row_id).isdisjoint(va.row_id)
  grouped=tr.groupby(['farm','day']);assert (grouped.size()==24).all()
  daily=grouped.sub_ec.transform('mean').to_numpy();means=grouped.sub_ec.mean();checks=0
  for key,g in grouped:
   assert abs(means.loc[key]-math.fsum(float(x) for x in g.sub_ec)/24)<1e-12;checks+=1
  targets=np.column_stack([tr.sub_ec.to_numpy(),daily]);assert targets.shape==(len(tr),2)
  for seed in [7,101,2024]:
   path=OUT/f'{v}_{k}_{seed}.csv'
   if path.exists():continue
   ref=outer[(outer.validator==v)&(outer.validation_fold==k)&(outer.seed==seed)].set_index('row_id').season_v2.reindex(va.row_id).to_numpy();old=oldcache.loc[[(v,k,r) for r in va.row_id],f'etS_{seed}'].to_numpy()
   model=core.et(seed);model.steps[-1][1].n_jobs=2
   with threadpool_limits(limits=2):
    model.fit(tr[cols],targets);model.steps[-1][1].n_jobs=1;multi=model.predict(va[cols]);raw=multi[:,0]
    batch=float(np.max(np.abs(raw[:8]-model.predict(va.iloc[:8][cols])[:,0])));assert batch<1e-12
    if v=='DIAG10' and k==0 and seed==7:
     fresh=clone(model);fresh.steps[-1][1].n_jobs=2;fresh.fit(tr[cols],targets);fresh.steps[-1][1].n_jobs=1;repeat=fresh.predict(va[cols])[:,0];diff=float(np.max(np.abs(raw-repeat)));assert diff<1e-12
     original=core.et(seed);original.steps[-1][1].n_jobs=2;unmodified=core.predict_model(original,tr,va,cols);old_diff=float(np.max(np.abs(core.shrink(unmodified,va)-old)));assert old_diff<1e-9
     x=model.steps[0][1].transform(va[cols]);tree=np.stack([t.predict(x)[:,0] for t in model.steps[-1][1].estimators_]);assert np.max(np.abs(tree.mean(axis=0)-raw))<1e-12
     causal=[]
     for farm in ['F13','F47']:
      for hour in [0,6,12]:
       mask=(va.farm==farm)&(va.hour<=hour);changed=va.copy();mutate=(va.farm!=farm)|((va.farm==farm)&(va.hour>hour));changed.loc[mutate,cols]=changed.loc[mutate,cols]*17+1000
       new=model.predict(changed[cols])[:,0];assert np.array_equal(raw[mask],new[mask]);assert np.max(np.abs(core.shrink(raw,va)[mask]-core.shrink(new,va)[mask]))<1e-12
       causal.append(dict(farm=farm,hour=hour,rows=int(mask.sum()),status='PASS'))
     (H/'first_fold_verification_v1.json').write_text(json.dumps(dict(status='PASS',repeat_maxdiff=diff,batch_maxdiff=batch,original_et_maxdiff=old_diff,target_mean_checks=checks,tree_mean_maxdiff=float(np.max(np.abs(tree.mean(axis=0)-raw))),causal=causal),indent=2),encoding='utf-8')
     np.savez(OUT/'first_fold_tree_predictions.npz',row_id=va.row_id.to_numpy(str),tree_predictions=tree,current_prediction=raw,mean_prediction=multi[:,1])
   pred=np.clip(ref+.48*(core.shrink(raw,va)-old),tr.sub_ec.min(),tr.sub_ec.max())
   d=va[['row_id','farm','day','hour']].copy();d['y']=va.sub_ec;d['baseline']=ref;d['candidate']=pred;d['new_et_raw']=raw;d['day_mean_output']=multi[:,1];d['old_et_shrunk']=old;d['validator']=v;d['fold']=k;d['seed']=seed;d['clip_lo']=tr.sub_ec.min();d['clip_hi']=tr.sub_ec.max()
   assert np.isfinite(d[['y','baseline','candidate']]).all().all();d.to_csv(path,index=False);audits.append(dict(validator=v,fold=k,seed=seed,train_days=checks,batch_maxdiff=batch));print(v,k,seed,'done',flush=True)
 (H/'target_audit_v1.json').write_text(json.dumps(dict(status='PASS',cells=audits),indent=2),encoding='utf-8');print('ALL_FITS_COMPLETE',flush=True)
if __name__=='__main__':main()
