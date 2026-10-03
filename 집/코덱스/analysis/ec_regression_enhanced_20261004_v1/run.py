from pathlib import Path
import sys,json,hashlib,gc
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;ROOT=H.parents[3]
sys.path.insert(0,str(ROOT/'집/클로드/research'));import env
sys.path.insert(0,str(ROOT/'집/코덱스/analysis/statistical_experiments_20261003_v1'));import support as S
import numpy as np,pandas as pd
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from threadpoolctl import threadpool_limits
OUT=ROOT/'집/코덱스/local/ec_regression_enhanced_20261004_v1'
def main():
 OUT.mkdir(parents=True,exist_ok=True);sha=S.sha(Path(__file__));sp=OUT/'source_sha.txt'
 if sp.exists():assert sp.read_text()==sha
 else:sp.write_text(sha)
 lab,core,wv,folds,outer=S.loadec();cols=[c for c in core.FULL if c!='day']+['season']
 assert len(cols)==38 and len(lab)==8640
 cache=pd.read_csv(ROOT/'집/클로드/research/local/ec3_DI1_all.csv',float_precision='round_trip').set_index(['validator','validation_fold','row_id'])
 audits=[]
 for v,k,tm,vm in folds:
  tr,va=S.seasonal(lab[tm],lab[vm],wv);tr=tr.reset_index(drop=True);va=va.reset_index(drop=True)
  assert set(tr.row_id).isdisjoint(va.row_id)
  lin=make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),Ridge(alpha=100.))
  with threadpool_limits(limits=2):lin.fit(tr[cols],tr.sub_ec)
  lp=lin.predict(va[cols]);res=tr.sub_ec.to_numpy()-lin.predict(tr[cols])
  for seed in [7,101,2024]:
   path=OUT/f'{v}_{k}_{seed}.csv'
   if path.exists():continue
   model=core.et(seed);model.steps[-1][1].n_jobs=2
   with threadpool_limits(limits=2):
    model.fit(tr[cols],res);model.steps[-1][1].n_jobs=1;raw=lp+model.predict(va[cols])
   old=cache.loc[[(v,k,r) for r in va.row_id],f'etS_{seed}'].to_numpy()
   ref=outer[(outer.validator==v)&(outer.validation_fold==k)&(outer.seed==seed)].set_index('row_id').season_v2.reindex(va.row_id).to_numpy()
   if v=='DIAG10' and k==0 and seed==7:
    z=lin.steps[1][1].transform(lin.steps[0][1].transform(tr[cols]));yc=tr.sub_ec.to_numpy()-tr.sub_ec.mean()
    beta=np.linalg.solve(z.T@z+100*np.eye(len(cols)),z.T@yc)
    lin_diff=float(np.max(np.abs(beta-lin.steps[-1][1].coef_)));assert lin_diff<1e-10
    original=core.et(seed);original.steps[-1][1].n_jobs=2
    with threadpool_limits(limits=2):op=core.predict_model(original,tr,va,cols)
    old_diff=float(np.max(np.abs(core.shrink(op,va)-old)));assert old_diff<1e-9
    fresh=clone(model);fresh.steps[-1][1].n_jobs=2
    with threadpool_limits(limits=2):fresh.fit(tr[cols],res)
    fresh.steps[-1][1].n_jobs=1;replay=float(np.max(np.abs(raw-(lp+fresh.predict(va[cols])))));assert replay<1e-12
    batch=float(np.max(np.abs(raw[:8]-(lin.predict(va.iloc[:8][cols])+model.predict(va.iloc[:8][cols])))));assert batch<1e-12
    tx=model.steps[0][1].transform(va[cols]);tp=np.stack([t.predict(tx) for t in model.steps[-1][1].estimators_]);tree_diff=float(np.max(np.abs(tp.mean(0)+lp-raw)));assert tree_diff<1e-12
    causal=[]
    for farm in ['F13','F47']:
     for hour in [0,6,12]:
      mask=(va.farm==farm)&(va.hour<=hour);mutate=(va.farm!=farm)|((va.farm==farm)&(va.hour>hour));changed=va.copy();changed.loc[mutate,cols]=changed.loc[mutate,cols]*17+1000
      nr=lin.predict(changed[cols])+model.predict(changed[cols]);assert np.max(np.abs(raw[mask]-nr[mask]))<1e-12;assert np.max(np.abs(core.shrink(raw,va)[mask]-core.shrink(nr,va)[mask]))<1e-12
      causal.append(dict(farm=farm,hour=hour,n=int(mask.sum()),status='PASS'))
    (H/'first_fold_verification_v1.json').write_text(json.dumps(dict(status='PASS',ridge_equation_maxdiff=lin_diff,original_et_maxdiff=old_diff,replay_maxdiff=replay,batch_maxdiff=batch,tree_mean_maxdiff=tree_diff,causal=causal),indent=2),encoding='utf-8')
    del original,fresh,tp;gc.collect()
   pred=np.clip(ref+.48*(core.shrink(raw,va)-old),tr.sub_ec.min(),tr.sub_ec.max())
   d=va[['row_id','farm','day','hour']].copy();d['y']=va.sub_ec;d['baseline']=ref;d['candidate']=pred;d['new_et_raw']=raw;d['linear_trend']=lp;d['residual_et']=raw-lp;d['old_et_shrunk']=old;d['validator']=v;d['fold']=k;d['seed']=seed
   assert np.isfinite(d[['y','baseline','candidate']]).all().all();d.to_csv(path,index=False)
   audits.append(dict(validator=v,fold=k,seed=seed,n_train=len(tr),n_valid=len(va),ridge_coef=lin.steps[-1][1].coef_.tolist()));print(v,k,seed,'done',flush=True)
   del model;gc.collect()
 (H/'fit_audit_v1.json').write_text(json.dumps(dict(status='PASS',cells=audits),indent=2),encoding='utf-8');print('ALL_FITS_COMPLETE',flush=True)
if __name__=='__main__':main()
