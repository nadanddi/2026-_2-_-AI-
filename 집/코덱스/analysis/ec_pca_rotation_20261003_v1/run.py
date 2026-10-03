from pathlib import Path
import sys,json,hashlib
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;ROOT=H.parents[3]
sys.path.insert(0,str(ROOT/'집/클로드/research'))
import env
sys.path.insert(0,str(ROOT/'집/코덱스/analysis/statistical_experiments_20261003_v1'))
import support as S
import numpy as np,pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.base import clone
from threadpoolctl import threadpool_limits
import joblib
OUT=ROOT/'집/코덱스/local/ec_pca_rotation_20261003_v1'
def main():
 OUT.mkdir(parents=True,exist_ok=True)
 sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();p=OUT/'source_sha.txt'
 if p.exists():assert p.read_text()==sha
 else:p.write_text(sha)
 lab,core,wv,folds,outer=S.loadec()
 oldcache=pd.read_csv(ROOT/'집/클로드/research/local/ec3_DI1_all.csv',float_precision='round_trip').set_index(['validator','validation_fold','row_id'])
 cols=[c for c in core.FULL if c!='day']+['season'];audit=[]
 for v,k,tm,vm in folds:
  tr,va=S.seasonal(lab[tm],lab[vm],wv);tr=tr.reset_index(drop=True);va=va.reset_index(drop=True)
  assert set(tr.row_id).isdisjoint(va.row_id)
  for seed in [7,101,2024]:
   path=OUT/f'{v}_{k}_{seed}.csv'
   if path.exists():continue
   ref=outer[(outer.validator==v)&(outer.validation_fold==k)&(outer.seed==seed)].set_index('row_id').season_v2.reindex(va.row_id).to_numpy()
   old=oldcache.loc[[(v,k,r) for r in va.row_id],f'etS_{seed}'].to_numpy()
   et=core.et(seed).steps[-1][1];et.n_jobs=2
   model=make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),PCA(n_components=None,svd_solver='full',whiten=False),et)
   with threadpool_limits(limits=2):
    raw=core.predict_model(model,tr,va,cols)
    im,sc,pc=[s[1] for s in model.steps[:3]];x=sc.transform(im.transform(tr[cols]));z=pc.transform(x)
    assert z.shape[1]==len(cols)==38
    inverse=float(np.max(np.abs(pc.inverse_transform(z)-x)));orth=float(np.max(np.abs(pc.components_@pc.components_.T-np.eye(38))))
    assert inverse<1e-10 and orth<1e-10
    batch=float(np.max(np.abs(model.predict(va.iloc[:8][cols])-raw[:8])));assert batch<1e-12
    if v=='DIAG10' and k==0 and seed==7:
     fresh=clone(model);again=core.predict_model(fresh,tr,va,cols);diff=float(np.max(np.abs(again-raw)));assert diff<1e-12
     joblib.dump(model,OUT/'first_fold_model.joblib')
     causal=[]
     for farm in ['F13','F47']:
      for hour in [0,6,12]:
       mask=(va.farm==farm)&(va.hour<=hour);changed=va.copy();mutate=(va.farm!=farm)|((va.farm==farm)&(va.hour>hour))
       changed.loc[mutate,cols]=changed.loc[mutate,cols]*17+1000
       new=model.predict(changed[cols]);assert np.array_equal(raw[mask],new[mask])
       assert np.max(np.abs(core.shrink(raw,va)[mask]-core.shrink(new,va)[mask]))<1e-12
       causal.append(dict(farm=farm,hour=hour,rows=int(mask.sum()),status='PASS'))
     (H/'first_fold_verification_v1.json').write_text(json.dumps(dict(status='PASS',repeat_maxdiff=diff,batch_maxdiff=batch,inverse_maxdiff=inverse,orthogonality_maxdiff=orth,causal=causal,scope='first fold seed7 frozen query transform; all38 principal axes retained'),indent=2),encoding='utf-8')
   pred=np.clip(ref+.48*(core.shrink(raw,va)-old),tr.sub_ec.min(),tr.sub_ec.max())
   a=va[['row_id','farm','day','hour']].copy();a['y']=va.sub_ec;a['baseline']=ref;a['candidate']=pred;a['new_et_raw']=raw;a['old_et_shrunk']=old;a['validator']=v;a['fold']=k;a['seed']=seed;a['clip_lo']=tr.sub_ec.min();a['clip_hi']=tr.sub_ec.max()
   assert np.isfinite(a[['y','baseline','candidate']]).all().all();a.to_csv(path,index=False)
   audit.append(dict(validator=v,fold=k,seed=seed,inverse_maxdiff=inverse,orthogonality_maxdiff=orth,batch_maxdiff=batch));print(v,k,seed,'done',flush=True)
 (H/'transform_audit_v1.json').write_text(json.dumps(dict(status='PASS',cells=audit),indent=2),encoding='utf-8');print('ALL_FITS_COMPLETE',flush=True)
if __name__=='__main__':main()
