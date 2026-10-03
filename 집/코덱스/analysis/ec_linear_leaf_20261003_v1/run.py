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
from lightgbm import LGBMRegressor
from threadpoolctl import threadpool_limits
OUT=ROOT/'집/코덱스/local/ec_linear_leaf_20261003_v1'
def model(seed,linear):
    return make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler(),LGBMRegressor(objective='regression',n_estimators=400,learning_rate=.035,num_leaves=15,min_child_samples=100,reg_lambda=15,max_bin=127,subsample=.8,subsample_freq=1,colsample_bytree=.8,deterministic=True,force_col_wise=True,n_jobs=2,random_state=seed,linear_tree=linear,linear_lambda=10,verbosity=-1))
def main():
    OUT.mkdir(parents=True,exist_ok=True);sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();tag=OUT/'source_sha.txt'
    if tag.exists():assert tag.read_text()==sha
    else:tag.write_text(sha)
    lab,core,wv,folds,outer=S.loadec();cols=[c for c in core.FULL if c!='day']+['season']
    for arm,linear in [('GB_CONST20',False),('GB_LINEAR20',True)]:
        dest=OUT/arm;dest.mkdir(exist_ok=True);(H/arm).mkdir(exist_ok=True)
        for v,k,tm,vm in folds:
            tr,va=S.seasonal(lab[tm],lab[vm],wv);tr=tr.reset_index(drop=True);va=va.reset_index(drop=True)
            assert set(tr.row_id).isdisjoint(va.row_id)
            for seed in [7,101,2024]:
                path=dest/f'{v}_{k}_{seed}.csv'
                if path.exists():continue
                ref=outer[(outer.validator==v)&(outer.validation_fold==k)&(outer.seed==seed)].set_index('row_id').season_v2.reindex(va.row_id).to_numpy()
                with threadpool_limits(limits=2):
                    m=model(seed,linear);m.fit(tr[cols],tr.sub_ec);raw=m.predict(va[cols])
                    if v=='DIAG10' and k==0 and seed==7:
                        again=model(seed,linear);again.fit(tr[cols],tr.sub_ec);repeat=again.predict(va[cols]);diff=float(np.max(np.abs(raw-repeat)));assert diff<1e-12
                        dump=m.steps[-1][1].booster_.dump_model()
                        def leaves(t):return [t] if 'leaf_index' in t else leaves(t['left_child'])+leaves(t['right_child'])
                        count=sum(len(x.get('leaf_coeff',[]))>0 for tree in dump['tree_info'] for x in leaves(tree['tree_structure']))
                        assert (count>0)==linear
                        (H/arm/'reproduction_v1.json').write_text(json.dumps(dict(status='PASS',maxdiff=diff,linear_leaf_count=count)),encoding='utf-8')
                pred=np.clip(.8*ref+.2*core.shrink(raw,va),tr.sub_ec.min(),tr.sub_ec.max())
                a=va[['row_id','farm','day','hour']].copy();a['y']=va.sub_ec;a['baseline']=ref;a['candidate']=pred;a['new_et_raw']=raw;a['validator']=v;a['fold']=k;a['seed']=seed;a['clip_lo']=tr.sub_ec.min();a['clip_hi']=tr.sub_ec.max()
                assert np.isfinite(a[['y','baseline','candidate']]).all().all();a.to_csv(path,index=False);print(arm,v,k,seed,'done',flush=True)
    print('ALL_FITS_COMPLETE',flush=True)
if __name__=='__main__':main()
