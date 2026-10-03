from pathlib import Path
import sys,json,hashlib
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;ROOT=H.parents[3]
sys.path.insert(0,str(ROOT/'집/클로드/research'))
import env
sys.path.insert(0,str(ROOT/'집/코덱스/analysis/statistical_experiments_20261003_v1'))
import support as S
import numpy as np,pandas as pd
from threadpoolctl import threadpool_limits
OUT=ROOT/'집/코덱스/local/ec_power_partition_20261003_v1'
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();p=OUT/'source_sha.txt'
    if p.exists():assert p.read_text()==sha
    else:p.write_text(sha)
    lab,core,wv,folds,outer=S.loadec()
    oldcache=pd.read_csv(ROOT/'집/클로드/research/local/ec3_DI1_all.csv',float_precision='round_trip').set_index(['validator','validation_fold','row_id'])
    cols=[c for c in core.FULL if c!='day']+['season'];maxj=0.
    for v,k,tm,vm in folds:
        tr,va=S.seasonal(lab[tm],lab[vm],wv);tr=tr.reset_index(drop=True);va=va.reset_index(drop=True)
        assert set(tr.row_id).isdisjoint(va.row_id)
        for seed in [7,101,2024]:
            path=OUT/f'{v}_{k}_{seed}.csv'
            if path.exists():continue
            ref=outer[(outer.validator==v)&(outer.validation_fold==k)&(outer.seed==seed)].set_index('row_id').season_v2.reindex(va.row_id).to_numpy()
            old=oldcache.loc[[(v,k,r) for r in va.row_id],f'etS_{seed}'].to_numpy()
            with threadpool_limits(limits=2):
                model=core.et(seed);model.steps[-1][1].n_jobs=2
                power=tr.copy();power['sub_ec']=power.sub_ec**2
                rms=np.sqrt(np.maximum(core.predict_model(model,power,va,cols),0))
                x=model.steps[0][1].transform(va[cols]);acc=np.zeros(len(va))
                for tree in model.steps[-1][1].estimators_:acc+=np.sqrt(np.maximum(tree.predict(x),0))
                raw=acc/len(model.steps[-1][1].estimators_)
                assert np.all(rms+1e-12>=raw)
                maxj=max(maxj,float(np.max(rms-raw)))
            pred=np.clip(ref+.48*(core.shrink(raw,va)-old),tr.sub_ec.min(),tr.sub_ec.max())
            a=va[['row_id','farm','day','hour']].copy();a['y']=va.sub_ec;a['baseline']=ref;a['candidate']=pred;a['new_et_raw']=raw;a['old_et_shrunk']=old;a['rms_et_raw']=rms;a['validator']=v;a['fold']=k;a['seed']=seed
            assert np.isfinite(a[['y','baseline','candidate']]).all().all();a.to_csv(path,index=False)
            print(v,k,seed,'done Jensen max',maxj,flush=True)
    (H/'Jensen_audit_v1.json').write_text(json.dumps(dict(status='PASS',max_gap=maxj,fold_seed_cells=66)),encoding='utf-8')
    print('ALL_FITS_COMPLETE',flush=True)
if __name__=='__main__':main()
