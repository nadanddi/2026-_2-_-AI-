from pathlib import Path
import sys,json,hashlib,time
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;ROOT=H.parents[3]
sys.path.insert(0,str(ROOT/'집/클로드/research'))
import env
sys.path.insert(0,str(ROOT/'집/코덱스/analysis/statistical_experiments_20261003_v1'))
import support as S
import numpy as np,pandas as pd
from threadpoolctl import threadpool_limits
OUT=ROOT/'집/코덱스/local/ec_power_target_20261003_v1'
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    meta=OUT/'source_sha.txt'
    if meta.exists():assert meta.read_text()==sha
    else:meta.write_text(sha)
    lab,core,wv,folds,outer=S.loadec()
    cache=pd.read_csv(ROOT/'집/클로드/research/local/ec3_DI1_all.csv',float_precision='round_trip').set_index(['validator','validation_fold','row_id'])
    cols=[c for c in core.FULL if c!='day']+['season']
    for v,k,tm,vm in folds:
        tr,va=S.seasonal(lab[tm],lab[vm],wv);tr=tr.reset_index(drop=True);va=va.reset_index(drop=True)
        assert set(tr.row_id).isdisjoint(va.row_id)
        for seed in [7,101,2024]:
            p=OUT/f'{v}_{k}_{seed}.csv'
            if p.exists():continue
            ref=outer[(outer.validator==v)&(outer.validation_fold==k)&(outer.seed==seed)].set_index('row_id').season_v2.reindex(va.row_id).to_numpy()
            old=cache.loc[[(v,k,r) for r in va.row_id],f'etS_{seed}'].to_numpy()
            with threadpool_limits(limits=2):
                if v=='DIAG10' and k==0 and seed==7:
                    model=core.et(seed);model.steps[-1][1].n_jobs=2
                    original=core.predict_model(model,tr,va,cols)
                    diff=float(np.max(np.abs(core.shrink(original,va)-old)));assert diff<1e-10,diff
                    (H/'baseline_reproduction_v1.json').write_text(json.dumps(dict(maxdiff=diff,status='PASS')),encoding='utf-8')
                    print('baseline reproduction',diff,flush=True)
                power=tr.copy();power['sub_ec']=power.sub_ec**2
                model=core.et(seed);model.steps[-1][1].n_jobs=2
                raw=np.sqrt(np.maximum(core.predict_model(model,power,va,cols),0))
                if v=='DIAG10' and k==0 and seed==7:
                    probe=va.iloc[:24].copy();altered=va.copy();altered.loc[altered.hour>6,cols]=99999
                    ix=va.index[(va.farm==va.iloc[0].farm)&(va.day==va.iloc[0].day)&(va.hour<=6)]
                    assert np.array_equal(raw[ix],np.sqrt(np.maximum(model.predict(altered.loc[ix,cols]),0)))
            pred=np.clip(ref+.48*(core.shrink(raw,va)-old),tr.sub_ec.min(),tr.sub_ec.max())
            a=va[['row_id','farm','day','hour']].copy();a['y']=va.sub_ec;a['baseline']=ref;a['candidate']=pred;a['new_et_raw']=raw;a['old_et_shrunk']=old;a['validator']=v;a['fold']=k;a['seed']=seed
            assert np.isfinite(a[['y','baseline','candidate']]).all().all()
            a.to_csv(p,index=False);print(v,k,seed,'done',flush=True)
    print('ALL_FITS_COMPLETE',flush=True)
if __name__=='__main__':main()
