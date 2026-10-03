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
OUT=ROOT/'집/코덱스/local/ec_tree_median_20261003_v1'
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();p=OUT/'source_sha.txt'
    if p.exists():assert p.read_text()==sha
    else:p.write_text(sha)
    lab,core,wv,folds,outer=S.loadec()
    oldcache=pd.read_csv(ROOT/'집/클로드/research/local/ec3_DI1_all.csv',float_precision='round_trip').set_index(['validator','validation_fold','row_id'])
    cols=[c for c in core.FULL if c!='day']+['season']
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
                rawmean=core.predict_model(model,tr,va,cols)
                q=model.steps[0][1].transform(va[cols]);trees=np.stack([t.predict(q) for t in model.steps[-1][1].estimators_]);raw=np.median(trees,axis=0)
                ordered=np.sort(trees,axis=0);manual=(ordered[299]+ordered[300])/2;assert np.max(np.abs(manual-raw))<1e-12
                assert np.max(np.abs(trees.mean(axis=0)-rawmean))<1e-12
                olddiff=float(np.max(np.abs(core.shrink(rawmean,va)-old)));assert olddiff<1e-9
                if v=='DIAG10' and k==0 and seed==7:
                    np.savez(OUT/'first_fold_tree_predictions.npz',row_id=va.row_id.to_numpy(str),tree_predictions=trees,raw_median=raw,raw_mean=rawmean)
                if v=='DIAG10' and k==0 and seed==7:
                    again=core.et(seed);again.steps[-1][1].n_jobs=2
                    core.predict_model(again,tr,va,cols);repeat=np.median(np.stack([t.predict(again.steps[0][1].transform(va[cols])) for t in again.steps[-1][1].estimators_]),axis=0);diff=float(np.max(np.abs(repeat-raw)));assert diff<1e-12
                    (H/'reproduction_v1.json').write_text(json.dumps(dict(status='PASS',maxdiff=diff)),encoding='utf-8')
            pred=np.clip(ref+.48*(core.shrink(raw,va)-old),tr.sub_ec.min(),tr.sub_ec.max())
            a=va[['row_id','farm','day','hour']].copy();a['y']=va.sub_ec;a['baseline']=ref;a['candidate']=pred;a['new_et_raw']=raw;a['old_et_shrunk']=old;a['validator']=v;a['fold']=k;a['seed']=seed
            a['clip_lo']=tr.sub_ec.min();a['clip_hi']=tr.sub_ec.max()
            a['raw_mean']=rawmean;a['tree_std']=trees.std(axis=0);a['high_vote_fraction']=(trees>=1).mean(axis=0);a['raw_q25']=np.quantile(trees,.25,axis=0);a['raw_q75']=np.quantile(trees,.75,axis=0);a['old_mean_reproduction_maxdiff']=olddiff
            assert np.isfinite(a[['y','baseline','candidate']]).all().all();a.to_csv(path,index=False);print(v,k,seed,'done',flush=True)
    print('ALL_FITS_COMPLETE',flush=True)
if __name__=='__main__':main()
