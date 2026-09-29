"""One prespecified cold-day sample-weight experiment for the EC ET member."""
import importlib.util
import json
import os
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve()
ROOT = HERE.parents[2]
CV = HERE.parents[1] / 'ec_three_seed_ensemble_cv'
spec = importlib.util.spec_from_file_location('ec_cv_previous', CV/'run.py')
cv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cv)
bag, core, previous, env = cv.bag, cv.core, cv.previous, cv.env
np, pd = cv.np, cv.pd
SEARCH = ROOT/'local/ec_three_seed_ensemble_cv/20260928_035914'


def main():
    out=ROOT/'local/ec_cold_weight_et'/datetime.now().strftime('%Y%m%d_%H%M%S')
    out.mkdir(parents=True,exist_ok=False)
    manifest=dict(protocol_sha=core.sha(HERE.with_name('PROTOCOL.md')),
                  code_sha=core.sha(HERE),splits_sha=core.sha(bag.SPLITS),
                  input_sha={n:core.sha(env.DATA/n) for n in ('train_X.csv','train_y.csv')},
                  folds=(0,2,4,6),seeds=(7,101,2024),
                  threshold_midnight_in_temp=10,weight_cold=2,
                  baseline='EC v2 three-seed cached members and four TabPFN contexts')
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    dev=bag.prepare()
    scores=[]
    all_rows=[]
    for fold in manifest['folds']:
        trm,vam=core.split(dev,fold)
        tr,va=dev[trm],dev[vam].reset_index(drop=True)
        cold=tr.in_temp_h0.le(10).fillna(False).to_numpy(bool)
        weights=np.where(cold,2.,1.)
        n_cold_days=int(tr.loc[cold,['farm','day']].drop_duplicates().shape[0])
        assert n_cold_days>0
        saved=pd.read_csv(SEARCH/f'fold{fold}.csv')
        bag_dir=cv.OLD if fold in (0,2) else cv.NEW
        bag_saved=pd.read_csv(bag_dir/f'fold{fold}_bag.csv')
        third_saved=pd.read_csv(cv.THIRD/f'fold{fold}.csv')
        assert saved.row_id.tolist()==va.row_id.tolist()
        assert bag_saved.row_id.tolist()==va.row_id.tolist()
        assert third_saved.row_id.tolist()==va.row_id.tolist()
        for seed in manifest['seeds']:
            cache=ROOT/'local/ec_baseline_cache'/(previous.cache_key(tr,va,seed)+'.npz')
            assert cache.is_file(),f'missing baseline cache {fold}/{seed}'
            with np.load(cache) as z:
                assert np.array_equal(z['row_id'],va.row_id.to_numpy(str))
                et,lgb,mlp=[z[k].copy() for k in ('et','lgb','mlp')]
            # Verify exact reference identity before evaluating the changed member.
            original=np.clip(core.shrink(.8*(.6*et+.3*lgb+.1*mlp)+.2*bag_saved.bag.to_numpy(float),va),
                             tr.sub_ec.min(),tr.sub_ec.max())
            reference=(third_saved.blend if seed==2024 else bag_saved[f'blend_{seed}']).to_numpy(float)
            np.testing.assert_allclose(original,reference,rtol=0,atol=1e-12)
            weighted_et=core.predict_model(core.et(seed),tr,va,core.FULL,weight=weights)
            raw=.6*weighted_et+.3*lgb+.1*mlp
            candidate=np.clip(core.shrink(.8*raw+.2*bag_saved.bag.to_numpy(float),va),
                              tr.sub_ec.min(),tr.sub_ec.max())
            old,new=cv.rmse(va.sub_ec,original),cv.rmse(va.sub_ec,candidate)
            rec=dict(fold=fold,seed=seed,baseline=old,candidate=new,relative_change=new/old-1,
                     n_train_days=int(len(tr)/24),n_cold_train_days=n_cold_days)
            scores.append(rec)
            row=va[['row_id','farm','day','hour','sub_ec']].copy()
            row['fold']=fold;row['seed']=seed;row['baseline']=original;row['candidate']=candidate
            all_rows.append(row)
            (out/f'fold{fold}_seed{seed}.csv').write_text(row.to_csv(index=False,float_format='%.17g'),encoding='utf-8')
            (out/'progress.json').write_text(json.dumps(scores,ensure_ascii=False,indent=2),encoding='utf-8')
            print(json.dumps(rec),flush=True)
    all_rows=pd.concat(all_rows,ignore_index=True)
    # Each seed has the same validation rows; calculate per-seed and farm.
    by_farm={}
    for farm,g in all_rows.groupby('farm'):
        a,b=cv.rmse(g.sub_ec,g.baseline),cv.rmse(g.sub_ec,g.candidate)
        by_farm[farm]=dict(baseline=a,candidate=b,relative_change=b/a-1)
    a,b=cv.rmse(all_rows.sub_ec,all_rows.baseline),cv.rmse(all_rows.sub_ec,all_rows.candidate)
    result=dict(scores=scores,overall=dict(baseline=a,candidate=b,relative_change=b/a-1),
                by_farm=by_farm,passes_search=bool(all(z['relative_change']<0 for z in scores)
                  and b/a-1<=-.01 and all(z['relative_change']<0 for z in by_farm.values())),
                confirmation_scored=False,hidden_labels_read=False,no_submission_file_created=True)
    (out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='scores'},ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    with bag.threadpool_limits(limits=4):
        main()
