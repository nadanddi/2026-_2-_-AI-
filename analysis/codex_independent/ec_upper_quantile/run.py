"""H10 fixed conditional upper-quantile model screen against EC v2."""
import importlib.util
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from threadpoolctl import threadpool_limits


HERE = Path(__file__).resolve().parent
BASE_CODE = HERE.parent / 'ec_dual_view_et/run.py'
spec = importlib.util.spec_from_file_location('ec_h10_base', BASE_CODE)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
LOCAL, FOLDS, SEEDS = base.LOCAL, base.FOLDS, base.SEEDS


def fit_quantile(tr, va, seed):
    model = make_pipeline(SimpleImputer(strategy='median'), GradientBoostingRegressor(
        loss='quantile', alpha=.65, n_estimators=300, learning_rate=.03,
        max_depth=3, min_samples_leaf=40, subsample=.8,
        max_features=.9, random_state=seed))
    model.fit(tr[base.FULL], tr.sub_ec.to_numpy(float))
    q = model.predict(va[base.FULL])
    assert np.isfinite(q).all()
    return q


def main():
    _, lab, dev, lock = base.prepare()
    paths = [(base.SEARCH if f < 8 else base.CONFIRM) / f'fold{f}.csv' for f in FOLDS]
    hashes = {str(p):base.sha(p) for p in [HERE/'PROTOCOL.md', HERE/'run.py',
                                           BASE_CODE, base.DATA/'train_X.csv',
                                           base.DATA/'train_y.csv', base.SPLITS,
                                           base.LOCK, *paths]}
    out = LOCAL/'ec_upper_quantile'/datetime.now().strftime('%Y%m%d_%H%M%S')
    out.mkdir(parents=True, exist_ok=False)
    (out/'manifest.json').write_text(json.dumps(dict(hashes=hashes, folds=FOLDS,
        seeds=SEEDS, quantile=.65, blend=.15, v2_gate=.6),
        ensure_ascii=False, indent=2), encoding='utf-8')
    scores = []
    for fold in FOLDS:
        va, v2 = base.validation(lab, fold)
        val = {(f,int(d)) for f,d in va[['farm','day']].itertuples(index=False,name=None)}
        assert not (val & lock)
        tr = dev[base.split_mask(dev, val | lock)].copy() if fold < 8 else dev[base.split_mask(dev, lock)].copy()
        assert not set(tr.row_id) & set(va.row_id)
        lo, hi = float(tr.sub_ec.min()), float(tr.sub_ec.max())
        gate = v2 >= .6
        for seed in SEEDS:
            q = fit_quantile(tr, va, seed)
            candidate = np.clip(np.where(gate, .85*v2+.15*q, v2), lo, hi)
            row = va[['row_id','farm','day','hour','sub_ec']].copy()
            row['fold'] = fold
            row['seed'] = seed
            row['v2'] = v2
            row['q65'] = q
            row['h10'] = candidate
            row.to_csv(out/f'fold{fold}_seed{seed}.csv', index=False, float_format='%.17g')
            a,b = base.rmse(row.sub_ec,row.v2),base.rmse(row.sub_ec,row.h10)
            rec = dict(fold=fold,seed=seed,train_days=len(tr)//24,
                       gate_rows=int(gate.sum()),v2_rmse=a,h10_rmse=b,
                       relative_change=b/a-1)
            scores.append(rec)
            (out/'progress.json').write_text(json.dumps(scores,ensure_ascii=False,indent=2),encoding='utf-8')
            print(json.dumps(rec),flush=True)
    rows = pd.concat([pd.read_csv(out/f'fold{f}_seed{s}.csv') for f in FOLDS for s in SEEDS],ignore_index=True)
    rows['delta_sq']=(rows.sub_ec-rows.h10)**2-(rows.sub_ec-rows.v2)**2
    summary={}
    for seed,g in rows.groupby('seed'):
        a,b=base.rmse(g.sub_ec,g.v2),base.rmse(g.sub_ec,g.h10)
        ci,p=base.bootstrap(g,2912)
        farms={farm:base.rmse(z.sub_ec,z.h10)/base.rmse(z.sub_ec,z.v2)-1
               for farm,z in g.groupby('farm')}
        summary[str(seed)]=dict(v2_rmse=a,h10_rmse=b,relative_change=b/a-1,
                                by_farm=farms,bootstrap_mse_delta_95ci=ci,
                                bootstrap_p_worse=p)
    passes=all(z['relative_change']<0 for z in scores)
    passes=passes and all(all(x<0 for x in z['by_farm'].values())
                           and z['bootstrap_mse_delta_95ci'][1]<0 for z in summary.values())
    result=dict(scores=scores,per_seed=summary,screen_pass=bool(passes),
                final_lock_scored=False,test_X_read=False,
                hidden_labels_read=False,submission_created=False)
    (out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='scores'},ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=4):
        main()
