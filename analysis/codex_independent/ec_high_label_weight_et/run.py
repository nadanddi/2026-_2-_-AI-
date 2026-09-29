"""H9: fixed high-label-day EC ExtraTrees weighting against saved v2 OOF."""
import importlib.util
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits


HERE = Path(__file__).resolve().parent
BASE_CODE = HERE.parent / 'ec_dual_view_et/run.py'
spec = importlib.util.spec_from_file_location('ec_h9_base', BASE_CODE)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
ROOT, LOCAL = base.ROOT, base.LOCAL
FOLDS, SEEDS = base.FOLDS, base.SEEDS


def fit_pair(tr, va, seed):
    xtr, xva = tr[base.FULL], va[base.FULL]
    labels = tr.sub_ec.to_numpy(float)
    high = tr.groupby(['farm', 'day']).sub_ec.transform('mean').ge(1.2).to_numpy(bool)
    assert high.sum() % 24 == 0
    raw = base.model(seed)
    raw.fit(xtr, labels)
    raw_p = raw.predict(xva)
    weighted = base.model(seed)
    weighted.fit(xtr, labels, extratreesregressor__sample_weight=np.where(high, 2., 1.))
    weighted_p = weighted.predict(xva)
    assert np.isfinite(raw_p).all() and np.isfinite(weighted_p).all()
    return raw_p, weighted_p, int(high.sum() // 24)


def main():
    _, lab, dev, lock = base.prepare()
    paths = [(base.SEARCH if f < 8 else base.CONFIRM) / f'fold{f}.csv' for f in FOLDS]
    hashes = {str(p): base.sha(p) for p in [HERE / 'PROTOCOL.md', HERE / 'run.py',
                                            BASE_CODE, base.DATA / 'train_X.csv',
                                            base.DATA / 'train_y.csv', base.SPLITS,
                                            base.LOCK, *paths]}
    out = LOCAL / 'ec_high_label_weight_et' / datetime.now().strftime('%Y%m%d_%H%M%S')
    out.mkdir(parents=True, exist_ok=False)
    (out / 'manifest.json').write_text(json.dumps(dict(hashes=hashes, folds=FOLDS,
        seeds=SEEDS, high_threshold=1.2, high_weight=2., final_scale=.24),
        ensure_ascii=False, indent=2), encoding='utf-8')
    results = []
    for fold in FOLDS:
        va, v2 = base.validation(lab, fold)
        val = {(f, int(d)) for f, d in va[['farm', 'day']].itertuples(index=False, name=None)}
        assert not (val & lock)
        tr = dev[base.split_mask(dev, val | lock)].copy() if fold < 8 else dev[base.split_mask(dev, lock)].copy()
        assert not set(tr.row_id) & set(va.row_id)
        lo, hi = float(tr.sub_ec.min()), float(tr.sub_ec.max())
        assert np.all((v2 > lo + 1e-10) & (v2 < hi - 1e-10)), 'v2 clipping invalidates additive delta'
        for seed in SEEDS:
            raw_p, weighted_p, n_high = fit_pair(tr, va, seed)
            candidate = np.clip(v2 + .24 * base.shrink(weighted_p - raw_p, va), lo, hi)
            row = va[['row_id', 'farm', 'day', 'hour', 'sub_ec']].copy()
            row['fold'] = fold
            row['seed'] = seed
            row['v2'] = v2
            row['h9'] = candidate
            row.to_csv(out / f'fold{fold}_seed{seed}.csv', index=False, float_format='%.17g')
            a, b = base.rmse(row.sub_ec, row.v2), base.rmse(row.sub_ec, row.h9)
            rec = dict(fold=fold, seed=seed, train_days=len(tr) // 24,
                       high_train_days=n_high, v2_rmse=a, h9_rmse=b,
                       relative_change=b/a-1)
            results.append(rec)
            (out/'progress.json').write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
            print(json.dumps(rec), flush=True)
    rows = pd.concat([pd.read_csv(out/f'fold{f}_seed{s}.csv') for f in FOLDS for s in SEEDS], ignore_index=True)
    rows['delta_sq'] = (rows.sub_ec-rows.h9)**2 - (rows.sub_ec-rows.v2)**2
    summary = {}
    for seed, g in rows.groupby('seed'):
        a, b = base.rmse(g.sub_ec, g.v2), base.rmse(g.sub_ec, g.h9)
        ci, p = base.bootstrap(g, 2911)
        farms = {farm: base.rmse(z.sub_ec, z.h9) / base.rmse(z.sub_ec, z.v2) - 1
                 for farm, z in g.groupby('farm')}
        summary[str(seed)] = dict(v2_rmse=a, h9_rmse=b, relative_change=b/a-1,
                                  by_farm=farms, bootstrap_mse_delta_95ci=ci,
                                  bootstrap_p_worse=p)
    passes = all(r['relative_change'] < 0 for r in results)
    passes = passes and all(all(x < 0 for x in z['by_farm'].values())
                            and z['bootstrap_mse_delta_95ci'][1] < 0 for z in summary.values())
    result = dict(scores=results, per_seed=summary, screen_pass=bool(passes),
                  final_lock_scored=False, test_X_read=False,
                  hidden_labels_read=False, submission_created=False)
    (out/'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k != 'scores'}, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=4):
        main()
