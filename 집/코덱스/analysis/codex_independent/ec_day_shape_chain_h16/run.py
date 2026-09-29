"""H16 EC daily level and within-day shape with causal source-chain inputs."""
import env  # first project import

import hashlib
import importlib.util
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
DATA = ROOT / '온라인대회자료/정형데이터/참가자_배포'
LOCAL = ROOT / 'analysis/local'
SEARCH = LOCAL / 'ec_three_seed_ensemble_cv/20260928_035914'
CONFIRM = LOCAL / 'ec_locked_confirmation/20260928_044934'
SPLITS = LOCAL / 'rl_ec_v1/20260927_173801/splits.csv'
LOCK = HERE.parent / 'ec_final_lock/locked_days.json'
H12CODE = HERE.parent / 'ec_chain_causal_h12/run_v2.py'
spec = importlib.util.spec_from_file_location('h12_for_h16', H12CODE)
h12 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h12)
FULL = h12.FULL
CHAIN = ['rad_total_chain3', 'vpd_total_chain3', 'sealed_run',
         'vent_mean_chain1', 'fan_mean_chain1', 'link_gap']
FOLDS = (0, 2, 4, 6, 8, 9)
SEEDS = (7, 101)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def level(seed):
    return HistGradientBoostingRegressor(max_iter=250, learning_rate=.05,
        max_leaf_nodes=7, min_samples_leaf=100, l2_regularization=10,
        early_stopping=False, random_state=seed)


def shape(seed):
    return make_pipeline(SimpleImputer(strategy='median'), ExtraTreesRegressor(
        n_estimators=400, max_features=1.0, min_samples_leaf=2,
        n_jobs=4, random_state=seed))


def fit_predict(tr, va, seed):
    base_level, chain_level, shape_model = level(seed), level(seed), shape(seed)
    base_level.fit(tr[FULL], tr.ec_day_mean.to_numpy(float))
    chain_level.fit(tr[FULL + CHAIN], tr.ec_day_mean.to_numpy(float))
    shape_model.fit(tr[FULL], tr.sub_ec.to_numpy(float)-tr.ec_day_mean.to_numpy(float))
    shape_model.steps[-1][1].n_jobs = 1
    s = shape_model.predict(va[FULL])
    bp = base_level.predict(va[FULL]) + s
    cp = chain_level.predict(va[FULL + CHAIN]) + s
    return bp, cp, s


def decomposition(g, col):
    d = g.groupby(['farm', 'day'])[['sub_ec', col]].mean()
    daily = float(np.mean((d.sub_ec-d[col])**2))
    total = float(np.mean((g.sub_ec-g[col])**2))
    return {'daily_level_mse': daily, 'total_mse': total,
            'daily_fraction': daily/total if total else np.nan}


def main():
    raw = pd.read_csv(DATA / 'train_X.csv')
    raw = raw[raw.row_id.str[:3].isin(('F13', 'F47'))].reset_index(drop=True)
    assert len(raw) == 9600
    lab = h12.features(raw)[['row_id', 'farm', 'hour', *FULL, *CHAIN]]
    lab = lab.merge(pd.read_csv(DATA / 'train_y.csv', usecols=['row_id', 'sub_ec']),
                    on='row_id', validate='one_to_one')
    lab = lab.merge(pd.read_csv(SPLITS)[['row_id', 'fold']], on='row_id', validate='one_to_one')
    means = lab.groupby(['farm', 'day']).sub_ec.mean().rename('ec_day_mean').reset_index()
    lab = lab.merge(means, on=['farm', 'day'], validate='many_to_one')
    lab['high'] = lab.ec_day_mean.ge(1.2)
    assert len(lab) == 9600 and lab.sub_ec.notna().all()
    lock = {(r['farm'], r['day']) for r in json.loads(LOCK.read_text(encoding='utf-8'))['selected']}
    hold = {(f, int(d)) for f, d in lab.loc[lab.fold.isin((8, 9)), ['farm', 'day']]
            .itertuples(index=False, name=None)}
    dev = lab[h12.split_mask(lab, hold)].copy()
    paths = [(SEARCH if f < 8 else CONFIRM) / f'fold{f}.csv' for f in FOLDS]
    hashes = {str(p): sha(p) for p in [HERE / 'PROTOCOL.md', HERE / 'run.py',
                                       H12CODE, DATA / 'train_X.csv', DATA / 'train_y.csv',
                                       SPLITS, LOCK, *paths]}
    out = LOCAL / 'ec_day_shape_chain_h16' / datetime.now().strftime('%Y%m%d_%H%M%S')
    out.mkdir(parents=True, exist_ok=False)
    (out / 'manifest.json').write_text(json.dumps({'hashes': hashes, 'folds': FOLDS,
        'seeds': SEEDS, 'chain_features': CHAIN, 'candidate_share': .2},
        ensure_ascii=False, indent=2), encoding='utf-8')
    scores = []
    for fold in FOLDS:
        saved = pd.read_csv((SEARCH if fold < 8 else CONFIRM) / f'fold{fold}.csv')
        v2 = saved['blend' if fold < 8 else 'candidate'].to_numpy(float)
        va = lab.set_index('row_id').loc[saved.row_id].reset_index()
        np.testing.assert_allclose(va.sub_ec, saved.sub_ec, rtol=0, atol=1e-12)
        val_days = {(f, int(d)) for f, d in va[['farm', 'day']].itertuples(index=False, name=None)}
        assert not val_days & lock
        tr = dev[h12.split_mask(dev, val_days | lock)].copy() if fold < 8 else dev[h12.split_mask(dev, lock)].copy()
        lo, hi = float(tr.sub_ec.min()), float(tr.sub_ec.max())
        for seed in SEEDS:
            bp, cp, s = fit_predict(tr, va, seed)
            candidate = np.clip(.8*v2 + .2*cp, lo, hi)
            rec = va[['row_id', 'farm', 'day', 'hour', 'sub_ec', 'high']].copy()
            rec['fold'] = fold; rec['seed'] = seed
            rec['v2'] = v2; rec['base_two_stage'] = bp
            rec['chain_two_stage'] = cp; rec['shape'] = s; rec['h16'] = candidate
            rec.to_csv(out / f'fold{fold}_seed{seed}.csv', index=False, float_format='%.17g')
            a, b = h12.rmse(rec.sub_ec, rec.v2), h12.rmse(rec.sub_ec, rec.h16)
            item = {'fold': fold, 'seed': seed, 'train_days': len(tr)//24,
                'v2': a, 'h16': b, 'relative_change': b/a-1,
                'base_two_stage': h12.rmse(rec.sub_ec, bp),
                'chain_two_stage': h12.rmse(rec.sub_ec, cp)}
            scores.append(item)
            (out / 'progress.json').write_text(json.dumps(scores, ensure_ascii=False, indent=2), encoding='utf-8')
            print(json.dumps(item), flush=True)
    rows = pd.concat([pd.read_csv(out / f'fold{f}_seed{s}.csv') for f in FOLDS for s in SEEDS], ignore_index=True)
    rows['delta_sq'] = (rows.sub_ec-rows.h16)**2-(rows.sub_ec-rows.v2)**2
    summary = {}
    for seed, g in rows.groupby('seed'):
        ci, p = h12.bootstrap(g, 2916+int(seed))
        subsets = {str(bool(k)): {'days': len(q)//24, 'v2': h12.rmse(q.sub_ec, q.v2),
                    'h16': h12.rmse(q.sub_ec, q.h16),
                    'base_two_stage': h12.rmse(q.sub_ec, q.base_two_stage),
                    'chain_two_stage': h12.rmse(q.sub_ec, q.chain_two_stage)}
                    for k, q in g.groupby('high')}
        summary[str(seed)] = {'v2': h12.rmse(g.sub_ec, g.v2), 'h16': h12.rmse(g.sub_ec, g.h16),
            'base_two_stage': h12.rmse(g.sub_ec, g.base_two_stage),
            'chain_two_stage': h12.rmse(g.sub_ec, g.chain_two_stage),
            'relative_change': h12.rmse(g.sub_ec, g.h16)/h12.rmse(g.sub_ec, g.v2)-1,
            'bootstrap_ci': ci, 'p_worse': p, 'by_high': subsets,
            'v2_decomposition': decomposition(g, 'v2'),
            'h16_decomposition': decomposition(g, 'h16'),
            'by_farm': {f: h12.rmse(q.sub_ec, q.h16)/h12.rmse(q.sub_ec, q.v2)-1
                        for f, q in g.groupby('farm')}}
    screen = all(s['relative_change'] < 0 for s in scores)
    screen &= all(v['p_worse'] < .025/12 and all(x < 0 for x in v['by_farm'].values())
                  for v in summary.values())
    final = {'scores': scores, 'per_seed': summary, 'screen_pass': bool(screen),
             'chain_missing_fraction': {c: float(lab[c].isna().mean()) for c in CHAIN},
             'final_lock_scored': False, 'test_X_read': False, 'submission_created': False}
    (out / 'result.json').write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'out': str(out), 'screen_pass': screen, 'per_seed': summary},
                     ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=4):
        main()
