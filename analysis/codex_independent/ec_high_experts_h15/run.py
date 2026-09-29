"""H15 training-time high-EC classifier and conditional level experts."""
import env  # first project import

import hashlib
import importlib.util
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingClassifier
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
spec = importlib.util.spec_from_file_location('h12_for_h15', H12CODE)
h12 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h12)
FULL = h12.FULL
FOLDS = (0, 2, 4, 6, 8, 9)
SEEDS = (7, 101)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def et(seed):
    return make_pipeline(SimpleImputer(strategy='median'), ExtraTreesRegressor(
        n_estimators=400, max_features=1.0, min_samples_leaf=2,
        n_jobs=4, random_state=seed))


def fit_predict(tr, va, seed):
    assert tr.high.any() and (~tr.high).any()
    classifier = HistGradientBoostingClassifier(max_iter=150, learning_rate=.05,
        max_leaf_nodes=7, min_samples_leaf=100, l2_regularization=10,
        early_stopping=False, random_state=2915)
    classifier.fit(tr[FULL], tr.high.to_numpy(int), sample_weight=np.full(len(tr), 1/24))
    q_raw = classifier.predict_proba(va[FULL])[:, 1]
    v = va[['farm', 'day', 'hour']].reset_index(drop=True).copy()
    v['q'] = q_raw
    v = v.sort_values(['farm', 'day', 'hour'])
    q = np.empty(len(va))
    q[v.index.to_numpy()] = v.groupby(['farm', 'day']).q.transform(
        lambda s: s.expanding().mean()).to_numpy()
    global_m, low_m, high_m = et(seed), et(seed), et(seed)
    global_m.fit(tr[FULL], tr.sub_ec.to_numpy(float))
    low = tr[~tr.high]
    high = tr[tr.high]
    low_m.fit(low[FULL], low.sub_ec.to_numpy(float))
    high_m.fit(high[FULL], high.sub_ec.to_numpy(float))
    for m in (global_m, low_m, high_m):
        m.steps[-1][1].n_jobs = 1
    g = global_m.predict(va[FULL]); lo = low_m.predict(va[FULL]); hi = high_m.predict(va[FULL])
    p = .5*g + .5*(q*hi + (1-q)*lo)
    return p, q, g, lo, hi


def main():
    raw = pd.read_csv(DATA / 'train_X.csv')
    raw = raw[raw.row_id.str[:3].isin(('F13', 'F47'))].reset_index(drop=True)
    assert len(raw) == 9600
    lab = h12.features(raw)[['row_id', 'farm', 'hour', *FULL]]
    lab = lab.merge(pd.read_csv(DATA / 'train_y.csv', usecols=['row_id', 'sub_ec']),
                    on='row_id', validate='one_to_one')
    lab = lab.merge(pd.read_csv(SPLITS)[['row_id', 'fold']], on='row_id', validate='one_to_one')
    means = lab.groupby(['farm', 'day']).sub_ec.mean().ge(1.2).rename('high').reset_index()
    lab = lab.merge(means, on=['farm', 'day'], validate='many_to_one')
    assert len(lab) == 9600 and lab.sub_ec.notna().all()
    lock = {(r['farm'], r['day']) for r in json.loads(LOCK.read_text(encoding='utf-8'))['selected']}
    hold = {(f, int(d)) for f, d in lab.loc[lab.fold.isin((8, 9)), ['farm', 'day']]
            .itertuples(index=False, name=None)}
    dev = lab[h12.split_mask(lab, hold)].copy()
    paths = [(SEARCH if f < 8 else CONFIRM) / f'fold{f}.csv' for f in FOLDS]
    hashes = {str(p): sha(p) for p in [HERE / 'PROTOCOL.md', HERE / 'run.py',
                                       H12CODE, DATA / 'train_X.csv', DATA / 'train_y.csv',
                                       SPLITS, LOCK, *paths]}
    out = LOCAL / 'ec_high_experts_h15' / datetime.now().strftime('%Y%m%d_%H%M%S')
    out.mkdir(parents=True, exist_ok=False)
    (out / 'manifest.json').write_text(json.dumps({'hashes': hashes, 'folds': FOLDS,
        'seeds': SEEDS, 'high_threshold': 1.2, 'expert_share': .5,
        'candidate_share': .2}, ensure_ascii=False, indent=2), encoding='utf-8')
    scores = []
    for fold in FOLDS:
        saved = pd.read_csv((SEARCH if fold < 8 else CONFIRM) / f'fold{fold}.csv')
        v2 = saved['blend' if fold < 8 else 'candidate'].to_numpy(float)
        va = lab.set_index('row_id').loc[saved.row_id].reset_index()
        np.testing.assert_allclose(va.sub_ec, saved.sub_ec, rtol=0, atol=1e-12)
        val_days = {(f, int(d)) for f, d in va[['farm', 'day']].itertuples(index=False, name=None)}
        assert not val_days & lock
        tr = dev[h12.split_mask(dev, val_days | lock)].copy() if fold < 8 else dev[h12.split_mask(dev, lock)].copy()
        n_high = tr[tr.high][['farm', 'day']].drop_duplicates().shape[0]
        n_low = tr[~tr.high][['farm', 'day']].drop_duplicates().shape[0]
        lo, hi = float(tr.sub_ec.min()), float(tr.sub_ec.max())
        for seed in SEEDS:
            p, q, global_p, low_p, high_p = fit_predict(tr, va, seed)
            candidate = np.clip(.8*v2 + .2*p, lo, hi)
            rec = va[['row_id', 'farm', 'day', 'hour', 'sub_ec', 'high']].copy()
            rec['fold'] = fold; rec['seed'] = seed
            rec['v2'] = v2; rec['q'] = q; rec['global'] = global_p
            rec['low_expert'] = low_p; rec['high_expert'] = high_p; rec['h15'] = candidate
            rec.to_csv(out / f'fold{fold}_seed{seed}.csv', index=False, float_format='%.17g')
            a, b = h12.rmse(rec.sub_ec, rec.v2), h12.rmse(rec.sub_ec, rec.h15)
            item = {'fold': fold, 'seed': seed, 'train_high_days': n_high,
                'train_other_days': n_low, 'v2': a, 'h15': b, 'relative_change': b/a-1,
                'model_rmse': h12.rmse(rec.sub_ec, p),
                'q_high_mean': float(q[va.high.to_numpy()].mean()) if va.high.any() else None,
                'q_other_mean': float(q[~va.high.to_numpy()].mean())}
            scores.append(item)
            (out / 'progress.json').write_text(json.dumps(scores, ensure_ascii=False, indent=2), encoding='utf-8')
            print(json.dumps(item), flush=True)
    rows = pd.concat([pd.read_csv(out / f'fold{f}_seed{s}.csv') for f in FOLDS for s in SEEDS], ignore_index=True)
    rows['delta_sq'] = (rows.sub_ec-rows.h15)**2-(rows.sub_ec-rows.v2)**2
    summary = {}
    for seed, g in rows.groupby('seed'):
        ci, p = h12.bootstrap(g, 2915+int(seed))
        subsets = {str(bool(k)): {'days': len(q)//24, 'v2': h12.rmse(q.sub_ec, q.v2),
                    'h15': h12.rmse(q.sub_ec, q.h15), 'q_mean': float(q.q.mean()),
                    'high_expert_rmse': h12.rmse(q.sub_ec, q.high_expert),
                    'low_expert_rmse': h12.rmse(q.sub_ec, q.low_expert)}
                    for k, q in g.groupby('high')}
        summary[str(seed)] = {'v2': h12.rmse(g.sub_ec, g.v2), 'h15': h12.rmse(g.sub_ec, g.h15),
            'relative_change': h12.rmse(g.sub_ec, g.h15)/h12.rmse(g.sub_ec, g.v2)-1,
            'bootstrap_ci': ci, 'p_worse': p, 'by_high': subsets,
            'by_farm': {f: h12.rmse(q.sub_ec, q.h15)/h12.rmse(q.sub_ec, q.v2)-1
                        for f, q in g.groupby('farm')}}
    screen = all(s['relative_change'] < 0 for s in scores)
    screen &= all(v['p_worse'] < .025/12 and all(x < 0 for x in v['by_farm'].values())
                  for v in summary.values())
    final = {'scores': scores, 'per_seed': summary, 'screen_pass': bool(screen),
             'final_lock_scored': False, 'test_X_read': False, 'submission_created': False}
    (out / 'result.json').write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'out': str(out), 'screen_pass': screen, 'per_seed': summary},
                     ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=4):
        main()
