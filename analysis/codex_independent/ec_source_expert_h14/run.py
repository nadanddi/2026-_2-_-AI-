"""H14 source-specific partial-sharing Ridge trained on input-only source clusters."""
import env  # first project import

import hashlib
import importlib.util
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
DATA = ROOT / '온라인대회자료/정형데이터/참가자_배포'
LOCAL = ROOT / 'analysis/local'
SEARCH = LOCAL / 'ec_three_seed_ensemble_cv/20260928_035914'
CONFIRM = LOCAL / 'ec_locked_confirmation/20260928_044934'
SPLITS = LOCAL / 'rl_ec_v1/20260927_173801/splits.csv'
LOCK = HERE.parent / 'ec_final_lock/locked_days.json'
H13CODE = HERE.parent / 'ec_source_identity_h13/run.py'
spec = importlib.util.spec_from_file_location('h13_source_profile', H13CODE)
h13 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h13)
h12 = h13.h12
FULL = h12.FULL
FOLDS = (0, 2, 4, 6, 8, 9)
SEEDS = (7, 101)
INTERACT = ['in_temp', 'in_temp_h0', 'act_heating_h0', 'act_vent']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def groups(frame):
    return np.where(frame.farm.eq('F13'), 0, 4) + frame.cluster.to_numpy(int)


def interaction(x, frame):
    # x is the global FULL matrix after training-only imputation and scaling.
    z = np.column_stack([np.ones(len(x)), *[x[:, FULL.index(v)] for v in INTERACT]])
    gid = groups(frame)
    return np.hstack([z * gid[:, None].__eq__(j) for j in range(8)])


def predict_pair(tr, va, seed):
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(tr))
    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    xtr = scaler.fit_transform(imputer.fit_transform(tr.iloc[order][FULL]))
    xva = scaler.transform(imputer.transform(va[FULL]))
    y = tr.sub_ec.to_numpy(float)[order]
    tro = tr.iloc[order].reset_index(drop=True)
    global_model = Ridge(alpha=100, solver='lsqr')
    expert_model = Ridge(alpha=100, solver='lsqr')
    global_model.fit(xtr, y)
    exptr = np.hstack([xtr, interaction(xtr, tro)])
    expva = np.hstack([xva, interaction(xva, va)])
    expert_model.fit(exptr, y)
    return global_model.predict(xva), expert_model.predict(expva)


def main():
    raw = pd.read_csv(DATA / 'train_X.csv')
    raw = raw[raw.row_id.str[:3].isin(('F13', 'F47'))].reset_index(drop=True)
    assert len(raw) == 9600
    full = h12.features(raw)[['row_id', 'farm', 'hour', *FULL]]
    profile = h13.source_profiles(raw)
    lab = full.merge(pd.read_csv(DATA / 'train_y.csv', usecols=['row_id', 'sub_ec']),
                     on='row_id', validate='one_to_one')
    lab = lab.merge(pd.read_csv(SPLITS)[['row_id', 'fold']], on='row_id', validate='one_to_one')
    assert len(lab) == 9600 and lab.sub_ec.notna().all()
    lock = {(r['farm'], r['day']) for r in json.loads(LOCK.read_text(encoding='utf-8'))['selected']}
    hold = {(f, int(d)) for f, d in lab.loc[lab.fold.isin((8, 9)), ['farm', 'day']]
            .itertuples(index=False, name=None)}
    dev = lab[h12.split_mask(lab, hold)].copy()
    paths = [(SEARCH if f < 8 else CONFIRM) / f'fold{f}.csv' for f in FOLDS]
    hashes = {str(p): sha(p) for p in [HERE / 'PROTOCOL.md', HERE / 'run.py', H13CODE,
        h13.H12CODE, DATA / 'train_X.csv', DATA / 'train_y.csv', SPLITS, LOCK, *paths]}
    out = LOCAL / 'ec_source_expert_h14' / datetime.now().strftime('%Y%m%d_%H%M%S')
    out.mkdir(parents=True, exist_ok=False)
    (out / 'manifest.json').write_text(json.dumps({'hashes': hashes, 'folds': FOLDS,
        'seeds': SEEDS, 'ridge_alpha': 100, 'interactions': INTERACT, 'delta_share': .24},
        ensure_ascii=False, indent=2), encoding='utf-8')
    scores, clusters = [], []
    for fold in FOLDS:
        saved = pd.read_csv((SEARCH if fold < 8 else CONFIRM) / f'fold{fold}.csv')
        v2 = saved['blend' if fold < 8 else 'candidate'].to_numpy(float)
        va = lab.set_index('row_id').loc[saved.row_id].reset_index()
        np.testing.assert_allclose(va.sub_ec, saved.sub_ec, rtol=0, atol=1e-12)
        val_days = {(f, int(d)) for f, d in va[['farm', 'day']].itertuples(index=False, name=None)}
        assert not val_days & lock
        tr = dev[h12.split_mask(dev, val_days | lock)].copy() if fold < 8 else dev[h12.split_mask(dev, lock)].copy()
        train_days = {(f, int(d)) for f, d in tr[['farm', 'day']].itertuples(index=False, name=None)}
        embedding = h13.fold_distances(profile, train_days, val_days)
        embedding['cluster'] = embedding[h13.DIST].to_numpy().argmin(axis=1)
        embedding = embedding[['farm', 'day', 'cluster']]
        tr = tr.merge(embedding, on=['farm', 'day'], validate='many_to_one')
        va = va.merge(embedding, on=['farm', 'day'], validate='many_to_one')
        assert len(va) == len(saved) and va.row_id.tolist() == saved.row_id.tolist()
        summary = tr.groupby(['farm', 'cluster']).agg(rows=('sub_ec', 'size'), ec_mean=('sub_ec', 'mean')).reset_index()
        for r in summary.itertuples(index=False):
            clusters.append({'fold': fold, 'farm': r.farm, 'cluster': int(r.cluster),
                             'train_days': int(r.rows)//24, 'train_ec_mean': float(r.ec_mean)})
        lo, hi = float(tr.sub_ec.min()), float(tr.sub_ec.max())
        for seed in SEEDS:
            bp, ep = predict_pair(tr, va, seed)
            candidate = np.clip(v2 + .24*h12.shrink(ep-bp, va), lo, hi)
            rec = va[['row_id', 'farm', 'day', 'hour', 'sub_ec']].copy()
            rec['fold'] = fold; rec['seed'] = seed
            rec['v2'] = v2; rec['base_ridge'] = bp; rec['expert_ridge'] = ep; rec['h14'] = candidate
            rec.to_csv(out / f'fold{fold}_seed{seed}.csv', index=False, float_format='%.17g')
            a, b = h12.rmse(rec.sub_ec, rec.v2), h12.rmse(rec.sub_ec, rec.h14)
            item = {'fold': fold, 'seed': seed, 'train_days': len(tr)//24,
                'v2': a, 'h14': b, 'relative_change': b/a-1,
                'base_ridge': h12.rmse(rec.sub_ec, bp), 'expert_ridge': h12.rmse(rec.sub_ec, ep)}
            scores.append(item)
            (out / 'progress.json').write_text(json.dumps(scores, ensure_ascii=False, indent=2), encoding='utf-8')
            print(json.dumps(item), flush=True)
    (out / 'cluster_diagnostic.json').write_text(json.dumps(clusters, ensure_ascii=False, indent=2), encoding='utf-8')
    rows = pd.concat([pd.read_csv(out / f'fold{f}_seed{s}.csv') for f in FOLDS for s in SEEDS], ignore_index=True)
    rows['delta_sq'] = (rows.sub_ec-rows.h14)**2-(rows.sub_ec-rows.v2)**2
    high = set(tuple(x) for x in lab.groupby(['farm', 'day']).sub_ec.mean().loc[lambda s: s >= 1.2].index)
    summary = {}
    for seed, g in rows.groupby('seed'):
        ci, p = h12.bootstrap(g, 2914+int(seed))
        g = g.copy()
        g['high'] = [(f, int(d)) in high for f, d in g[['farm', 'day']].itertuples(index=False, name=None)]
        subsets = {str(bool(k)): {'days': len(q)//24, 'v2': h12.rmse(q.sub_ec, q.v2),
                    'h14': h12.rmse(q.sub_ec, q.h14)} for k, q in g.groupby('high')}
        summary[str(seed)] = {'v2': h12.rmse(g.sub_ec, g.v2), 'h14': h12.rmse(g.sub_ec, g.h14),
            'base_ridge': h12.rmse(g.sub_ec, g.base_ridge),
            'expert_ridge': h12.rmse(g.sub_ec, g.expert_ridge),
            'relative_change': h12.rmse(g.sub_ec, g.h14)/h12.rmse(g.sub_ec, g.v2)-1,
            'bootstrap_ci': ci, 'p_worse': p, 'by_high': subsets,
            'by_farm': {f: h12.rmse(q.sub_ec, q.h14)/h12.rmse(q.sub_ec, q.v2)-1
                        for f, q in g.groupby('farm')}}
    determinism = {}
    for fold in FOLDS:
        a = pd.read_csv(out / f'fold{fold}_seed7.csv')
        b = pd.read_csv(out / f'fold{fold}_seed101.csv')
        determinism[str(fold)] = float(np.max(np.abs(a.h14-b.h14)))
    screen = all(s['relative_change'] < 0 for s in scores)
    screen &= all(v['p_worse'] < .025/12 and all(x < 0 for x in v['by_farm'].values())
                  for v in summary.values())
    screen &= all(v < 1e-6 for v in determinism.values())
    final = {'scores': scores, 'per_seed': summary, 'seed_max_abs_difference': determinism,
             'screen_pass': bool(screen), 'final_lock_scored': False,
             'test_X_read': False, 'submission_created': False}
    (out / 'result.json').write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'out': str(out), 'screen_pass': screen, 'per_seed': summary,
                      'seed_max_abs_difference': determinism}, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=4):
        main()
