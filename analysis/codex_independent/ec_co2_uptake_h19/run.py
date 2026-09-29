"""H19 causal CO2 drawdown uptake proxy in closed, undosed daylight hours."""
import env  # first project import

import hashlib
import importlib.util
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
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
spec = importlib.util.spec_from_file_location('h12_for_h17', H12CODE)
h12 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h12)
FULL = h12.FULL
UPTAKE = ['uptake_valid_count', 'uptake_valid_mean', 'uptake_recent3_mean']
FOLDS = (0, 2, 4, 6, 8, 9)
SEEDS = (7, 101)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def uptake_features(raw):
    x = h12.identify(raw).sort_values(['farm', 'day', 'hour']).reset_index(drop=True)
    assert len(x) == 9600
    g = x.groupby(['farm', 'day'], sort=False)
    co2_prev = g.in_co2.shift(1)
    vent_prev = g.act_vent.shift(1)
    valid = (x.hour.ge(1) & co2_prev.notna() & x.in_co2.notna() &
             vent_prev.eq(0) & x.act_vent.eq(0) & x.act_co2.eq(0) &
             x.out_rad.notna() & x.out_rad.gt(50))
    x['valid'] = valid.astype(int)
    x['drawdown'] = np.where(valid, (co2_prev-x.in_co2).clip(0, 200), 0.0)
    x['uptake_valid_count'] = x.groupby(['farm', 'day']).valid.cumsum().astype(float)
    total = x.groupby(['farm', 'day']).drawdown.cumsum()
    x['uptake_valid_mean'] = total/x.uptake_valid_count.replace(0, np.nan)
    recent = np.full(len(x), np.nan)
    for _, part in x.groupby(['farm', 'day'], sort=False):
        values = []
        for i in part.index:
            if x.at[i, 'valid']:
                values.append(float(x.at[i, 'drawdown']))
            if values:
                recent[i] = float(np.mean(values[-3:]))
    x['uptake_recent3_mean'] = recent
    x.loc[x.uptake_valid_count.eq(0), 'uptake_valid_count'] = np.nan
    return x[['row_id', 'farm', 'day', 'hour', *UPTAKE]]


def model(seed):
    return make_pipeline(SimpleImputer(strategy='median'), ExtraTreesRegressor(
        n_estimators=600, max_features=1.0, min_samples_leaf=1,
        n_jobs=4, random_state=seed))


def main():
    raw = pd.read_csv(DATA / 'train_X.csv')
    raw = raw[raw.row_id.str[:3].isin(('F13', 'F47'))].reset_index(drop=True)
    base = h12.features(raw)[['row_id', 'farm', 'hour', *FULL]]
    memory = uptake_features(raw)
    lab = base.merge(memory[['row_id', *UPTAKE]], on='row_id', validate='one_to_one')
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
        H12CODE, DATA / 'train_X.csv', DATA / 'train_y.csv', SPLITS, LOCK, *paths]}
    out = LOCAL / 'ec_co2_uptake_h19' / datetime.now().strftime('%Y%m%d_%H%M%S')
    out.mkdir(parents=True, exist_ok=False)
    (out / 'manifest.json').write_text(json.dumps({'hashes': hashes, 'folds': FOLDS,
        'seeds': SEEDS, 'uptake_features': UPTAKE, 'delta_share': .24},
        ensure_ascii=False, indent=2), encoding='utf-8')
    memory[['row_id', 'farm', 'day', 'hour', *UPTAKE]].to_csv(
        out / 'input_only_uptake.csv', index=False)
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
            bm, lm = model(seed), model(seed)
            bm.fit(tr[FULL], tr.sub_ec.to_numpy(float))
            lm.fit(tr[FULL + UPTAKE], tr.sub_ec.to_numpy(float))
            bm.steps[-1][1].n_jobs = 1; lm.steps[-1][1].n_jobs = 1
            bp = bm.predict(va[FULL]); lp = lm.predict(va[FULL + UPTAKE])
            candidate = np.clip(v2 + .24*h12.shrink(lp-bp, va), lo, hi)
            rec = va[['row_id', 'farm', 'day', 'hour', 'sub_ec', 'high', 'uptake_valid_count']].copy()
            rec['fold'] = fold; rec['seed'] = seed
            rec['v2'] = v2; rec['base_et'] = bp; rec['uptake_et'] = lp; rec['h19'] = candidate
            rec.to_csv(out / f'fold{fold}_seed{seed}.csv', index=False, float_format='%.17g')
            a, b = h12.rmse(rec.sub_ec, rec.v2), h12.rmse(rec.sub_ec, rec.h19)
            item = {'fold': fold, 'seed': seed, 'train_days': len(tr)//24,
                'v2': a, 'h19': b, 'relative_change': b/a-1,
                'base_et': h12.rmse(rec.sub_ec, bp), 'uptake_et': h12.rmse(rec.sub_ec, lp)}
            scores.append(item)
            (out / 'progress.json').write_text(json.dumps(scores, ensure_ascii=False, indent=2), encoding='utf-8')
            print(json.dumps(item), flush=True)
    rows = pd.concat([pd.read_csv(out / f'fold{f}_seed{s}.csv') for f in FOLDS for s in SEEDS], ignore_index=True)
    rows['delta_sq'] = (rows.sub_ec-rows.h19)**2-(rows.sub_ec-rows.v2)**2
    summary = {}
    for seed, g in rows.groupby('seed'):
        ci, p = h12.bootstrap(g, 2917+int(seed))
        subsets = {str(bool(k)): {'days': len(q)//24, 'v2': h12.rmse(q.sub_ec, q.v2),
                    'h19': h12.rmse(q.sub_ec, q.h19)} for k, q in g.groupby('high')}
        early = g[g.hour.le(6)]
        valid_days = g.groupby(['farm', 'day']).uptake_valid_count.max().notna()
        day_idx = pd.MultiIndex.from_frame(g[['farm', 'day']])
        having = g[valid_days.reindex(day_idx).to_numpy(bool)]
        summary[str(seed)] = {'v2': h12.rmse(g.sub_ec, g.v2), 'h19': h12.rmse(g.sub_ec, g.h19),
            'base_et': h12.rmse(g.sub_ec, g.base_et), 'uptake_et': h12.rmse(g.sub_ec, g.uptake_et),
            'relative_change': h12.rmse(g.sub_ec, g.h19)/h12.rmse(g.sub_ec, g.v2)-1,
            'bootstrap_ci': ci, 'p_worse': p, 'by_high': subsets,
            'early_hours': {'rows': len(early), 'v2': h12.rmse(early.sub_ec, early.v2),
                'h19': h12.rmse(early.sub_ec, early.h19)},
            'uptake_days': {'days': len(having)//24,
                'v2': h12.rmse(having.sub_ec, having.v2) if len(having) else None,
                'h19': h12.rmse(having.sub_ec, having.h19) if len(having) else None},
            'by_farm': {f: h12.rmse(q.sub_ec, q.h19)/h12.rmse(q.sub_ec, q.v2)-1
                        for f, q in g.groupby('farm')}}
    screen = all(s['relative_change'] < 0 for s in scores)
    screen &= all(v['p_worse'] < .025/12 and all(x < 0 for x in v['by_farm'].values())
                  for v in summary.values())
    final = {'scores': scores, 'per_seed': summary, 'screen_pass': bool(screen),
             'uptake_missing_fraction': {c: float(lab[c].isna().mean()) for c in UPTAKE},
             'final_lock_scored': False, 'test_X_read': False, 'submission_created': False}
    (out / 'result.json').write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'out': str(out), 'screen_pass': screen, 'per_seed': summary},
                     ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=4):
        main()
