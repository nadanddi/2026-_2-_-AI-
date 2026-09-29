"""H20 causal indoor-outdoor moisture balance proxy for transpiration."""
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
MOIST = ['ah_gap', 'vent_flux_cum', 'closed_rise_cum']
FOLDS = (0, 2, 4, 6, 8, 9)
SEEDS = (7, 101)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def absolute_humidity(temp, rh):
    e = 6.112*np.exp(17.67*temp/(temp+243.5))
    return 216.7*e*rh/100/(temp+273.15)


def moisture_features(raw):
    x = h12.identify(raw).sort_values(['farm', 'day', 'hour']).reset_index(drop=True)
    assert len(x) == 9600
    g = x.groupby(['farm', 'day'], sort=False)
    inside = absolute_humidity(x.in_temp, x.in_hum)
    outside = absolute_humidity(x.out_temp, x.out_hum)
    x['ah_gap'] = inside-outside
    x['flux_hour'] = x.ah_gap.clip(lower=0)*x.act_vent.clip(0, 100)/100
    x['vent_flux_cum'] = x.groupby(['farm', 'day']).flux_hour.cumsum()
    x['inside_ah'] = inside
    prev_ah = x.groupby(['farm', 'day']).inside_ah.shift(1)
    vent_prev = g.act_vent.shift(1)
    condition = (x.hour.ge(1) & prev_ah.notna() & inside.notna() &
                 vent_prev.eq(0) & x.act_vent.eq(0) & x.act_fog.eq(0))
    rise = (inside-prev_ah).clip(lower=0)
    x['rise_hour'] = np.where(condition, rise, 0.0)
    missing = x.hour.ge(1) & (prev_ah.isna() | inside.isna() |
        vent_prev.isna() | x.act_vent.isna() | x.act_fog.isna())
    x.loc[missing, 'rise_hour'] = np.nan
    x['closed_rise_cum'] = x.groupby(['farm', 'day']).rise_hour.cumsum()
    return x[['row_id', 'farm', 'day', 'hour', 'flux_hour', *MOIST]]


def model(seed):
    return make_pipeline(SimpleImputer(strategy='median'), ExtraTreesRegressor(
        n_estimators=600, max_features=1.0, min_samples_leaf=1,
        n_jobs=4, random_state=seed))


def main():
    raw = pd.read_csv(DATA / 'train_X.csv')
    raw = raw[raw.row_id.str[:3].isin(('F13', 'F47'))].reset_index(drop=True)
    base = h12.features(raw)[['row_id', 'farm', 'hour', *FULL]]
    memory = moisture_features(raw)
    lab = base.merge(memory[['row_id', 'flux_hour', *MOIST]], on='row_id', validate='one_to_one')
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
    out = LOCAL / 'ec_moisture_flux_h20' / datetime.now().strftime('%Y%m%d_%H%M%S')
    out.mkdir(parents=True, exist_ok=False)
    (out / 'manifest.json').write_text(json.dumps({'hashes': hashes, 'folds': FOLDS,
        'seeds': SEEDS, 'moisture_features': MOIST, 'delta_share': .24},
        ensure_ascii=False, indent=2), encoding='utf-8')
    memory[['row_id', 'farm', 'day', 'hour', 'flux_hour', *MOIST]].to_csv(
        out / 'input_only_moisture.csv', index=False)
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
            lm.fit(tr[FULL + MOIST], tr.sub_ec.to_numpy(float))
            bm.steps[-1][1].n_jobs = 1; lm.steps[-1][1].n_jobs = 1
            bp = bm.predict(va[FULL]); lp = lm.predict(va[FULL + MOIST])
            candidate = np.clip(v2 + .24*h12.shrink(lp-bp, va), lo, hi)
            rec = va[['row_id', 'farm', 'day', 'hour', 'sub_ec', 'high', 'flux_hour']].copy()
            rec['fold'] = fold; rec['seed'] = seed
            rec['v2'] = v2; rec['base_et'] = bp; rec['moisture_et'] = lp; rec['h20'] = candidate
            rec.to_csv(out / f'fold{fold}_seed{seed}.csv', index=False, float_format='%.17g')
            a, b = h12.rmse(rec.sub_ec, rec.v2), h12.rmse(rec.sub_ec, rec.h20)
            item = {'fold': fold, 'seed': seed, 'train_days': len(tr)//24,
                'v2': a, 'h20': b, 'relative_change': b/a-1,
                'base_et': h12.rmse(rec.sub_ec, bp), 'moisture_et': h12.rmse(rec.sub_ec, lp)}
            scores.append(item)
            (out / 'progress.json').write_text(json.dumps(scores, ensure_ascii=False, indent=2), encoding='utf-8')
            print(json.dumps(item), flush=True)
    rows = pd.concat([pd.read_csv(out / f'fold{f}_seed{s}.csv') for f in FOLDS for s in SEEDS], ignore_index=True)
    rows['delta_sq'] = (rows.sub_ec-rows.h20)**2-(rows.sub_ec-rows.v2)**2
    summary = {}
    for seed, g in rows.groupby('seed'):
        ci, p = h12.bootstrap(g, 2917+int(seed))
        subsets = {str(bool(k)): {'days': len(q)//24, 'v2': h12.rmse(q.sub_ec, q.v2),
                    'h20': h12.rmse(q.sub_ec, q.h20)} for k, q in g.groupby('high')}
        early = g[g.hour.le(6)]
        valid_days = g.groupby(['farm', 'day']).flux_hour.max().gt(0)
        day_idx = pd.MultiIndex.from_frame(g[['farm', 'day']])
        having = g[valid_days.reindex(day_idx).to_numpy(bool)]
        summary[str(seed)] = {'v2': h12.rmse(g.sub_ec, g.v2), 'h20': h12.rmse(g.sub_ec, g.h20),
            'base_et': h12.rmse(g.sub_ec, g.base_et), 'moisture_et': h12.rmse(g.sub_ec, g.moisture_et),
            'relative_change': h12.rmse(g.sub_ec, g.h20)/h12.rmse(g.sub_ec, g.v2)-1,
            'bootstrap_ci': ci, 'p_worse': p, 'by_high': subsets,
            'early_hours': {'rows': len(early), 'v2': h12.rmse(early.sub_ec, early.v2),
                'h20': h12.rmse(early.sub_ec, early.h20)},
            'flux_days': {'days': len(having)//24,
                'v2': h12.rmse(having.sub_ec, having.v2) if len(having) else None,
                'h20': h12.rmse(having.sub_ec, having.h20) if len(having) else None},
            'by_farm': {f: h12.rmse(q.sub_ec, q.h20)/h12.rmse(q.sub_ec, q.v2)-1
                        for f, q in g.groupby('farm')}}
    screen = all(s['relative_change'] < 0 for s in scores)
    screen &= all(v['p_worse'] < .025/12 and all(x < 0 for x in v['by_farm'].values())
                  for v in summary.values())
    final = {'scores': scores, 'per_seed': summary, 'screen_pass': bool(screen),
             'moisture_missing_fraction': {c: float(lab[c].isna().mean()) for c in MOIST},
             'final_lock_scored': False, 'test_X_read': False, 'submission_created': False}
    (out / 'result.json').write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'out': str(out), 'screen_pass': screen, 'per_seed': summary},
                     ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=4):
        main()
