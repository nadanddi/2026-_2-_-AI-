"""H11: fixed-offset input-only EC source-chain model, preregistered in PROTOCOL.md."""
import env  # first project import

import hashlib
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
ORIGINAL_LINKS = ROOT / 'research/local/deep_cal_10_links.csv'
FOLDS = (0, 2, 4, 6, 8, 9)
SEEDS = (7, 101)
ACTS = ['act_vent', 'act_shade', 'act_thermal', 'act_heating',
        'act_circfan', 'act_co2', 'act_fog']
INDOOR = ['in_temp', 'in_hum', 'in_co2']
RAW = INDOOR + ACTS
BASE = RAW + ['day', 'hr_sin', 'hr_cos', 'midnight']
FP = [v + '_h0' for v in ACTS + INDOOR]
FP += [name for v in ACTS for name in (v + '_tdm', v + '_tdz')]
FULL = BASE + FP
DAYV = ['rad_sum', 'vpd_sum', 'vent_mean', 'fan_mean', 'heat_mean',
        'thermal_mean', 'fog_mean', 'sealed']
HISTORY = [f'{v}_lag{lag}' for lag in (2, 4) for v in DAYV]
HISTORY += ['rad_sum_2day', 'vpd_sum_2day', 'sealed_run']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identify(raw):
    x = raw.copy()
    x['farm'] = x.row_id.str[:3]
    x['day'] = x.row_id.str[4:7].astype(int)
    x['hour'] = x.row_id.str[8:10].astype(int)
    return x


def features(raw):
    # raw is train_X alone in model CV; this enforces MASK(test_X=NaN).
    x = identify(raw).sort_values(['farm', 'day', 'hour']).reset_index(drop=True)
    assert x.row_id.is_unique and set(x.farm) <= {'F13', 'F47'}
    x['hr_sin'] = np.sin(2 * np.pi * x.hour / 24)
    x['hr_cos'] = np.cos(2 * np.pi * x.hour / 24)
    x['midnight'] = x.hour.eq(0).astype(float)
    g = x.groupby(['farm', 'day'], sort=False)
    h0 = x[x.hour.eq(0)].set_index(['farm', 'day'])
    key = pd.MultiIndex.from_arrays([x.farm, x.day])
    for v in ACTS + INDOOR:
        x[v + '_h0'] = h0[v].reindex(key).values
    for v in ACTS:
        x[v + '_tdm'] = g[v].transform(lambda s: s.expanding().mean())
        x[v + '_tdz'] = g[v].transform(
            lambda s: s.eq(0).astype(float).where(s.notna()).expanding().mean())

    es = 0.61078 * np.exp(17.27 * x.in_temp / (x.in_temp + 237.3))
    x['vpd_hour'] = (es * (1 - x.in_hum / 100)).clip(lower=0)
    x['rad_hour'] = x.out_rad.clip(lower=0)
    day = x.groupby(['farm', 'day'], sort=False).agg(
        rad_sum=('rad_hour', 'sum'), vpd_sum=('vpd_hour', 'sum'),
        vent_mean=('act_vent', 'mean'), fan_mean=('act_circfan', 'mean'),
        heat_mean=('act_heating', 'mean'), thermal_mean=('act_thermal', 'mean'),
        fog_mean=('act_fog', 'mean'),
        vent_zero=('act_vent', lambda s: s.eq(0).mean()),
    ).reset_index()
    day['sealed'] = ((day.vent_zero >= .85) & (day.fan_mean < 10)).astype(float)
    assert x.groupby(['farm', 'day']).size().eq(24).all()
    for lag in (2, 4):
        prev = day[['farm', 'day', *DAYV]].copy()
        prev['day'] += lag
        prev = prev.rename(columns={v: f'{v}_lag{lag}' for v in DAYV})
        x = x.merge(prev, on=['farm', 'day'], how='left', validate='many_to_one')
    x['rad_sum_2day'] = x.rad_sum_lag2 + x.rad_sum_lag4
    x['vpd_sum_2day'] = x.vpd_sum_lag2 + x.vpd_sum_lag4
    x['sealed_run'] = np.where(x.sealed_lag2.eq(1),
                               1 + x.sealed_lag4.eq(1).astype(float), 0.0)
    x.loc[x.sealed_lag2.isna(), 'sealed_run'] = np.nan
    x.loc[x.sealed_lag2.eq(1) & x.sealed_lag4.isna(), 'sealed_run'] = np.nan
    return x[['row_id', 'farm', 'hour', *FULL, *HISTORY]]


def split_mask(frame, days):
    near = {(farm, day + shift) for farm, day in days for shift in (-1, 0, 1)}
    return np.array([(f, int(d)) not in near for f, d in
                     frame[['farm', 'day']].itertuples(index=False, name=None)])


def shrink(p, frame):
    z = frame[['farm', 'day', 'hour']].reset_index(drop=True).copy()
    z['p'] = p
    z = z.sort_values(['farm', 'day', 'hour'])
    avg = z.groupby(['farm', 'day']).p.transform(lambda s: s.expanding().mean())
    out = np.empty(len(p))
    out[z.index.to_numpy()] = (.5 * z.p + .5 * avg).to_numpy()
    return out


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def bootstrap(rows, seed):
    delta = rows.groupby(['farm', 'day']).delta_sq.mean().to_numpy(float)
    assert len(delta) == 234
    rng = np.random.default_rng(seed)
    draws = np.empty(20000)
    for i in range(len(draws)):
        draws[i] = delta[rng.integers(len(delta), size=len(delta))].mean()
    return [float(x) for x in np.quantile(draws, [.025, .975])], float(np.mean(draws >= 0))


def link_diagnostic(lab):
    if not ORIGINAL_LINKS.exists():
        return {'available': False, 'reason': 'analysis-only source-link proxy missing'}
    links = pd.read_csv(ORIGINAL_LINKS)
    links = links[links.d_from.lt(links.d_to)].copy()
    links['hit'] = links.d_from.eq(links.d_to - 2)
    links['period'] = np.where(links.d_to < 179, 'early', 'late')
    valid_days = lab[['farm', 'day']].drop_duplicates()
    proxy = links.merge(valid_days, left_on=['farm', 'd_to'], right_on=['farm', 'day'])
    groups = {}
    for (farm, period), g in proxy.groupby(['farm', 'period']):
        groups[f'{farm}_{period}'] = {'n': len(g), 'hit_rate': float(g.hit.mean())}
    return {'available': True, 'reference': str(ORIGINAL_LINKS),
            'proxy_link_count': len(proxy), 'fixed_lag_hit_rate': float(proxy.hit.mean()),
            'by_farm_period': groups,
            'caveat': 'The analysis-only link proxy is label-checked, not true source ID; never used for features or model selection.'}


def main():
    raw = pd.read_csv(DATA / 'train_X.csv')
    raw = raw[raw.row_id.str[:3].isin(('F13', 'F47'))].reset_index(drop=True)
    assert len(raw) == 9600
    lab = features(raw).merge(pd.read_csv(DATA / 'train_y.csv', usecols=['row_id', 'sub_ec']),
                              on='row_id', validate='one_to_one')
    lab = lab.merge(pd.read_csv(SPLITS)[['row_id', 'fold']], on='row_id', validate='one_to_one')
    assert len(lab) == 9600 and lab.sub_ec.notna().all()
    lock = {(r['farm'], r['day']) for r in json.loads(LOCK.read_text(encoding='utf-8'))['selected']}
    assert len(lock) == 40
    hold = {(f, int(d)) for f, d in lab.loc[lab.fold.isin((8, 9)), ['farm', 'day']]
            .itertuples(index=False, name=None)}
    dev = lab[split_mask(lab, hold)].copy()
    paths = [(SEARCH if f < 8 else CONFIRM) / f'fold{f}.csv' for f in FOLDS]
    hashes = {str(p): sha(p) for p in [HERE / 'PROTOCOL.md', HERE / 'run.py',
                                       DATA / 'train_X.csv', DATA / 'train_y.csv',
                                       SPLITS, LOCK, *paths]}
    out = LOCAL / 'ec_chain_fixed_lag_h11' / datetime.now().strftime('%Y%m%d_%H%M%S')
    out.mkdir(parents=True, exist_ok=False)
    (out / 'manifest.json').write_text(json.dumps({'hashes': hashes, 'folds': FOLDS,
        'seeds': SEEDS, 'features': HISTORY, 'model_share': .2}, ensure_ascii=False, indent=2), encoding='utf-8')
    results = []
    for fold in FOLDS:
        saved = pd.read_csv((SEARCH if fold < 8 else CONFIRM) / f'fold{fold}.csv')
        v2 = saved['blend' if fold < 8 else 'candidate'].to_numpy(float)
        va = lab.set_index('row_id').loc[saved.row_id].reset_index()
        np.testing.assert_allclose(va.sub_ec, saved.sub_ec, atol=1e-12, rtol=0)
        days = {(f, int(d)) for f, d in va[['farm', 'day']].itertuples(index=False, name=None)}
        assert not days & lock
        tr = dev[split_mask(dev, days | lock)].copy() if fold < 8 else dev[split_mask(dev, lock)].copy()
        lo, hi = float(tr.sub_ec.min()), float(tr.sub_ec.max())
        for seed in SEEDS:
            model = make_pipeline(SimpleImputer(strategy='median'), ExtraTreesRegressor(
                n_estimators=400, max_features=1.0, min_samples_leaf=2,
                n_jobs=4, random_state=seed))
            model.fit(tr[FULL + HISTORY], tr.sub_ec.to_numpy(float))
            model.steps[-1][1].n_jobs = 1
            independent = np.clip(shrink(model.predict(va[FULL + HISTORY]), va), lo, hi)
            candidate = .8 * v2 + .2 * independent
            rec = va[['row_id', 'farm', 'day', 'hour', 'sub_ec']].copy()
            rec['fold'] = fold
            rec['seed'] = seed
            rec['v2'] = v2
            rec['h11'] = candidate
            rec.to_csv(out / f'fold{fold}_seed{seed}.csv', index=False, float_format='%.17g')
            a, b = rmse(rec.sub_ec, rec.v2), rmse(rec.sub_ec, rec.h11)
            item = {'fold': fold, 'seed': seed, 'train_days': len(tr) // 24,
                    'v2': a, 'h11': b, 'relative_change': b/a-1}
            results.append(item)
            (out / 'progress.json').write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
            print(json.dumps(item), flush=True)
    rows = pd.concat([pd.read_csv(out / f'fold{f}_seed{s}.csv')
                      for f in FOLDS for s in SEEDS], ignore_index=True)
    rows['delta_sq'] = (rows.sub_ec - rows.h11)**2 - (rows.sub_ec - rows.v2)**2
    high_ids = set(tuple(x) for x in lab.groupby(['farm', 'day']).sub_ec.mean().loc[lambda s: s >= 1.2].index)
    summary = {}
    for seed, g in rows.groupby('seed'):
        ci, p = bootstrap(g, 2911 + int(seed))
        g = g.copy()
        g['high'] = [(f, int(d)) in high_ids for f, d in g[['farm', 'day']].itertuples(index=False, name=None)]
        groups = {}
        for key, q in g.groupby('high'):
            groups['high' if key else 'other'] = {'days': q[['farm', 'day']].drop_duplicates().shape[0],
                'v2': rmse(q.sub_ec, q.v2), 'h11': rmse(q.sub_ec, q.h11)}
        summary[str(seed)] = {'v2': rmse(g.sub_ec, g.v2), 'h11': rmse(g.sub_ec, g.h11),
            'relative_change': rmse(g.sub_ec, g.h11)/rmse(g.sub_ec, g.v2)-1,
            'bootstrap_ci': ci, 'p_worse': p, 'groups': groups,
            'by_farm': {f: rmse(q.sub_ec, q.h11)/rmse(q.sub_ec, q.v2)-1 for f, q in g.groupby('farm')}}
    screen = all(r['relative_change'] < 0 for r in results)
    screen &= all(s['p_worse'] < .025/12 and all(v < 0 for v in s['by_farm'].values())
                  for s in summary.values())
    final = {'scores': results, 'per_seed': summary, 'screen_pass': bool(screen),
             'link_diagnostic': link_diagnostic(lab),
             'history_missing_fraction': {c: float(lab[c].isna().mean()) for c in HISTORY},
             'final_lock_scored': False, 'test_X_read': False, 'submission_created': False}
    (out / 'result.json').write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'out': str(out), 'screen_pass': screen, 'per_seed': summary,
                      'link_diagnostic': final['link_diagnostic']}, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=4):
        main()
