"""H12: causal input-only EC source-chain feature ablation."""
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
FOLDS = (0, 2, 4, 6, 8, 9)
SEEDS = (7, 101)
W = ['out_temp', 'out_hum', 'out_rad', 'out_wspd']
WSC = np.array([1.186, 4.83, 50.0, .538])
ISC = {'in_temp': 1.0, 'in_hum': 5.0, 'in_co2': 40.0}
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
HISTORY = [f'{v}_chain{lag}' for lag in (1, 2, 3) for v in DAYV]
HISTORY += ['rad_sum_chain3', 'vpd_sum_chain3', 'sealed_run', 'chain_length', 'link_gap']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identify(raw):
    x = raw.copy()
    x['farm'] = x.row_id.str[:3]
    x['day'] = x.row_id.str[4:7].astype(int)
    x['hour'] = x.row_id.str[8:10].astype(int)
    return x


def wcost(e, d):
    c = 0.0
    for v, scale, weight in [('out_temp', WSC[0], 1.0),
                             ('out_hum', WSC[1], 1.0),
                             ('out_wspd', WSC[3], .3)]:
        z = (d[v][0] - (e[v][23] + .5 * (e[v][23] - e[v][22]))) / scale
        c += 0.0 if np.isnan(z) else weight * z * z
    return c


def icost(e, d):
    c = 0.0
    for v in INDOOR:
        z = (d[v][0] - (e[v][23] + .5 * (e[v][23] - e[v][22]))) / ISC[v]
        c += 0.0 if np.isnan(z) else z * z
    for v in ['act_heating', 'act_thermal', 'act_circfan', 'act_vent']:
        z = abs(d[v][0] - e[v][23]) / 50
        c += 0.0 if np.isnan(z) else z
    return c


def build_links(x):
    """Port of deep_cal_feats.build2 over caller-supplied, same-record inputs."""
    links = {}
    for farm, frame in x.groupby('farm'):
        dd = {int(day): part.set_index('hour').reindex(range(24))
              for day, part in frame.groupby('day')}
        days = sorted(dd)
        w0 = {day: dd[day][W].iloc[0].to_numpy(float) for day in days}
        grp, P = {}, {}
        for i, day in enumerate(days):
            current = dd[day]
            prev = days[:i]
            twin = None
            if prev:
                dist = np.array([np.nanmean(np.abs(w0[p] - w0[day]) / WSC)
                                 for p in prev])
                if np.isfinite(dist).any() and np.nanmin(dist) < .02:
                    twin = prev[int(np.nanargmin(dist))]
            grp[day] = grp[twin] if twin is not None else day
            def members(group, limit):
                return [p for p in days if p < limit and grp.get(p) == group]
            if twin is not None:
                pg = P.get(grp[twin], [])
                pgid = grp[pg[0]] if pg else None
                cand = members(pgid, day) if pgid is not None else []
            else:
                cand = []
                older = [p for p in prev if grp[p] != grp[day]]
                if older:
                    p = older[-1]
                    if wcost(dd[p], current) < 3.0:
                        cand = members(grp[p], day)
                    else:
                        costs = [wcost(dd[q], current) + 2.0*np.log1p(day-q)
                                 for q in older]
                        q = older[int(np.argmin(costs))]
                        cand = members(grp[q], day)
            P[day] = cand
            if grp[day] != day:
                P[grp[day]] = P.get(grp[day]) or cand
            pred = None
            if cand:
                costs = np.array([icost(dd[c], current) for c in cand])
                pred = cand[int(np.argmin(costs))]
            links[(farm, day)] = pred
            assert pred is None or pred < day
    return links


def features(raw):
    # raw is train_X alone in CV. Model evaluation can call separately with
    # train/eval inputs; training features must always use train_X alone.
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
    links = build_links(x)
    summary = {(r.farm, int(r.day)): r for _, r in day.iterrows()}
    history = []
    for farm, d in x[['farm', 'day']].drop_duplicates().itertuples(index=False, name=None):
        prev = links[(farm, int(d))]
        row = {'farm': farm, 'day': int(d), 'link_day': prev,
               'link_gap': np.nan if prev is None else int(d)-prev}
        rad, vpd, seals = [], [], []
        for lag in (1, 2, 3):
            if prev is None:
                for v in DAYV:
                    row[f'{v}_chain{lag}'] = np.nan
            else:
                s = summary[(farm, prev)]
                for v in DAYV:
                    row[f'{v}_chain{lag}'] = float(s[v])
                rad.append(float(s.rad_sum)); vpd.append(float(s.vpd_sum))
                seals.append(float(s.sealed))
                prev = links[(farm, prev)]
        row['rad_sum_chain3'] = np.nan if not rad else float(np.sum(rad))
        row['vpd_sum_chain3'] = np.nan if not vpd else float(np.sum(vpd))
        row['chain_length'] = len(rad)
        row['sealed_run'] = np.nan if not seals else next((i for i, v in enumerate(seals) if v != 1), len(seals))
        history.append(row)
    x = x.merge(pd.DataFrame(history), on=['farm', 'day'], how='left', validate='many_to_one')
    return x[['row_id', 'farm', 'hour', 'link_day', *FULL, *HISTORY]]


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


def causal_checks(raw, original):
    """Future-hour and other-record perturbations cannot change prior features."""
    sample = identify(raw)
    cut_farm, cut_day, cut_hour = 'F13', 100, 12
    keep = sample.farm.eq(cut_farm) & ((sample.day < cut_day) |
        (sample.day.eq(cut_day) & sample.hour.le(cut_hour)))
    future = sample.farm.eq(cut_farm) & ((sample.day > cut_day) |
        (sample.day.eq(cut_day) & sample.hour.gt(cut_hour)))
    for mask in (future, sample.farm.ne(cut_farm)):
        changed = raw.copy()
        cols = W + INDOOR + ACTS
        changed.loc[mask, cols] = changed.loc[mask, cols] * 10 + 999
        rebuilt = features(changed).set_index('row_id')
        rows = sample.loc[keep, 'row_id']
        pd.testing.assert_frame_equal(original.set_index('row_id').loc[rows], rebuilt.loc[rows])


def main():
    raw = pd.read_csv(DATA / 'train_X.csv')
    raw = raw[raw.row_id.str[:3].isin(('F13', 'F47'))].reset_index(drop=True)
    assert len(raw) == 9600
    lab = features(raw).merge(pd.read_csv(DATA / 'train_y.csv', usecols=['row_id', 'sub_ec']),
                              on='row_id', validate='one_to_one')
    lab = lab.merge(pd.read_csv(SPLITS)[['row_id', 'fold']], on='row_id', validate='one_to_one')
    assert len(lab) == 9600 and lab.sub_ec.notna().all()
    causal_checks(raw, lab.drop(columns=['sub_ec', 'fold']))
    lock = {(r['farm'], r['day']) for r in json.loads(LOCK.read_text(encoding='utf-8'))['selected']}
    assert len(lock) == 40
    hold = {(f, int(d)) for f, d in lab.loc[lab.fold.isin((8, 9)), ['farm', 'day']]
            .itertuples(index=False, name=None)}
    dev = lab[split_mask(lab, hold)].copy()
    paths = [(SEARCH if f < 8 else CONFIRM) / f'fold{f}.csv' for f in FOLDS]
    hashes = {str(p): sha(p) for p in [HERE / 'PROTOCOL.md', HERE / 'run.py',
                                       DATA / 'train_X.csv', DATA / 'train_y.csv',
                                       SPLITS, LOCK, *paths]}
    out = LOCAL / 'ec_chain_causal_h12' / datetime.now().strftime('%Y%m%d_%H%M%S')
    out.mkdir(parents=True, exist_ok=False)
    (out / 'manifest.json').write_text(json.dumps({'hashes': hashes, 'folds': FOLDS,
        'seeds': SEEDS, 'features': HISTORY, 'delta_share': .24}, ensure_ascii=False, indent=2), encoding='utf-8')
    links = lab[['farm', 'day', 'link_day']].drop_duplicates()
    assert len(links) == 400
    links.to_csv(out / 'input_only_links.csv', index=False)
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
            def model():
                return make_pipeline(SimpleImputer(strategy='median'), ExtraTreesRegressor(
                    n_estimators=600, max_features=1.0, min_samples_leaf=1,
                    n_jobs=4, random_state=seed))
            base_model, chain_model = model(), model()
            base_model.fit(tr[FULL], tr.sub_ec.to_numpy(float))
            chain_model.fit(tr[FULL + HISTORY], tr.sub_ec.to_numpy(float))
            base_model.steps[-1][1].n_jobs = 1
            chain_model.steps[-1][1].n_jobs = 1
            base = base_model.predict(va[FULL])
            chain = chain_model.predict(va[FULL + HISTORY])
            candidate = np.clip(v2 + .24 * shrink(chain-base, va), lo, hi)
            rec = va[['row_id', 'farm', 'day', 'hour', 'sub_ec']].copy()
            rec['fold'] = fold
            rec['seed'] = seed
            rec['v2'] = v2
            rec['base_et'] = base
            rec['chain_et'] = chain
            rec['h12'] = candidate
            rec.to_csv(out / f'fold{fold}_seed{seed}.csv', index=False, float_format='%.17g')
            a, b = rmse(rec.sub_ec, rec.v2), rmse(rec.sub_ec, rec.h12)
            item = {'fold': fold, 'seed': seed, 'train_days': len(tr) // 24,
                    'v2': a, 'h12': b, 'relative_change': b/a-1,
                    'base_et': rmse(rec.sub_ec, rec.base_et),
                    'chain_et': rmse(rec.sub_ec, rec.chain_et)}
            results.append(item)
            (out / 'progress.json').write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
            print(json.dumps(item), flush=True)
    rows = pd.concat([pd.read_csv(out / f'fold{f}_seed{s}.csv')
                      for f in FOLDS for s in SEEDS], ignore_index=True)
    rows['delta_sq'] = (rows.sub_ec - rows.h12)**2 - (rows.sub_ec - rows.v2)**2
    high_ids = set(tuple(x) for x in lab.groupby(['farm', 'day']).sub_ec.mean().loc[lambda s: s >= 1.2].index)
    summary = {}
    for seed, g in rows.groupby('seed'):
        ci, p = bootstrap(g, 2911 + int(seed))
        g = g.copy()
        g['high'] = [(f, int(d)) in high_ids for f, d in g[['farm', 'day']].itertuples(index=False, name=None)]
        groups = {}
        for key, q in g.groupby('high'):
            groups['high' if key else 'other'] = {'days': q[['farm', 'day']].drop_duplicates().shape[0],
                'v2': rmse(q.sub_ec, q.v2), 'h12': rmse(q.sub_ec, q.h12)}
        summary[str(seed)] = {'v2': rmse(g.sub_ec, g.v2), 'h12': rmse(g.sub_ec, g.h12),
            'base_et': rmse(g.sub_ec, g.base_et), 'chain_et': rmse(g.sub_ec, g.chain_et),
            'relative_change': rmse(g.sub_ec, g.h12)/rmse(g.sub_ec, g.v2)-1,
            'bootstrap_ci': ci, 'p_worse': p, 'groups': groups,
            'by_farm': {f: rmse(q.sub_ec, q.h12)/rmse(q.sub_ec, q.v2)-1 for f, q in g.groupby('farm')}}
    screen = all(r['relative_change'] < 0 for r in results)
    screen &= all(s['p_worse'] < .025/12 and all(v < 0 for v in s['by_farm'].values())
                  for s in summary.values())
    final = {'scores': results, 'per_seed': summary, 'screen_pass': bool(screen),
             'link_diagnostic': 'Use input_only_links.csv with analysis-only proxy after scoring',
             'history_missing_fraction': {c: float(lab[c].isna().mean()) for c in HISTORY},
             'final_lock_scored': False, 'test_X_read': False, 'submission_created': False}
    (out / 'result.json').write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'out': str(out), 'screen_pass': screen, 'per_seed': summary},
                     ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=4):
        main()
