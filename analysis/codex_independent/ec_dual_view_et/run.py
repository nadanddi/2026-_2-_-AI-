"""H8 EC ExtraTrees dual-view training without importing the prior LightGBM stack."""
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
ACTS = ['act_vent', 'act_shade', 'act_thermal', 'act_heating',
        'act_circfan', 'act_co2', 'act_fog']
INDOOR = ['in_temp', 'in_hum', 'in_co2']
RAW = INDOOR + ACTS
BASE = RAW + ['day', 'hr_sin', 'hr_cos', 'midnight']
FP = [v + '_h0' for v in ACTS + INDOOR]
FP += [name for v in ACTS for name in (v + '_tdm', v + '_tdz')]
FULL = BASE + FP


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def identify(raw):
    x = raw.copy()
    x['farm'] = x.row_id.str[:3]
    x['day'] = x.row_id.str[4:7].astype(int)
    x['hour'] = x.row_id.str[8:10].astype(int)
    return x


def features(raw):
    x = identify(raw).sort_values(['farm', 'day', 'hour']).reset_index(drop=True)
    assert x.row_id.is_unique
    x['hr_sin'] = np.sin(2 * np.pi * x.hour / 24)
    x['hr_cos'] = np.cos(2 * np.pi * x.hour / 24)
    x['midnight'] = (x.hour == 0).astype(float)
    g = x.groupby(['farm', 'day'], sort=False)
    h0 = x[x.hour.eq(0)].set_index(['farm', 'day'])
    key = pd.MultiIndex.from_arrays([x.farm, x.day])
    for v in ACTS + INDOOR:
        x[v + '_h0'] = h0[v].reindex(key).values
    for v in ACTS:
        x[v + '_tdm'] = g[v].transform(lambda s: s.expanding().mean())
        x[v + '_tdz'] = g[v].transform(
            lambda s: (s == 0).astype(float).where(s.notna()).expanding().mean())
    return x[['row_id', 'farm', 'hour', *FULL]]


def smoothed(raw):
    x = identify(raw).sort_values(['farm', 'day', 'hour']).reset_index(drop=True)
    for v in INDOOR:
        x[v] = x.groupby(['farm', 'day'], sort=False)[v].transform(
            lambda s: s.ewm(alpha=.5, adjust=False, ignore_na=True).mean())
    return x[raw.columns]


def split_mask(frame, days):
    near = {(farm, day + shift) for farm, day in days for shift in (-1, 0, 1)}
    return np.array([((f, int(d)) not in near) for f, d in
                     frame[['farm', 'day']].itertuples(index=False, name=None)])


def shrink(p, frame):
    x = frame[['farm', 'day', 'hour']].reset_index(drop=True).copy()
    x['p'] = p
    x = x.sort_values(['farm', 'day', 'hour'])
    avg = x.groupby(['farm', 'day']).p.transform(lambda s: s.expanding().mean())
    out = np.empty(len(p))
    out[x.index.values] = (.5 * x.p + .5 * avg).values
    return out


def model(seed):
    return make_pipeline(SimpleImputer(strategy='median'), ExtraTreesRegressor(
        n_estimators=600, max_features=1.0, min_samples_leaf=1,
        n_jobs=4, random_state=seed))


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def prepare():
    raw = pd.read_csv(DATA / 'train_X.csv')
    raw = raw[raw.row_id.str[:3].isin(('F13', 'F47'))].reset_index(drop=True)
    assert len(raw) == 9600 and raw.row_id.is_unique
    y = pd.read_csv(DATA / 'train_y.csv', usecols=['row_id', 'sub_ec'])
    lab = features(raw).merge(y, on='row_id', validate='one_to_one')
    lab = lab.merge(pd.read_csv(SPLITS)[['row_id', 'fold']], on='row_id', validate='one_to_one')
    assert len(lab) == 9600 and lab.sub_ec.notna().all()
    lock = {(r['farm'], r['day']) for r in json.loads(LOCK.read_text(encoding='utf-8'))['selected']}
    assert len(lock) == 40
    hold8 = {(f, int(d)) for f, d in lab.loc[lab.fold.eq(8), ['farm', 'day']].itertuples(index=False, name=None)}
    hold9 = {(f, int(d)) for f, d in lab.loc[lab.fold.eq(9), ['farm', 'day']].itertuples(index=False, name=None)}
    dev = lab[split_mask(lab, hold8 | hold9)].copy()
    return raw, lab, dev, lock


def validation(lab, fold):
    saved = pd.read_csv((SEARCH if fold < 8 else CONFIRM) / f'fold{fold}.csv')
    saved['v2'] = saved['blend'] if fold < 8 else saved['candidate']
    va = lab[lab.fold.eq(fold)].set_index('row_id').loc[saved.row_id].reset_index()
    assert va.row_id.tolist() == saved.row_id.tolist()
    np.testing.assert_allclose(va.sub_ec, saved.sub_ec, rtol=0, atol=1e-12)
    return va, saved.v2.to_numpy(float)


def fit_pair(tr, va, smooth_features, seed):
    original = model(seed)
    original.fit(tr[FULL], tr.sub_ec.to_numpy(float))
    base = original.predict(va[FULL])
    smooth = smooth_features.set_index('row_id').loc[tr.row_id].reset_index()
    assert smooth.row_id.tolist() == tr.row_id.tolist()
    dual_x = pd.concat([tr[FULL], smooth[FULL]], ignore_index=True)
    dual_y = np.tile(tr.sub_ec.to_numpy(float), 2)
    dual_w = np.full(len(dual_y), .5)
    augmented = model(seed)
    augmented.fit(dual_x, dual_y, extratreesregressor__sample_weight=dual_w)
    aug = augmented.predict(va[FULL])
    assert np.isfinite(base).all() and np.isfinite(aug).all()
    return base, aug


def bootstrap(rows, seed):
    delta = rows.groupby(['farm', 'day']).delta_sq.mean().to_numpy(float)
    assert len(delta) == 234
    rng = np.random.default_rng(seed)
    draws = np.empty(20000)
    for i in range(len(draws)):
        draws[i] = delta[rng.integers(len(delta), size=len(delta))].mean()
    return [float(x) for x in np.quantile(draws, [.025, .975])], float(np.mean(draws >= 0))


def main():
    raw, lab, dev, lock = prepare()
    paths = [(SEARCH if f < 8 else CONFIRM) / f'fold{f}.csv' for f in FOLDS]
    hashes = {str(p): sha(p) for p in [HERE / 'PROTOCOL.md', HERE / 'run.py',
                                       DATA / 'train_X.csv', DATA / 'train_y.csv',
                                       SPLITS, LOCK, *paths]}
    out = LOCAL / 'ec_dual_view_et' / datetime.now().strftime('%Y%m%d_%H%M%S')
    out.mkdir(parents=True, exist_ok=False)
    (out / 'manifest.json').write_text(json.dumps(dict(hashes=hashes, folds=FOLDS,
        seeds=SEEDS, alpha=.5, dual_weight=.5, final_scale=.24), ensure_ascii=False, indent=2), encoding='utf-8')
    smooth_features = features(smoothed(raw))
    assert smooth_features.row_id.tolist() == lab.row_id.tolist()
    results = []
    for fold in FOLDS:
        va, v2 = validation(lab, fold)
        val_days = {(f, int(d)) for f, d in va[['farm', 'day']].itertuples(index=False, name=None)}
        assert not (val_days & lock)
        tr = dev[split_mask(dev, val_days | lock)].copy() if fold < 8 else dev[split_mask(dev, lock)].copy()
        assert not set(tr.row_id) & set(va.row_id)
        lo, hi = float(tr.sub_ec.min()), float(tr.sub_ec.max())
        assert np.all((v2 > lo + 1e-10) & (v2 < hi - 1e-10)), 'v2 clipping invalidates additive delta'
        for seed in SEEDS:
            base, aug = fit_pair(tr, va, smooth_features, seed)
            candidate = np.clip(v2 + .24 * shrink(aug - base, va), lo, hi)
            rec = va[['row_id', 'farm', 'day', 'hour', 'sub_ec']].copy()
            rec['fold'] = fold
            rec['seed'] = seed
            rec['v2'] = v2
            rec['h8'] = candidate
            rec.to_csv(out / f'fold{fold}_seed{seed}.csv', index=False, float_format='%.17g')
            a, b = rmse(rec.sub_ec, rec.v2), rmse(rec.sub_ec, rec.h8)
            summary = dict(fold=fold, seed=seed, train_days=len(tr) // 24,
                           v2_rmse=a, h8_rmse=b, relative_change=b/a-1)
            results.append(summary)
            (out / 'progress.json').write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
            print(json.dumps(summary), flush=True)
    rows = pd.concat([pd.read_csv(out / f'fold{f}_seed{s}.csv') for f in FOLDS for s in SEEDS], ignore_index=True)
    rows['delta_sq'] = (rows.sub_ec - rows.h8)**2 - (rows.sub_ec - rows.v2)**2
    summary = {}
    for seed, g in rows.groupby('seed'):
        a, b = rmse(g.sub_ec, g.v2), rmse(g.sub_ec, g.h8)
        ci, p = bootstrap(g, 2909 + int(seed))
        farms = {}
        for farm, z in g.groupby('farm'):
            farms[farm] = rmse(z.sub_ec, z.h8) / rmse(z.sub_ec, z.v2) - 1
        summary[str(seed)] = dict(v2_rmse=a, h8_rmse=b, relative_change=b/a-1,
                                  by_farm=farms, bootstrap_mse_delta_95ci=ci,
                                  bootstrap_p_worse=p)
    passes = all(r['relative_change'] < 0 for r in results)
    passes = passes and all(all(x < 0 for x in z['by_farm'].values())
                            and z['bootstrap_mse_delta_95ci'][1] < 0 for z in summary.values())
    final = dict(scores=results, per_seed=summary, screen_pass=bool(passes),
                 final_lock_scored=False, test_X_read=False, hidden_labels_read=False,
                 submission_created=False)
    (out / 'result.json').write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in final.items() if k != 'scores'}, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=4):
        main()
