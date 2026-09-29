"""H13 fold-local source-identity operational fingerprint model."""
import env  # first project import

import hashlib
import importlib.util
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
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
H12CODE = HERE.parent / 'ec_chain_causal_h12/run_v2.py'
spec = importlib.util.spec_from_file_location('h12_input_linker', H12CODE)
h12 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h12)
FOLDS = (0, 2, 4, 6, 8, 9)
SEEDS = (7, 101)
ACTS = h12.ACTS
FULL = h12.FULL
HIST = [f'identity_prev_{v}' for v in ACTS]
DIST = [f'identity_kmeans_dist{i}' for i in range(4)]
F2 = HIST + DIST


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_profiles(raw):
    x = h12.identify(raw).sort_values(['farm', 'day', 'hour']).reset_index(drop=True)
    links = h12.build_links(x)
    night = x[x.hour.le(6)].groupby(['farm', 'day'])[ACTS].mean()
    h0 = x[x.hour.eq(0)].set_index(['farm', 'day'])[ACTS]
    rows = []
    for farm, day in night.index:
        day = int(day)
        prev = links[(farm, day)]
        sums = np.zeros(len(ACTS), dtype=float)
        weight = np.zeros(len(ACTS), dtype=float)
        for lag, w in enumerate((1.0, .5, .25), start=1):
            if prev is None:
                break
            val = night.loc[(farm, prev)].to_numpy(float)
            good = np.isfinite(val)
            sums[good] += w * val[good]
            weight[good] += w
            prev = links[(farm, prev)]
        hist = np.divide(sums, weight, out=np.full(len(ACTS), np.nan), where=weight > 0)
        row = {'farm': farm, 'day': day}
        row.update({name: value for name, value in zip(HIST, hist)})
        row.update({f'identity_h0_{v}': value for v, value in
                    zip(ACTS, h0.loc[(farm, day)].to_numpy(float))})
        rows.append(row)
    profile = pd.DataFrame(rows)
    assert len(profile) == 400
    return profile


def fold_distances(profile, train_days, valid_days):
    """Fit imputer/scaler/clusters on training days only, separately per farm."""
    raw_cols = [f'identity_h0_{v}' for v in ACTS] + HIST
    parts = []
    for farm in ('F13', 'F47'):
        z = profile[profile.farm.eq(farm)].copy()
        train = z.day.isin({d for f, d in train_days if f == farm})
        valid = z.day.isin({d for f, d in valid_days if f == farm})
        assert train.sum() >= 4 and not (train & valid).any()
        transform = make_pipeline(SimpleImputer(strategy='median'), StandardScaler())
        fitx = transform.fit_transform(z.loc[train, raw_cols])
        kmeans = KMeans(n_clusters=4, n_init=10, random_state=2913)
        kmeans.fit(fitx)
        use = train | valid
        d = kmeans.transform(transform.transform(z.loc[use, raw_cols])) ** 2
        q = z.loc[use, ['farm', 'day', *HIST]].copy()
        for j, col in enumerate(DIST):
            q[col] = d[:, j]
        parts.append(q)
    return pd.concat(parts, ignore_index=True)


def model(seed):
    return make_pipeline(SimpleImputer(strategy='median'), ExtraTreesRegressor(
        n_estimators=600, max_features=1.0, min_samples_leaf=1,
        n_jobs=4, random_state=seed))


def main():
    raw = pd.read_csv(DATA / 'train_X.csv')
    raw = raw[raw.row_id.str[:3].isin(('F13', 'F47'))].reset_index(drop=True)
    assert len(raw) == 9600
    full = h12.features(raw)[['row_id', 'farm', 'hour', *FULL]]
    profile = source_profiles(raw)
    lab = full.merge(pd.read_csv(DATA / 'train_y.csv', usecols=['row_id', 'sub_ec']),
                     on='row_id', validate='one_to_one')
    lab = lab.merge(pd.read_csv(SPLITS)[['row_id', 'fold']], on='row_id', validate='one_to_one')
    assert len(lab) == 9600 and lab.sub_ec.notna().all()
    lock = {(r['farm'], r['day']) for r in json.loads(LOCK.read_text(encoding='utf-8'))['selected']}
    hold = {(f, int(d)) for f, d in lab.loc[lab.fold.isin((8, 9)), ['farm', 'day']]
            .itertuples(index=False, name=None)}
    dev = lab[h12.split_mask(lab, hold)].copy()
    paths = [(SEARCH if f < 8 else CONFIRM) / f'fold{f}.csv' for f in FOLDS]
    hashes = {str(p): sha(p) for p in [HERE / 'PROTOCOL.md', HERE / 'run.py',
                                       H12CODE, DATA / 'train_X.csv', DATA / 'train_y.csv',
                                       SPLITS, LOCK, *paths]}
    out = LOCAL / 'ec_source_identity_h13' / datetime.now().strftime('%Y%m%d_%H%M%S')
    out.mkdir(parents=True, exist_ok=False)
    (out / 'manifest.json').write_text(json.dumps({'hashes': hashes, 'folds': FOLDS,
        'seeds': SEEDS, 'fingerprint': HIST, 'distances': DIST, 'delta_share': .24},
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
        train_days = {(f, int(d)) for f, d in tr[['farm', 'day']].itertuples(index=False, name=None)}
        embedding = fold_distances(profile, train_days, val_days)
        tr = tr.merge(embedding, on=['farm', 'day'], validate='many_to_one')
        va = va.merge(embedding, on=['farm', 'day'], validate='many_to_one')
        assert len(va) == len(saved) and va.row_id.tolist() == saved.row_id.tolist()
        lo, hi = float(tr.sub_ec.min()), float(tr.sub_ec.max())
        for seed in SEEDS:
            baseline, enriched = model(seed), model(seed)
            baseline.fit(tr[FULL], tr.sub_ec.to_numpy(float))
            enriched.fit(tr[FULL + F2], tr.sub_ec.to_numpy(float))
            baseline.steps[-1][1].n_jobs = 1
            enriched.steps[-1][1].n_jobs = 1
            bp = baseline.predict(va[FULL]); ep = enriched.predict(va[FULL + F2])
            candidate = np.clip(v2 + .24*h12.shrink(ep-bp, va), lo, hi)
            rec = va[['row_id', 'farm', 'day', 'hour', 'sub_ec']].copy()
            rec['fold'] = fold; rec['seed'] = seed
            rec['v2'] = v2; rec['base_et'] = bp; rec['identity_et'] = ep; rec['h13'] = candidate
            rec.to_csv(out / f'fold{fold}_seed{seed}.csv', index=False, float_format='%.17g')
            a, b = h12.rmse(rec.sub_ec, rec.v2), h12.rmse(rec.sub_ec, rec.h13)
            item = {'fold': fold, 'seed': seed, 'train_days': len(tr)//24,
                'v2': a, 'h13': b, 'relative_change': b/a-1,
                'base_et': h12.rmse(rec.sub_ec, bp), 'identity_et': h12.rmse(rec.sub_ec, ep)}
            scores.append(item)
            (out / 'progress.json').write_text(json.dumps(scores, ensure_ascii=False, indent=2), encoding='utf-8')
            print(json.dumps(item), flush=True)
    rows = pd.concat([pd.read_csv(out / f'fold{f}_seed{s}.csv') for f in FOLDS for s in SEEDS], ignore_index=True)
    rows['delta_sq'] = (rows.sub_ec-rows.h13)**2-(rows.sub_ec-rows.v2)**2
    high = set(tuple(x) for x in lab.groupby(['farm', 'day']).sub_ec.mean().loc[lambda s: s >= 1.2].index)
    summary = {}
    for seed, g in rows.groupby('seed'):
        ci, p = h12.bootstrap(g, 2913+int(seed))
        g = g.copy()
        g['high'] = [(f, int(d)) in high for f, d in g[['farm', 'day']].itertuples(index=False, name=None)]
        subsets = {str(bool(k)): {'days': len(q)//24, 'v2': h12.rmse(q.sub_ec, q.v2),
                    'h13': h12.rmse(q.sub_ec, q.h13)} for k, q in g.groupby('high')}
        summary[str(seed)] = {'v2': h12.rmse(g.sub_ec, g.v2), 'h13': h12.rmse(g.sub_ec, g.h13),
            'base_et': h12.rmse(g.sub_ec, g.base_et), 'identity_et': h12.rmse(g.sub_ec, g.identity_et),
            'relative_change': h12.rmse(g.sub_ec, g.h13)/h12.rmse(g.sub_ec, g.v2)-1,
            'bootstrap_ci': ci, 'p_worse': p, 'by_high': subsets,
            'by_farm': {f: h12.rmse(q.sub_ec, q.h13)/h12.rmse(q.sub_ec, q.v2)-1
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
