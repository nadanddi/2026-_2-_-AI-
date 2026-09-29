"""H17 compact 7-day causal source-chain physical memory."""
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
LONG = ['sealed_run7', 'sealed_count7', 'rad_memory7', 'vpd_memory7',
        'vent_memory7', 'fog_memory7']
FOLDS = (0, 2, 4, 6, 8, 9)
SEEDS = (7, 101)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def long_features(raw):
    x = h12.identify(raw).sort_values(['farm', 'day', 'hour']).reset_index(drop=True)
    assert len(x) == 9600
    links = h12.build_links(x)
    es = .61078*np.exp(17.27*x.in_temp/(x.in_temp+237.3))
    x['vpd_hour'] = (es*(1-x.in_hum/100)).clip(lower=0)
    x['rad_hour'] = x.out_rad.clip(lower=0)
    d = x.groupby(['farm', 'day']).agg(rad=('rad_hour', 'sum'),
        vpd=('vpd_hour', 'sum'), vent=('act_vent', 'mean'),
        fog=('act_fog', 'mean'), fan=('act_circfan', 'mean'),
        vent_zero=('act_vent', lambda s: s.eq(0).mean())).reset_index()
    d['sealed'] = d.vent_zero.ge(.85) & d.fan.lt(10)
    by = {(r.farm, int(r.day)): r for _, r in d.iterrows()}
    rows = []
    for farm, day in d[['farm', 'day']].itertuples(index=False, name=None):
        day = int(day)
        prev = links[(farm, day)]
        ancestors = []
        for _ in range(7):
            if prev is None:
                break
            ancestors.append(int(prev))
            prev = links[(farm, int(prev))]
        r = {'farm': farm, 'day': day}
        if not ancestors:
            r.update({v: np.nan for v in LONG})
        else:
            summary = [by[(farm, a)] for a in ancestors]
            seals = [bool(q.sealed) for q in summary]
            r['sealed_run7'] = next((i for i, v in enumerate(seals) if not v), len(seals))
            r['sealed_count7'] = sum(seals)
            weights = .7**np.arange(len(ancestors))
            for src, dest in [('rad', 'rad_memory7'), ('vpd', 'vpd_memory7'),
                              ('vent', 'vent_memory7'), ('fog', 'fog_memory7')]:
                values = np.array([q[src] for q in summary], dtype=float)
                valid = np.isfinite(values)
                r[dest] = np.nan if not valid.any() else float(np.average(values[valid], weights=weights[valid]))
        r['link_day'] = links[(farm, day)]
        rows.append(r)
    z = pd.DataFrame(rows)
    assert len(z) == 400
    return z


def model(seed):
    return make_pipeline(SimpleImputer(strategy='median'), ExtraTreesRegressor(
        n_estimators=600, max_features=1.0, min_samples_leaf=1,
        n_jobs=4, random_state=seed))


def main():
    raw = pd.read_csv(DATA / 'train_X.csv')
    raw = raw[raw.row_id.str[:3].isin(('F13', 'F47'))].reset_index(drop=True)
    base = h12.features(raw)[['row_id', 'farm', 'hour', *FULL]]
    memory = long_features(raw)
    lab = base.merge(memory, on=['farm', 'day'], validate='many_to_one')
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
    out = LOCAL / 'ec_long_sealed_chain_h17' / datetime.now().strftime('%Y%m%d_%H%M%S')
    out.mkdir(parents=True, exist_ok=False)
    (out / 'manifest.json').write_text(json.dumps({'hashes': hashes, 'folds': FOLDS,
        'seeds': SEEDS, 'long_features': LONG, 'decay': .7, 'delta_share': .24},
        ensure_ascii=False, indent=2), encoding='utf-8')
    memory[['farm', 'day', 'link_day', *LONG]].to_csv(out / 'input_only_memory.csv', index=False)
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
            lm.fit(tr[FULL + LONG], tr.sub_ec.to_numpy(float))
            bm.steps[-1][1].n_jobs = 1; lm.steps[-1][1].n_jobs = 1
            bp = bm.predict(va[FULL]); lp = lm.predict(va[FULL + LONG])
            candidate = np.clip(v2 + .24*h12.shrink(lp-bp, va), lo, hi)
            rec = va[['row_id', 'farm', 'day', 'hour', 'sub_ec', 'high', 'sealed_run7']].copy()
            rec['fold'] = fold; rec['seed'] = seed
            rec['v2'] = v2; rec['base_et'] = bp; rec['long_et'] = lp; rec['h17'] = candidate
            rec.to_csv(out / f'fold{fold}_seed{seed}.csv', index=False, float_format='%.17g')
            a, b = h12.rmse(rec.sub_ec, rec.v2), h12.rmse(rec.sub_ec, rec.h17)
            item = {'fold': fold, 'seed': seed, 'train_days': len(tr)//24,
                'v2': a, 'h17': b, 'relative_change': b/a-1,
                'base_et': h12.rmse(rec.sub_ec, bp), 'long_et': h12.rmse(rec.sub_ec, lp)}
            scores.append(item)
            (out / 'progress.json').write_text(json.dumps(scores, ensure_ascii=False, indent=2), encoding='utf-8')
            print(json.dumps(item), flush=True)
    rows = pd.concat([pd.read_csv(out / f'fold{f}_seed{s}.csv') for f in FOLDS for s in SEEDS], ignore_index=True)
    rows['delta_sq'] = (rows.sub_ec-rows.h17)**2-(rows.sub_ec-rows.v2)**2
    summary = {}
    for seed, g in rows.groupby('seed'):
        ci, p = h12.bootstrap(g, 2917+int(seed))
        subsets = {str(bool(k)): {'days': len(q)//24, 'v2': h12.rmse(q.sub_ec, q.v2),
                    'h17': h12.rmse(q.sub_ec, q.h17)} for k, q in g.groupby('high')}
        long = g[g.sealed_run7.ge(4)]
        summary[str(seed)] = {'v2': h12.rmse(g.sub_ec, g.v2), 'h17': h12.rmse(g.sub_ec, g.h17),
            'base_et': h12.rmse(g.sub_ec, g.base_et), 'long_et': h12.rmse(g.sub_ec, g.long_et),
            'relative_change': h12.rmse(g.sub_ec, g.h17)/h12.rmse(g.sub_ec, g.v2)-1,
            'bootstrap_ci': ci, 'p_worse': p, 'by_high': subsets,
            'sealed_4plus': {'days': len(long)//24,
                'v2': h12.rmse(long.sub_ec, long.v2) if len(long) else None,
                'h17': h12.rmse(long.sub_ec, long.h17) if len(long) else None},
            'by_farm': {f: h12.rmse(q.sub_ec, q.h17)/h12.rmse(q.sub_ec, q.v2)-1
                        for f, q in g.groupby('farm')}}
    screen = all(s['relative_change'] < 0 for s in scores)
    screen &= all(v['p_worse'] < .025/12 and all(x < 0 for x in v['by_farm'].values())
                  for v in summary.values())
    final = {'scores': scores, 'per_seed': summary, 'screen_pass': bool(screen),
             'long_missing_fraction': {c: float(lab[c].isna().mean()) for c in LONG},
             'final_lock_scored': False, 'test_X_read': False, 'submission_created': False}
    (out / 'result.json').write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'out': str(out), 'screen_pass': screen, 'per_seed': summary},
                     ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=4):
        main()
