"""Causal indoor-prefix feature experiment; confirmation folds remain closed."""
import sys
from pathlib import Path

V1 = Path(__file__).resolve().parents[1] / 'rl_ec_v1'
sys.path.insert(0, str(V1))
import env  # first project import
import run as core

import hashlib
import json
import platform
from datetime import datetime

import numpy as np
import pandas as pd
import sklearn
import lightgbm
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
SPLITS = ROOT / 'local/rl_ec_v1/20260927_173801/splits.csv'
KINDS = {
    'prefix_moments': ['mean', 'std', 'min', 'max', 'devmean'],
    'prefix_robust': ['median', 'iqr', 'devmedian', 'observed'],
}
COLS = {a: [f'{v}__{s}' for v in core.INDOOR for s in stats] for a, stats in KINDS.items()}


def extra_features(raw):
    a = core.identify(raw).sort_values(['farm', 'day', 'hour']).reset_index(drop=True)
    g = a.groupby(['farm', 'day'], sort=False)
    out = pd.DataFrame({'row_id': a.row_id})
    for v in core.INDOOR:
        mean = g[v].transform(lambda s: s.expanding().mean())
        median = g[v].transform(lambda s: s.expanding().median())
        out[f'{v}__mean'] = mean
        out[f'{v}__std'] = g[v].transform(lambda s: s.expanding().std(ddof=0))
        out[f'{v}__min'] = g[v].transform(lambda s: s.expanding().min())
        out[f'{v}__max'] = g[v].transform(lambda s: s.expanding().max())
        out[f'{v}__devmean'] = a[v] - mean
        out[f'{v}__median'] = median
        out[f'{v}__iqr'] = g[v].transform(
            lambda s: s.expanding().quantile(.75) - s.expanding().quantile(.25))
        out[f'{v}__devmedian'] = a[v] - median
        out[f'{v}__observed'] = g[v].transform(lambda s: s.notna().expanding().mean())
    return out


def checks(raw):
    a = core.identify(raw)
    x = extra_features(raw).set_index('row_id')
    pd.testing.assert_frame_equal(x, extra_features(raw.sample(frac=1, random_state=912)).set_index('row_id'))
    trials = 0
    for farm in ['F13', 'F47']:
        days = sorted(a[a.farm.eq(farm)].day.unique())
        for day in [days[3], days[len(days)//2], days[-4]]:
            for hour in [0, 7, 16]:
                before = a.farm.eq(farm) & ((a.day < day) | ((a.day == day) & (a.hour <= hour)))
                later = a.farm.eq(farm) & ~before
                # Test both perturbation and actual deletion of all future inputs.
                changed = raw.copy()
                changed.loc[later, core.INDOOR] = changed.loc[later, core.INDOOR] * 17 + 903
                ids = a.loc[before, 'row_id']
                pd.testing.assert_frame_equal(x.loc[ids], extra_features(changed).set_index('row_id').loc[ids])
                pd.testing.assert_frame_equal(x.loc[ids], extra_features(raw[before]).set_index('row_id').loc[ids])
                changed = raw.copy()
                changed.loc[a.farm.ne(farm), core.INDOOR] = -999
                pd.testing.assert_frame_equal(x.loc[ids], extra_features(changed).set_index('row_id').loc[ids])
                trials += 1
    return dict(cutoffs=trials, future_perturbation='PASS', actual_prefix_only='PASS',
                other_farm='PASS', row_order='PASS', test_X_read=False,
                confirmation_scored=False)


def cache_key(tr, va, seed):
    h = hashlib.sha256()
    for frame in [tr[['row_id', 'sub_ec'] + core.FULL], va[['row_id'] + core.FULL]]:
        h.update(pd.util.hash_pandas_object(frame, index=False).values.tobytes())
    h.update(json.dumps([seed, core.sha(V1 / 'run.py'), core.sha(V1 / 'env.py'),
                         sklearn.__version__, lightgbm.__version__, np.__version__]).encode())
    return h.hexdigest()


def get_baseline(tr, va, seed, log):
    cache = ROOT / 'local/ec_baseline_cache'
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / (cache_key(tr, va, seed) + '.npz')
    if path.exists():
        with np.load(path) as z:
            assert np.array_equal(z['row_id'], va.row_id.to_numpy().astype(str))
            mem = [z[k].copy() for k in ['et', 'lgb', 'mlp']]
        log('Using exact-input baseline cache: ' + path.name[:12])
    else:
        mem = core.members(tr, va, seed)
        np.savez(path, row_id=va.row_id.to_numpy().astype(str), et=mem[0], lgb=mem[1], mlp=mem[2])
    return mem


def prepare():
    raw = pd.read_csv(env.DATA / 'train_X.csv')
    raw = raw[raw.row_id.str[:3].isin(['F13', 'F47'])].reset_index(drop=True)
    label = pd.read_csv(env.DATA / 'train_y.csv', usecols=['row_id', 'sub_ec'])
    lab = core.features(raw).merge(label, on='row_id', validate='one_to_one').dropna(subset=['sub_ec'])
    splits = pd.read_csv(SPLITS)
    assert set(lab.row_id) == set(splits.row_id)
    lab = lab.merge(splits[['row_id', 'fold', 'block']], on='row_id', validate='one_to_one')
    allowed = core.split(lab, 8)[0] & core.split(lab, 9)[0]
    return raw, lab, allowed


def main():
    out = ROOT / 'local/rl_ec_v2' / datetime.now().strftime('%Y%m%d_%H%M%S')
    out.mkdir(parents=True, exist_ok=False)
    def save(name, obj):
        (out / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
    def log(message):
        print(message, flush=True)
        with (out / 'progress.log').open('a', encoding='utf-8') as f:
            f.write(message + '\n')
    save('manifest.json', dict(code_hash=core.sha(__file__), v1_hash=core.sha(V1/'run.py'),
         env_hash=core.sha(V1/'env.py'), protocol_hash=core.sha(Path(__file__).with_name('PROTOCOL.md')),
         splits_hash=core.sha(SPLITS), inputs={n: core.sha(env.DATA/n) for n in ['train_X.csv', 'train_y.csv']},
         python=platform.python_version(), sklearn=sklearn.__version__, lightgbm=lightgbm.__version__,
         numpy=np.__version__, seeds=core.SEEDS, arms=COLS))
    log('OUTPUT ' + str(out))
    raw, lab, allowed = prepare()
    # A small, prechosen set of days is sufficient for structural perturbation tests.
    ids = core.identify(raw)
    testdays = ids.groupby('farm').day.transform(lambda s: s.isin(sorted(s.unique())[::17]))
    save('checks.json', checks(raw[testdays].reset_index(drop=True)))
    log('Prefix, future, farm-isolation and row-order checks PASS')
    enriched = lab.merge(extra_features(raw), on='row_id', validate='one_to_one')
    pd.testing.assert_frame_equal(lab[core.FULL], enriched[core.FULL])
    dev = enriched[allowed].reset_index(drop=True)
    log(f'Search rows {len(dev)}; confirmation rows remain unscored')
    history = []
    def evaluate(fold, seed, arms):
        trm, vam = core.split(dev, fold)
        tr, va = dev[trm], dev[vam].reset_index(drop=True)
        log(f'Training fold={fold} seed={seed} arms={arms}')
        mem = get_baseline(tr, va, seed, log)
        base = core.finish(mem, tr, va)
        predictions = va[['row_id', 'farm', 'day', 'hour', 'block', 'sub_ec']].copy()
        predictions['baseline'] = base
        for arm in arms:
            ep = core.predict_model(core.et(seed), tr, va, core.FULL + COLS[arm])
            pred = core.finish([ep, mem[1], mem[2]], tr, va)
            a, b = core.rmse(va.sub_ec, base), core.rmse(va.sub_ec, pred)
            record = dict(arm=arm, fold=fold, seed=seed, baseline=a, candidate=b, reward=1-b/a)
            history.append(record)
            save('history.json', history)
            predictions[arm] = pred
            log(f'{arm} fold={fold} seed={seed}: {a:.6f} -> {b:.6f} ({b/a-1:+.3%})')
        predictions.to_csv(out / f'search_fold{fold}_seed{seed}.csv', index=False)
    for seed in core.SEEDS:
        for fold in [0, 2]:
            evaluate(fold, seed, list(KINDS))
    initial = {a: [r['reward'] for r in history if r['arm'] == a] for a in KINDS}
    eligible = [a for a, values in initial.items() if min(values) > 0 and np.mean(values) >= .01]
    save('screen_lock.json', dict(initial_rewards=initial, eligible=eligible))
    log('Eligible for extended search: ' + repr(eligible))
    if eligible:
        for seed in core.SEEDS:
            for fold in [4, 6]:
                evaluate(fold, seed, eligible)
    summary = {}
    for arm in KINDS:
        rr = [r['reward'] for r in history if r['arm'] == arm]
        summary[arm] = dict(n=len(rr), mean_relative_rmse_change=-float(np.mean(rr)),
                            improved=sum(r > 0 for r in rr),
                            passes_search=len(rr) == 8 and min(rr) > 0 and np.mean(rr) >= .01)
    result = dict(summary=summary, adopted=False, confirmation_scored=False,
                  note='Search only; locked confirmation not opened; no significance claim.')
    save('result.json', result)
    log('FINISHED ' + json.dumps(result))


if __name__ == '__main__':
    with threadpool_limits(limits=4):
        main()
