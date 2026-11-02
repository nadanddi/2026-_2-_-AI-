import env  # first project import

import hashlib
import importlib.util
import json
import math
import platform
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ACTS = ['act_vent', 'act_shade', 'act_thermal', 'act_heating',
        'act_circfan', 'act_co2', 'act_fog']
INDOOR = ['in_temp', 'in_hum', 'in_co2']
RAW = INDOOR + ACTS
BASE = RAW + ['day', 'hr_sin', 'hr_cos', 'midnight']
FP = [v + '_h0' for v in ACTS + INDOOR]
FP += [name for v in ACTS for name in (v + '_tdm', v + '_tdz')]
FULL = BASE + FP
ARMS = ['leaf4', 'daybag', 'l2fp']
SEEDS = [7, 101]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identify(x):
    x = x.copy()
    x['farm'] = x.row_id.str[:3]
    x['day'] = x.row_id.str[4:7].astype(int)
    x['hour'] = x.row_id.str[8:10].astype(int)
    return x


def features(raw):
    a = identify(raw).sort_values(['farm', 'day', 'hour']).reset_index(drop=True)
    assert a.row_id.is_unique
    a['hr_sin'] = np.sin(2 * np.pi * a.hour / 24)
    a['hr_cos'] = np.cos(2 * np.pi * a.hour / 24)
    a['midnight'] = (a.hour == 0).astype(float)
    g = a.groupby(['farm', 'day'], sort=False)
    h0 = a[a.hour == 0].set_index(['farm', 'day'])
    key = pd.MultiIndex.from_arrays([a.farm, a.day])
    for v in ACTS + INDOOR:
        a[v + '_h0'] = h0[v].reindex(key).values
    for v in ACTS:
        a[v + '_tdm'] = g[v].transform(lambda s: s.expanding().mean())
        a[v + '_tdz'] = g[v].transform(
            lambda s: (s == 0).astype(float).where(s.notna()).expanding().mean())
    return a[['row_id', 'farm', 'hour'] + FULL]


def shrink(p, frame):
    d = frame[['farm', 'day', 'hour']].reset_index(drop=True).copy()
    d['p'] = p
    d = d.sort_values(['farm', 'day', 'hour'])
    avg = d.groupby(['farm', 'day']).p.transform(lambda s: s.expanding().mean())
    out = np.empty(len(p))
    out[d.index.values] = (0.5 * d.p + 0.5 * avg).values
    return out


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def split(lab, fold):
    va = lab.fold.eq(fold).to_numpy()
    tr = np.ones(len(lab), dtype=bool)
    for farm, g in lab[va].groupby('farm'):
        days = set(g.day)
        near = days | {d - 1 for d in days} | {d + 1 for d in days}
        tr &= ~(lab.farm.eq(farm) & lab.day.isin(near)).to_numpy()
    assert not (tr & va).any()
    return tr, va


def et(seed, leaf=1, trees=600):
    return make_pipeline(SimpleImputer(strategy='median'), ExtraTreesRegressor(
        n_estimators=trees, max_features=1.0, min_samples_leaf=leaf,
        n_jobs=4, random_state=seed))


def predict_model(m, tr, va, cols, weight=None):
    kwargs = {} if weight is None else {'extratreesregressor__sample_weight': weight}
    m.fit(tr[cols], tr.sub_ec.to_numpy(), **kwargs)
    if hasattr(m, 'steps') and isinstance(m.steps[-1][1], ExtraTreesRegressor):
        m.steps[-1][1].n_jobs = 1
    return m.predict(va[cols])


def lg(seed, objective):
    opts = dict(n_estimators=800, learning_rate=0.03, num_leaves=31,
                min_child_samples=40, subsample=0.8, subsample_freq=1,
                colsample_bytree=0.8, reg_lambda=1.0, deterministic=True,
                force_col_wise=True, n_jobs=4, verbose=-1, random_state=seed,
                objective=objective)
    if objective == 'tweedie':
        opts['tweedie_variance_power'] = 1.5
    return lgb.LGBMRegressor(**opts)


def members(tr, va, seed):
    mlp = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(),
        MLPRegressor(hidden_layer_sizes=(128, 64), alpha=1e-2,
                     learning_rate_init=1e-3, max_iter=800, early_stopping=True,
                     n_iter_no_change=25, validation_fraction=0.12, random_state=seed))
    return [predict_model(et(seed), tr, va, FULL),
            predict_model(lg(seed, 'tweedie'), tr, va, BASE),
            predict_model(mlp, tr, va, BASE)]


def candidate(arm, tr, va, seed, base):
    e, l, m = base
    if arm == 'leaf4':
        e = predict_model(et(seed, leaf=4), tr, va, FULL)
    elif arm == 'daybag':
        codes, days = pd.factorize(pd.MultiIndex.from_frame(tr[['farm', 'day']]))
        rng = np.random.default_rng(seed)
        predictions = []
        for b in range(8):
            counts = np.bincount(rng.integers(0, len(days), len(days)), minlength=len(days))
            weights = counts[codes]
            keep = weights > 0
            predictions.append(predict_model(et(seed + b, trees=75), tr[keep], va,
                                             FULL, weights[keep]))
        e = np.mean(predictions, axis=0)
    elif arm == 'l2fp':
        l = predict_model(lg(seed, 'regression'), tr, va, FULL)
    else:
        raise ValueError(arm)
    return [e, l, m]


def finish(mem, tr, va):
    return np.clip(shrink(.6 * mem[0] + .3 * mem[1] + .1 * mem[2], va),
                   tr.sub_ec.min(), tr.sub_ec.max())


def checks(raw):
    original = features(raw).set_index('row_id')
    shuffled = features(raw.sample(frac=1, random_state=812)).set_index('row_id')
    pd.testing.assert_frame_equal(original, shuffled)
    for farm in ['F13', 'F47']:
        rows = identify(raw)
        times = rows.day * 24 + rows.hour
        for frac in [.2, .5, .8]:
            cut = int(times[rows.farm.eq(farm)].quantile(frac))
            prior = rows.farm.eq(farm) & times.le(cut)
            future = rows.farm.eq(farm) & times.gt(cut)
            for mask in [future, rows.farm.ne(farm)]:
                changed = raw.copy()
                changed.loc[mask, RAW] = changed.loc[mask, RAW] * 10 + 999
                new = features(changed).set_index('row_id')
                pd.testing.assert_frame_equal(original.loc[rows.loc[prior, 'row_id']],
                                              new.loc[rows.loc[prior, 'row_id']])
    # Future nonmissing observations must never replace missing midnight input.
    changed = raw.copy()
    changed.loc[changed.row_id.str.endswith('_00'), RAW] = np.nan
    built = features(changed)
    assert built[[v + '_h0' for v in RAW]].isna().all().all()
    # Compare fingerprint implementation directly with read-only original code.
    import types
    import sys
    fake = types.ModuleType('common')
    a = identify(raw)
    a['t'] = a.day * 24 + a.hour
    fake.TARGET_FARMS = ['F13', 'F47']
    fake.load_raw = lambda: (a, None, a.iloc[:0].copy())
    sys.modules['common'] = fake
    spec = importlib.util.spec_from_file_location('original_fp', env.SOURCE / 'research/fp_features.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    old = mod.build().set_index('row_id')
    pd.testing.assert_frame_equal(original[FP], old.loc[original.index, FP], check_dtype=False)
    del sys.modules['common']
    return dict(future_input='PASS: 6 cutoffs', other_farm='PASS: 6 cutoffs',
                row_order='PASS', missing_midnight='PASS', original_fp_parity='PASS',
                test_X_read=False, target_features=False)


def interval(frame, y, base, cand):
    # Resample contiguous 5-day blocks, preserving all hours in a sampled block.
    block = frame.groupby(['farm', 'block']).indices
    n = np.array([len(i) for i in block.values()])
    a = np.array([np.sum((base[i] - y[i]) ** 2) for i in block.values()])
    b = np.array([np.sum((cand[i] - y[i]) ** 2) for i in block.values()])
    rng = np.random.default_rng(270927)
    idx = rng.integers(0, len(n), (12000, len(n)))
    delta = np.sqrt(b[idx].sum(1) / n[idx].sum(1)) - np.sqrt(a[idx].sum(1) / n[idx].sum(1))
    return [float(v) for v in np.quantile(delta, [.05 / 6, 1 - .05 / 6])]


def main():
    out = Path(__file__).resolve().parents[2] / 'local/rl_ec_v1' / datetime.now().strftime('%Y%m%d_%H%M%S')
    out.mkdir(parents=True, exist_ok=False)
    def save(name, data):
        (out / name).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    def log(message):
        print(message, flush=True)
        with (out / 'progress.log').open('a', encoding='utf-8') as f:
            f.write(message + '\n')
    log('OUTPUT ' + str(out))
    save('manifest.json', dict(protocol_sha256=sha(Path(__file__).with_name('PROTOCOL.md')),
         code_sha256=sha(__file__), python=platform.python_version(), numpy=np.__version__,
         sklearn=sklearn.__version__, lightgbm=lgb.__version__,
         inputs={n: sha(env.DATA / n) for n in ['train_X.csv', 'train_y.csv']},
         source=str(env.SOURCE), arms=ARMS, seeds=SEEDS, episodes=12))
    raw = pd.read_csv(env.DATA / 'train_X.csv')
    raw = raw[raw.row_id.str[:3].isin(['F13', 'F47'])].reset_index(drop=True)
    save('checks.json', checks(raw))
    log('Causality and original fingerprint parity checks PASS')
    y = pd.read_csv(env.DATA / 'train_y.csv', usecols=['row_id', 'sub_ec'])
    lab = features(raw).merge(y, on='row_id', validate='one_to_one').dropna(subset=['sub_ec']).reset_index(drop=True)
    lab['fold'] = -1
    lab['block'] = -1
    for farm, g in lab.groupby('farm'):
        days = sorted(g.day.unique())
        mapping = {d: i // 5 for i, d in enumerate(days)}
        lab.loc[g.index, 'block'] = g.day.map(mapping)
        lab.loc[g.index, 'fold'] = g.day.map(mapping) % 10
    explore_mask = split(lab, 8)[0] & split(lab, 9)[0]
    dev = lab[explore_mask].reset_index(drop=True)
    lab[['row_id', 'farm', 'day', 'hour', 'block', 'fold']].to_csv(out / 'splits.csv', index=False)
    assert not dev.row_id.isin(lab[lab.fold.isin([8, 9])].row_id).any()
    log(f'EC labelled rows={len(lab)}; search rows={len(dev)}; locked confirmation rows={lab.fold.isin([8,9]).sum()}')
    environments = [(fold, seed) for seed in SEEDS for fold in [0, 2, 4, 6]]
    cache = {}
    rewards = {a: [] for a in ARMS}
    history = []
    def evaluate(arm):
        k = len(rewards[arm])
        fold, seed = environments[k]
        if k not in cache:
            trm, vam = split(dev, fold)
            tr, va = dev[trm], dev[vam].reset_index(drop=True)
            log(f'Training search baseline fold={fold} seed={seed}; n={len(tr)}/{len(va)}')
            mem = members(tr, va, seed)
            cache[k] = (tr, va, mem, finish(mem, tr, va))
        tr, va, mem, base = cache[k]
        pred = finish(candidate(arm, tr, va, seed, mem), tr, va)
        a, b = rmse(va.sub_ec, base), rmse(va.sub_ec, pred)
        reward = 1 - b / a
        rewards[arm].append(reward)
        row = dict(episode=len(history) + 1, arm=arm, fold=fold, seed=seed,
                   baseline_rmse=a, candidate_rmse=b, reward=reward)
        history.append(row)
        save('search_history.json', history)
        log(f'SEARCH {len(history):02d} {arm} fold={fold} seed={seed}: {a:.6f} -> {b:.6f}; reward={reward:+.3%}')
    for episode in range(12):
        if episode < len(ARMS):
            arm = ARMS[episode]
        else:
            eligible = [a for a in ARMS if len(rewards[a]) < len(environments)]
            arm = max(eligible, key=lambda a: np.mean(rewards[a]) +
                      .03 * math.sqrt(2 * math.log(episode + 1) / len(rewards[a])))
        evaluate(arm)
    chosen = max(ARMS, key=lambda a: np.mean(rewards[a]))
    save('selection_lock.json', dict(arm=chosen, rewards=rewards,
         note='Locked before completion/confirmation; no runner-up replacement permitted.'))
    log('LOCKED candidate: ' + chosen)
    while len(rewards[chosen]) < len(environments):
        evaluate(chosen)
    search_pass = all(r > 0 for r in rewards[chosen])
    result = dict(chosen=chosen, search_rewards=rewards, search_all_improve=search_pass,
                  adopted=False, reason='Confirmation and full benchmark required')
    confirmation = []
    if np.mean(rewards[chosen]) > 0:
        for fold in [8, 9]:
            # Both confirmation folds are excluded from ALL training.
            _, vam = split(lab, fold)
            tr, va = lab[explore_mask], lab[vam].reset_index(drop=True)
            bases, preds = [], []
            for seed in SEEDS:
                log(f'Training confirmation fold={fold} seed={seed}')
                mem = members(tr, va, seed)
                base = finish(mem, tr, va)
                pred = finish(candidate(chosen, tr, va, seed, mem), tr, va)
                a, b = rmse(va.sub_ec, base), rmse(va.sub_ec, pred)
                confirmation.append(dict(fold=fold, seed=seed, baseline_rmse=a, candidate_rmse=b))
                bases.append(base)
                preds.append(pred)
                log(f'CONFIRM fold={fold} seed={seed}: {a:.6f} -> {b:.6f} ({b/a-1:+.3%})')
            frame = va[['row_id', 'farm', 'day', 'hour', 'block', 'sub_ec']].copy()
            frame['baseline'] = np.mean(bases, axis=0)
            frame['candidate'] = np.mean(preds, axis=0)
            frame.to_csv(out / f'confirmation_{fold}.csv', index=False)
        frame = pd.concat([pd.read_csv(out / f'confirmation_{f}.csv') for f in [8, 9]], ignore_index=True)
        yy, bb, cc = (frame[c].to_numpy() for c in ['sub_ec', 'baseline', 'candidate'])
        a, b = rmse(yy, bb), rmse(yy, cc)
        ci = interval(frame, yy, bb, cc)
        passed = search_pass and all(r['candidate_rmse'] < r['baseline_rmse'] for r in confirmation) and b / a <= .99 and ci[1] < 0
        result.update(confirmation=confirmation, baseline_rmse=a, candidate_rmse=b,
                      relative_change=b/a-1, delta_ci_98_333=ci, screening_pass=passed,
                      reason='Pilot passed; full benchmark still required' if passed else 'Pre-registered evidence threshold failed')
        result['subgroups'] = {}
        for name, mask in [('F13', frame.farm.eq('F13')), ('F47', frame.farm.eq('F47')),
                           ('pass1', frame.day.le(178)), ('pass2', frame.day.ge(179))]:
            if mask.any():
                result['subgroups'][name] = dict(n=int(mask.sum()), baseline=rmse(yy[mask], bb[mask]), candidate=rmse(yy[mask], cc[mask]))
    else:
        result.update(screening_pass=False, reason='Nonpositive mean search reward; confirmation not opened')
    save('result.json', result)
    log('FINISHED: ' + result['reason'])
    log(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    with threadpool_limits(limits=4):
        main()
