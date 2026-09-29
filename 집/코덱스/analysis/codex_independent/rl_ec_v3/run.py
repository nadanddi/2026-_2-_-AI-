"""EC auxiliary-temperature multi-output ExtraTrees experiment."""
import sys
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve()
V2 = HERE.parents[1] / 'rl_ec_v2'
sys.path.insert(0, str(V2))
spec = importlib.util.spec_from_file_location('ec_v2_runner', V2 / 'run.py')
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)

core = previous.core
env = previous.env

import json
import platform
from datetime import datetime

import numpy as np
import pandas as pd
import sklearn
import lightgbm
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from threadpoolctl import threadpool_limits

ROOT = HERE.parents[2]
SPLITS = ROOT / 'local/rl_ec_v1/20260927_173801/splits.csv'
ARMS = {'aux_half': .5, 'aux_equal': 1.0}


def predict_aux(tr, va, seed, ratio):
    assert 'sub_temp' not in core.FULL and 'sub_ec' not in core.FULL
    yec = tr.sub_ec.to_numpy(float)
    yt = tr.sub_temp.to_numpy(float)
    assert np.isfinite(yec).all() and np.isfinite(yt).all()
    scale = ratio * float(np.std(yec)) / float(np.std(yt))
    y = np.column_stack([yec, (yt - np.mean(yt)) * scale])
    imputer = SimpleImputer(strategy='median')
    xt = imputer.fit_transform(tr[core.FULL])
    xv = imputer.transform(va[core.FULL])
    model = ExtraTreesRegressor(n_estimators=600, max_features=1.0,
                                min_samples_leaf=1, n_jobs=4, random_state=seed)
    model.fit(xt, y)
    model.n_jobs = 1
    result = model.predict(xv)[:, 0]
    # Validation temperature is never supplied to the fitted model.
    shuffled = va.copy()
    shuffled['sub_temp'] = np.nan
    assert np.array_equal(result, model.predict(imputer.transform(shuffled[core.FULL]))[:, 0])
    return result


def prepare():
    raw = pd.read_csv(env.DATA / 'train_X.csv')
    raw = raw[raw.row_id.str[:3].isin(['F13', 'F47'])].reset_index(drop=True)
    y = pd.read_csv(env.DATA / 'train_y.csv', usecols=['row_id', 'sub_temp', 'sub_ec'])
    lab = core.features(raw).merge(y, on='row_id', validate='one_to_one')
    lab = lab.dropna(subset=['sub_ec', 'sub_temp']).reset_index(drop=True)
    splits = pd.read_csv(SPLITS)
    assert set(lab.row_id) == set(splits.row_id), 'Auxiliary labels must not change EC population'
    lab = lab.merge(splits[['row_id', 'fold', 'block']], on='row_id', validate='one_to_one')
    allowed = core.split(lab, 8)[0] & core.split(lab, 9)[0]
    assert lab[allowed].fold.isin([8, 9]).sum() == 0
    return lab[allowed].reset_index(drop=True), len(lab) - int(allowed.sum())


def main():
    output = ROOT / 'local/rl_ec_v3' / datetime.now().strftime('%Y%m%d_%H%M%S')
    output.mkdir(parents=True, exist_ok=False)
    def save(name, value):
        (output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    def log(message):
        print(message, flush=True)
        with (output / 'progress.log').open('a', encoding='utf-8') as f:
            f.write(message + '\n')
    save('manifest.json', dict(code_hash=core.sha(__file__), protocol_hash=core.sha(HERE.with_name('PROTOCOL.md')),
         v1_hash=core.sha(HERE.parents[1] / 'rl_ec_v1/run.py'), v2_hash=core.sha(V2 / 'run.py'),
         env_hash=core.sha(HERE.parents[1] / 'rl_ec_v1/env.py'), splits_hash=core.sha(SPLITS),
         inputs={n: core.sha(env.DATA / n) for n in ['train_X.csv', 'train_y.csv']},
         python=platform.python_version(), sklearn=sklearn.__version__, lightgbm=lightgbm.__version__,
         numpy=np.__version__, arms=ARMS, seeds=core.SEEDS, cumulative_candidates=7))
    log('OUTPUT ' + str(output))
    dev, excluded = prepare()
    save('checks.json', dict(ec_population=len(dev)+excluded, search_rows=len(dev),
         locked_or_buffered_rows=excluded, validation_temp_in_features=False,
         validation_temp_perturbation='checked on every fit', confirmation_scored=False,
         test_X_read=False))
    log(f'Search population {len(dev)}; locked confirmation or buffered {excluded}')
    history = []
    def evaluate(fold, seed, arms):
        trm, vam = core.split(dev, fold)
        tr, va = dev[trm], dev[vam].reset_index(drop=True)
        assert not set(va.row_id) & set(tr.row_id)
        log(f'Training fold={fold} seed={seed} candidates={arms}')
        mem = previous.get_baseline(tr, va, seed, log)
        base = core.finish(mem, tr, va)
        frame = va[['row_id', 'farm', 'day', 'hour', 'block', 'sub_ec']].copy()
        frame['baseline'] = base
        for arm in arms:
            et = predict_aux(tr, va, seed, ARMS[arm])
            pred = core.finish([et, mem[1], mem[2]], tr, va)
            a, b = core.rmse(va.sub_ec, base), core.rmse(va.sub_ec, pred)
            row = dict(arm=arm, fold=fold, seed=seed, baseline=a, candidate=b, reward=1-b/a)
            history.append(row)
            save('history.json', history)
            frame[arm] = pred
            log(f'{arm} fold={fold} seed={seed}: {a:.6f} -> {b:.6f} ({b/a-1:+.3%})')
        frame.to_csv(output / f'search_fold{fold}_seed{seed}.csv', index=False)
    for seed in core.SEEDS:
        for fold in [0, 2]:
            evaluate(fold, seed, list(ARMS))
    initial = {arm: [x['reward'] for x in history if x['arm'] == arm] for arm in ARMS}
    eligible = [arm for arm, rewards in initial.items() if min(rewards) > 0 and np.mean(rewards) >= .01]
    save('screen_lock.json', dict(initial_rewards=initial, eligible=eligible))
    log('Eligible for extended search: ' + repr(eligible))
    if eligible:
        for seed in core.SEEDS:
            for fold in [4, 6]:
                evaluate(fold, seed, eligible)
    summary = {}
    for arm in ARMS:
        rewards = [x['reward'] for x in history if x['arm'] == arm]
        summary[arm] = dict(n=len(rewards), mean_relative_rmse_change=-float(np.mean(rewards)),
                            improved=sum(x > 0 for x in rewards),
                            passes_search=len(rewards) == 8 and min(rewards) > 0 and np.mean(rewards) >= .01)
    result = dict(summary=summary, adopted=False, confirmation_scored=False,
                  note='Search only; confirmation untouched; not a significance claim')
    save('result.json', result)
    log('FINISHED ' + json.dumps(result))


if __name__ == '__main__':
    with threadpool_limits(limits=4):
        main()
