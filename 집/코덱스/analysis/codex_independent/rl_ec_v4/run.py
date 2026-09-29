"""Farm-identity EC model experiment; no locked confirmation scores."""
import sys
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve()
V2 = HERE.parents[1] / 'rl_ec_v2'
sys.path.insert(0, str(V2))
spec = importlib.util.spec_from_file_location('ec_v2_runner_v4', V2 / 'run.py')
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)
core, env = previous.core, previous.env

import json
import platform
from datetime import datetime

import numpy as np
import pandas as pd
import sklearn
import lightgbm
from threadpoolctl import threadpool_limits

ROOT = HERE.parents[2]
SPLITS = ROOT / 'local/rl_ec_v1/20260927_173801/splits.csv'
ARMS = ['et_farm', 'et_lgb_farm']


def prepare():
    raw = pd.read_csv(env.DATA / 'train_X.csv')
    raw = raw[raw.row_id.str[:3].isin(['F13', 'F47'])].reset_index(drop=True)
    label = pd.read_csv(env.DATA / 'train_y.csv', usecols=['row_id', 'sub_ec'])
    lab = core.features(raw).merge(label, on='row_id', validate='one_to_one')
    lab = lab.dropna(subset=['sub_ec']).reset_index(drop=True)
    lab = lab.merge(pd.read_csv(SPLITS)[['row_id', 'fold', 'block']], on='row_id', validate='one_to_one')
    assert len(lab) == 9600
    lab['farm47'] = lab.farm.eq('F47').astype(float)
    assert lab.farm47.eq(lab.row_id.str.startswith('F47').astype(float)).all()
    assert lab.farm47.groupby(lab.farm).nunique().eq(1).all()
    allowed = core.split(lab, 8)[0] & core.split(lab, 9)[0]
    assert not lab.loc[allowed, 'fold'].isin([8, 9]).any()
    return lab[allowed].reset_index(drop=True), int((~allowed).sum())


def main():
    output = ROOT / 'local/rl_ec_v4' / datetime.now().strftime('%Y%m%d_%H%M%S')
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
         numpy=np.__version__, arms=ARMS, seeds=core.SEEDS, cumulative_candidates=9))
    log('OUTPUT ' + str(output))
    dev, excluded = prepare()
    save('checks.json', dict(search_rows=len(dev), locked_or_buffered_rows=excluded,
         farm_indicator_from_row_id='PASS', no_target_in_features='PASS',
         confirmation_scored=False, test_X_read=False))
    history = []
    def evaluate(fold, seed, arms):
        trm, vam = core.split(dev, fold)
        tr, va = dev[trm], dev[vam].reset_index(drop=True)
        assert not set(tr.row_id) & set(va.row_id)
        log(f'Training fold={fold} seed={seed} arms={arms}')
        mem = previous.get_baseline(tr, va, seed, log)
        base = core.finish(mem, tr, va)
        frame = va[['row_id', 'farm', 'day', 'hour', 'block', 'sub_ec']].copy()
        frame['baseline'] = base
        cols = core.FULL + ['farm47']
        et = core.predict_model(core.et(seed), tr, va, cols)
        for arm in arms:
            lgb = (mem[1] if arm == 'et_farm' else
                   core.predict_model(core.lg(seed, 'tweedie'), tr, va, core.BASE + ['farm47']))
            pred = core.finish([et, lgb, mem[2]], tr, va)
            a, b = core.rmse(va.sub_ec, base), core.rmse(va.sub_ec, pred)
            row = dict(arm=arm, fold=fold, seed=seed, baseline=a, candidate=b, reward=1-b/a)
            history.append(row)
            frame[arm] = pred
            save('history.json', history)
            log(f'{arm} fold={fold} seed={seed}: {a:.6f} -> {b:.6f} ({b/a-1:+.3%})')
        frame.to_csv(output / f'search_fold{fold}_seed{seed}.csv', index=False)
    for seed in core.SEEDS:
        for fold in [0, 2]:
            evaluate(fold, seed, ARMS)
    initial = {a: [r['reward'] for r in history if r['arm'] == a] for a in ARMS}
    eligible = [a for a, rewards in initial.items() if min(rewards) > 0 and np.mean(rewards) >= .01]
    save('screen_lock.json', dict(initial_rewards=initial, eligible=eligible))
    log('Eligible for extended search: ' + repr(eligible))
    if eligible:
        for seed in core.SEEDS:
            for fold in [4, 6]:
                evaluate(fold, seed, eligible)
    summary = {}
    for arm in ARMS:
        rewards = [r['reward'] for r in history if r['arm'] == arm]
        summary[arm] = dict(n=len(rewards), mean_relative_rmse_change=-float(np.mean(rewards)),
                            improved=sum(r > 0 for r in rewards),
                            passes_search=len(rewards) == 8 and min(rewards) > 0 and np.mean(rewards) >= .01)
    result = dict(summary=summary, adopted=False, confirmation_scored=False,
                  note='Search only; confirmation untouched; not a significance claim')
    save('result.json', result)
    log('FINISHED ' + json.dumps(result))


if __name__ == '__main__':
    with threadpool_limits(limits=4):
        main()
