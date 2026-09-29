"""Offline CPU-only probe of the previously selected TabPFN v2 EC member."""
import os
import sys
import importlib.util
import json
import platform
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve()
V2 = HERE.parents[1] / 'rl_ec_v2'
sys.path.insert(0, str(V2))
spec = importlib.util.spec_from_file_location('ec_v2_runner_cpu_probe', V2 / 'run.py')
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)
core, env = previous.core, previous.env

# Set before importing the model library: no remote model lookup during inference.
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['TABPFN_DISABLE_TELEMETRY'] = '1'
extra = env.SOURCE / '.analysis-tools' / 'extra'
_dll = []
for path in [env.SOURCE / '.analysis-tools/msvc', extra / 'torch/lib']:
    if path.is_dir() and hasattr(os, 'add_dll_directory'):
        _dll.append(os.add_dll_directory(str(path)))
sys.path.append(str(extra))

import numpy as np
import pandas as pd
import torch
import tabpfn
from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion
from threadpoolctl import threadpool_limits

ROOT = HERE.parents[2]
SPLITS = ROOT / 'local/rl_ec_v1/20260927_173801/splits.csv'
FOLD, BASE_SEED, CONTEXT_SEED, CONTEXT_SIZE = 0, 7, 1, 2000


def main():
    output = ROOT / 'local/tabpfn_cpu_probe' / datetime.now().strftime('%Y%m%d_%H%M%S')
    output.mkdir(parents=True, exist_ok=False)
    def save(name, obj):
        (output / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
    def log(message):
        print(message, flush=True)
        with (output / 'progress.log').open('a', encoding='utf-8') as f:
            f.write(message + '\n')
    save('manifest.json', dict(code_hash=core.sha(__file__), protocol_hash=core.sha(HERE.with_name('PROTOCOL.md')),
         v1_hash=core.sha(HERE.parents[1] / 'rl_ec_v1/run.py'),
         v2_hash=core.sha(V2 / 'run.py'), splits_hash=core.sha(SPLITS),
         inputs={n: core.sha(env.DATA/n) for n in ['train_X.csv', 'train_y.csv']},
         python=platform.python_version(), numpy=np.__version__, torch=torch.__version__,
         tabpfn=tabpfn.__version__, device='cpu', precision='float32',
         base_seed=BASE_SEED, context_seed=CONTEXT_SEED,
         context_size=CONTEXT_SIZE, n_estimators=4, offline=True))
    log('OUTPUT ' + str(output))
    start = time.perf_counter()
    raw = pd.read_csv(env.DATA / 'train_X.csv')
    raw = raw[raw.row_id.str[:3].isin(['F13', 'F47'])].reset_index(drop=True)
    y = pd.read_csv(env.DATA / 'train_y.csv', usecols=['row_id', 'sub_ec'])
    lab = core.features(raw).merge(y, on='row_id', validate='one_to_one').dropna(subset=['sub_ec'])
    lab = lab.merge(pd.read_csv(SPLITS)[['row_id', 'fold', 'block']], on='row_id', validate='one_to_one')
    allowed = core.split(lab, 8)[0] & core.split(lab, 9)[0]
    dev = lab[allowed].reset_index(drop=True)
    trm, vam = core.split(dev, FOLD)
    tr, va = dev[trm], dev[vam].reset_index(drop=True)
    assert not set(tr.row_id) & set(va.row_id)
    log(f'FEATURES READY {time.perf_counter()-start:.1f}s train={len(tr)} validation={len(va)}')
    baseline = previous.get_baseline(tr, va, BASE_SEED, log)
    p_base = core.finish(baseline, tr, va)
    features = core.FULL
    Xtr = tr[features].to_numpy(dtype=np.float32)
    Xva = va[features].to_numpy(dtype=np.float32)
    yt = tr.sub_ec.to_numpy(float)
    rng = np.random.default_rng(CONTEXT_SEED)
    idx = rng.choice(len(tr), size=CONTEXT_SIZE, replace=False)
    model = TabPFNRegressor.create_default_for_version(
        ModelVersion.V2, device='cpu', n_estimators=4,
        random_state=CONTEXT_SEED, ignore_pretraining_limits=True,
        inference_precision=torch.float32)
    log(f'MODEL LOADED {time.perf_counter()-start:.1f}s')
    model.fit(Xtr[idx], yt[idx])
    log(f'FIT DONE {time.perf_counter()-start:.1f}s')
    member = np.asarray(model.predict(Xva), dtype=float)
    log(f'PREDICT DONE {time.perf_counter()-start:.1f}s')
    again = np.asarray(model.predict(Xva[:min(8, len(Xva))]), dtype=float)
    repeat_max = float(np.max(np.abs(member[:len(again)] - again)))
    raw_base = .6*baseline[0] + .3*baseline[1] + .1*baseline[2]
    p_blend = np.clip(core.shrink(.8*raw_base + .2*member, va), tr.sub_ec.min(), tr.sub_ec.max())
    score = dict(n_train=len(tr), n_validation=len(va), n_context=len(idx),
                 baseline_rmse=core.rmse(va.sub_ec, p_base),
                 member_rmse=core.rmse(va.sub_ec, member),
                 blend_rmse=core.rmse(va.sub_ec, p_blend),
                 relative_change=core.rmse(va.sub_ec, p_blend)/core.rmse(va.sub_ec, p_base)-1,
                 repeat_first8_max_difference=repeat_max,
                 elapsed_seconds=time.perf_counter()-start, confirmation_scored=False)
    save('result.json', score)
    frame = va[['row_id', 'farm', 'day', 'hour', 'block', 'sub_ec']].copy()
    frame['baseline'] = p_base
    frame['member'] = member
    frame['blend'] = p_blend
    frame.to_csv(output / 'search_fold0_seed7_context1.csv', index=False)
    log('FINISHED ' + json.dumps(score))


if __name__ == '__main__':
    torch.set_num_threads(4)
    with threadpool_limits(limits=4):
        main()
