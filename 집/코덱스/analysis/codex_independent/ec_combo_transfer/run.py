"""Post-hoc, validation-only daily transfer diagnostics for the EC combo.

Do not run while another CPU/GPU training process uses this workstation.
No test prediction or submission file is created.
"""
import importlib.util
import json
import os
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve()
ROOT = HERE.parents[2]
V1 = HERE.parents[1] / 'rl_ec_v1'
V2 = HERE.parents[1] / 'rl_ec_v2'
sys.path.insert(0, str(V1))
import env  # noqa: E402  — read-only bootstrap for original libraries/data

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['TABPFN_DISABLE_TELEMETRY'] = '1'
_dll = []
gpu = env.SOURCE / '.analysis-tools/extra_gpu'
extra = env.SOURCE / '.analysis-tools/extra'
for directory in (env.SOURCE / '.analysis-tools/msvc', gpu / 'torch/lib'):
    if directory.is_dir() and hasattr(os, 'add_dll_directory'):
        _dll.append(os.add_dll_directory(str(directory)))
for directory in (gpu, extra):
    if directory.is_dir() and str(directory) not in sys.path:
        sys.path.append(str(directory))

spec = importlib.util.spec_from_file_location('ec_combo_transfer_previous', V2 / 'run.py')
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)
core = previous.core
import numpy as np
import pandas as pd
import torch
import tabpfn
from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion
from threadpoolctl import threadpool_limits

SPLITS = ROOT / 'local/rl_ec_v1/20260927_173801/splits.csv'
ORIGINAL_CODE = env.SOURCE / 'research/ec_combo_v1_gpu.py'
ORIGINAL_LOG = env.SOURCE / 'research/local/ec_combo_v1_gpu.log'
WEIGHTS = Path('C:/Users/aozks/AppData/Roaming/tabpfn/tabpfn-v2-regressor.ckpt')
FOLDS = (8, 9)
SERIES = {11: tuple(range(301, 309)), 12: tuple(range(311, 319))}
EXTRA_COLS = tuple(f'{name}_tdmean' for name in core.INDOOR)
CLIP = (0.062, 3.46)


def prepare():
    raw = pd.read_csv(env.DATA / 'train_X.csv')
    raw = raw[raw.row_id.str[:3].isin(('F13', 'F47'))].reset_index(drop=True)
    labels = pd.read_csv(env.DATA / 'train_y.csv', usecols=['row_id', 'sub_ec'])
    lab = core.features(raw).merge(labels, on='row_id', validate='one_to_one')
    lab = lab.dropna(subset=['sub_ec']).merge(
        pd.read_csv(SPLITS)[['row_id', 'fold', 'block']],
        on='row_id', validate='one_to_one')
    lab = lab.sort_values(['farm', 'day', 'hour']).reset_index(drop=True)
    for name in core.INDOOR:
        lab[f'{name}_tdmean'] = lab.groupby(['farm', 'day'])[name].transform(
            lambda s: s.expanding().mean())
    assert lab.row_id.is_unique and len(lab) == 9600
    assert set(lab.fold) == set(range(10))
    assert all(lab.groupby(['farm', 'day']).size().eq(24))
    return raw, lab


def tabpfn_member(x_train, y_train, x_valid, seed):
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(x_train), size=min(2000, len(x_train)), replace=False)
    model = TabPFNRegressor.create_default_for_version(
        ModelVersion.V2, device='cuda', n_estimators=4, random_state=seed,
        ignore_pretraining_limits=True, inference_precision=torch.float32)
    model.fit(x_train[indices], y_train[indices])
    result = np.asarray(model.predict(x_valid), float)
    del model
    torch.cuda.empty_cache()
    assert np.isfinite(result).all()
    return result


def cached_member(out, fold, base_seed, context_seed, view, train, target, valid, ids):
    file = out / f'fold{fold}_base{base_seed}_context{context_seed}_{view}.csv'
    if file.exists():
        saved = pd.read_csv(file)
        assert saved.row_id.tolist() == ids.tolist(), 'Cached validation rows differ'
        result = saved.pred.to_numpy(float)
    else:
        result = tabpfn_member(train, target, valid, context_seed)
        pd.DataFrame({'row_id': ids, 'pred': result}).to_csv(
            file, index=False, float_format='%.17g')
    assert np.isfinite(result).all() and len(result) == len(ids)
    return result


def measure(frame):
    y = frame.sub_ec.to_numpy(float)
    a = float(np.sqrt(np.mean((y - frame.v2.to_numpy(float))**2)))
    b = float(np.sqrt(np.mean((y - frame.combo.to_numpy(float))**2)))
    return {'rows': len(frame), 'days': frame[['farm', 'day']].drop_duplicates().shape[0],
            'v2': a, 'combo': b, 'relative_change': b / a - 1}


def diagnostics(frame):
    test_days = pd.read_csv(env.DATA / 'test_X.csv', usecols=['row_id'])
    test_days['farm'] = test_days.row_id.str[:3]
    test_days['day'] = test_days.row_id.str[4:7].astype(int)
    spans = test_days.groupby('farm').day.agg(['min', 'max']).to_dict('index')
    a = frame.copy()
    a['test_span'] = [spans[r.farm]['min'] <= r.day <= spans[r.farm]['max']
                      for r in a.itertuples()]
    groups = {'all': measure(a)}
    for name, col in [('fold', 'fold'), ('farm', 'farm'),
                      ('test_span', 'test_span'), ('sealed', 'sealed')]:
        groups[name] = {str(k): measure(g) for k, g in a.groupby(col)}
    a['gain'] = (a.sub_ec - a.v2)**2 - (a.sub_ec - a.combo)**2
    day_gain = a.groupby(['farm', 'day']).gain.sum().sort_values(ascending=False)
    positive = day_gain[day_gain > 0]
    top5 = positive.head(5)
    rest = a.set_index(['farm', 'day']).drop(index=top5.index).reset_index()
    groups['day_robustness'] = {
        'improved_days': int((day_gain > 0).sum()),
        'top5_share_positive_gain': float(top5.sum() / positive.sum()),
        'without_top5': measure(rest),
    }
    groups['test_spans'] = {farm: {k: int(v) for k, v in span.items()}
                            for farm, span in spans.items()}
    return groups


def main():
    assert torch.cuda.is_available(), 'This diagnostic requires the GPU; never run concurrently'
    assert core.sha(WEIGHTS) == '2ab5a07d5c41dfe6db9aa7ae106fc6de898326c2765be66505a07e2868c10736'
    if len(sys.argv) == 2:
        out = Path(sys.argv[1]).resolve()
        assert out.is_dir() and ROOT / 'local/ec_combo_transfer' in out.parents
    else:
        out = ROOT / 'local/ec_combo_transfer' / datetime.now().strftime('%Y%m%d_%H%M%S')
        out.mkdir(parents=True, exist_ok=False)
    manifest = {
        'code_hash': core.sha(HERE), 'protocol_hash': core.sha(HERE.with_name('PROTOCOL.md')),
        'original_code_hash': core.sha(ORIGINAL_CODE),
        'original_log_hash': core.sha(ORIGINAL_LOG),
        'v1_hash': core.sha(V1 / 'run.py'), 'v2_hash': core.sha(V2 / 'run.py'),
        'splits_hash': core.sha(SPLITS), 'weights_hash': core.sha(WEIGHTS),
        'train_X_hash': core.sha(env.DATA / 'train_X.csv'),
        'train_y_hash': core.sha(env.DATA / 'train_y.csv'),
        'test_X_hash': core.sha(env.DATA / 'test_X.csv'),
        'python': platform.python_version(), 'numpy': np.__version__,
        'torch': torch.__version__, 'tabpfn': tabpfn.__version__,
        'device': 'cuda', 'folds': list(FOLDS),
        'series': {str(seed): list(contexts) for seed, contexts in SERIES.items()},
        'no_test_predictions': True, 'no_submission_file': True,
    }
    manifest_path = out / 'manifest.json'
    if manifest_path.exists():
        assert json.loads(manifest_path.read_text(encoding='utf-8')) == manifest, (
            'Resume requires the exact same code, protocol, inputs and environment')
    else:
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    start = time.perf_counter()
    raw, lab = prepare()
    raw = core.identify(raw)
    daily = raw.groupby(['farm', 'day']).agg(
        fan=('act_circfan', 'mean'),
        vent0=('act_vent', lambda s: float((s == 0).mean()))).reset_index()
    daily['sealed'] = (daily.fan < 10) & (daily.vent0 > .85)
    for fold in FOLDS:
        trm, vam = core.split(lab, fold)
        tr, va = lab[trm].reset_index(drop=True), lab[vam].reset_index(drop=True)
        assert not set(tr.row_id) & set(va.row_id)
        raw_train = tr[core.FULL].to_numpy(np.float32)
        raw_valid = va[core.FULL].to_numpy(np.float32)
        plus_cols = list(core.FULL) + list(EXTRA_COLS)
        plus_train = tr[plus_cols].to_numpy(np.float32)
        plus_valid = va[plus_cols].to_numpy(np.float32)
        ytrain = tr.sub_ec.to_numpy(float)
        for base_seed, context_seeds in SERIES.items():
            original = previous.get_baseline(tr, va, base_seed, lambda m: print(m, flush=True))
            raw_baseline = .6 * original[0] + .3 * original[1] + .1 * original[2]
            first4, all8 = [], []
            for context_seed in context_seeds:
                augmented = cached_member(out, fold, base_seed, context_seed, 'augmented',
                                          plus_train, ytrain, plus_valid, va.row_id)
                all8.append(augmented)
                if len(first4) < 4:
                    first4.append(cached_member(out, fold, base_seed, context_seed, 'raw',
                                                raw_train, ytrain, raw_valid, va.row_id))
                print(f'fold={fold} base={base_seed} context={context_seed} '
                      f'elapsed={time.perf_counter()-start:.0f}s', flush=True)
            v2 = np.clip(core.shrink(.8 * raw_baseline + .2 * np.mean(first4, axis=0), va), *CLIP)
            combo = np.clip(core.shrink(.6 * raw_baseline + .4 * np.mean(all8, axis=0), va), *CLIP)
            row = va[['row_id', 'farm', 'day', 'hour', 'fold', 'block', 'sub_ec']].copy()
            row['v2'], row['combo'] = v2, combo
            row = row.merge(daily[['farm', 'day', 'sealed']], on=['farm', 'day'], validate='many_to_one')
            path = out / f'fold{fold}_seed{base_seed}.csv'
            row.to_csv(path, index=False, float_format='%.17g')
            print(f'SCORE fold={fold} base={base_seed} {measure(row)}', flush=True)
    result = {
        'status': 'POST_HOC_VALIDATION_ONLY',
        'results': {},
        'elapsed_seconds': time.perf_counter() - start,
        'test_labels_read': False, 'platform_submitted': False,
    }
    for seed in SERIES:
        subset = pd.concat([pd.read_csv(out / f'fold{fold}_seed{seed}.csv')
                            for fold in FOLDS], ignore_index=True)
        result['results'][str(seed)] = diagnostics(subset)
    (out / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    torch.set_num_threads(4)
    with threadpool_limits(limits=4):
        main()
