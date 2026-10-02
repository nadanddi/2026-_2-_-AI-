"""Fixed K2 E40_8 comparison and K3 raw context outputs; no submission files."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / '집' / '클로드' / 'research'))
import env
import env_extra

import argparse
from contextlib import contextmanager
import csv
import gc
import hashlib
import importlib.util
import json
import math
import os
import platform
import tempfile
import time

for name in ['HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'TABPFN_DISABLE_TELEMETRY',
             'HF_HUB_DISABLE_TELEMETRY']:
    os.environ[name] = '1'
sys.dont_write_bytecode = True

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
OUT = ROOT / '집' / '코덱스' / 'local' / 'ec_stage2_tabpfn_20261002_v1'
COMMON_PATH = HERE.parent / 'ec_model_common_20261002_v1' / 'common.py'
CKPT = Path.home() / 'AppData' / 'Roaming' / 'tabpfn' / 'tabpfn-v2-regressor.ckpt'
R3_SEEDS = [401, 402, 403]
PFN_SEEDS = list(range(5, 13))
REFERENCE_SEEDS = [7, 101, 2024]
VALIDATORS = ['DIAG10', 'A', 'B', 'EXT10', 'EXT12']
ALPHA = .025 / 2
BOOT_SEED = 918
NBOOT = 20000
N_SHARDS = 3


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def frame_hash(frame, columns):
    return hashlib.sha256(pd.util.hash_pandas_object(frame[columns], index=False).values.tobytes()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    assert not path.exists(), f'Existing artifact is preserved: {path}'
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', suffix='.json.tmp',
                                     prefix=path.name + '.', dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_npz(path, arrays):
    path = Path(path)
    assert not path.exists(), f'Existing artifact is preserved: {path}'
    with tempfile.NamedTemporaryFile(mode='wb', suffix='.npz.tmp', prefix=path.name + '.',
                                     dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_csv(path, frame):
    path = Path(path)
    assert not path.exists(), f'Existing artifact is preserved: {path}'
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8-sig', newline='',
                                     suffix='.csv.tmp', prefix=path.name + '.',
                                     dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        frame.to_csv(stream, index=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def log(message, log_path):
    print(message, flush=True)
    with log_path.open('a', encoding='utf-8') as stream:
        stream.write(message + '\n')


def load_common():
    spec = importlib.util.spec_from_file_location('stage2_fixed_common', COMMON_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def environment_manifest():
    import sklearn
    import lightgbm
    import torch
    import tabpfn
    assert CKPT.is_file(), f'Local V2 reg checkpoint unavailable: {CKPT}'
    return {
        'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__,
        'sklearn': sklearn.__version__, 'lightgbm': lightgbm.__version__,
        'torch': torch.__version__, 'tabpfn': tabpfn.__version__,
        'checkpoint_path': str(CKPT), 'checkpoint_sha256': sha(CKPT),
        'device': 'cpu', 'inference_precision': 'float32', 'n_estimators': 4,
        'context_size': 2000, 'pfn_context_seeds': PFN_SEEDS, 'r3_seeds': R3_SEEDS,
        'threads': 2, 'offline_telemetry_disabled': True,
    }


def base_provenance(common, environment):
    return {'script': sha(__file__), 'protocol': sha(HERE / 'PROTOCOL.md'),
            'environment': environment, 'shared': common.manifest()}


def fold_provenance(tr, va, core, base):
    assert tr.row_id.is_unique and va.row_id.is_unique
    assert not set(tr.row_id) & set(va.row_id)
    assert len(core.FULL) == 38 and len(core.BASE) == 14
    return base | {'train_hash': frame_hash(tr, ['row_id', 'sub_ec'] + core.FULL),
                   'validation_hash': frame_hash(va, ['row_id', 'sub_ec'] + core.FULL),
                   'train_rows': len(tr), 'validation_rows': len(va),
                   'train_days': len(tr[['farm', 'day']].drop_duplicates()),
                   'validation_days': len(va[['farm', 'day']].drop_duplicates()),
                   'train_target_bounds': [float(tr.sub_ec.min()), float(tr.sub_ec.max())]}


def cache_path(fold, kind, seed):
    name, index, _ = fold
    return OUT / f'{name}_{index}_{kind}_{seed}.npz'


def load_cache(path, key, expected_ids):
    manifest_path = path.with_suffix('.json')
    if not path.exists() and not manifest_path.exists():
        return None
    assert path.exists() and manifest_path.exists(), f'Incomplete cache preserved: {path}'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    assert manifest['key'] == key, f'Cache provenance mismatch: {path}'
    assert manifest['prediction_sha256'] == sha(path), f'Cache file hash mismatch: {path}'
    with np.load(path, allow_pickle=False) as stored:
        arrays = {name: stored[name].copy() for name in stored.files}
    assert arrays['row_id'].tolist() == expected_ids.tolist()
    for name, value in arrays.items():
        if value.dtype.kind in 'fc':
            assert np.isfinite(value).all(), f'Non-finite cache value: {path} / {name}'
    return arrays, manifest


def write_cache(path, key, arrays, metadata):
    atomic_npz(path, arrays)
    manifest = metadata | {'key': key, 'prediction_sha256': sha(path)}
    atomic_json(path.with_suffix('.json'), manifest)
    return arrays, manifest


@contextmanager
def fold_lock(fold):
    name, index, _ = fold
    path = OUT / f'{name}_{index}.lock'
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    except FileExistsError as error:
        raise RuntimeError(f'폴드 중복 실행/비정상 종료 lock 보존: {path}. '
                           '담당자가 프로세스 종료를 확인한 뒤 이 lock 한 파일만 제거해야 합니다.') from error
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as stream:
            json.dump({'pid': os.getpid(), 'fold': [name, index], 'script': sha(__file__)}, stream)
        yield
    finally:
        # Only unlink our own single lock. No recursive file operation is used.
        if path.exists():
            owner = json.loads(path.read_text(encoding='utf-8'))
            if owner['pid'] == os.getpid():
                path.unlink()


def rmse(y, prediction):
    return math.sqrt(math.fsum((float(a) - float(b)) ** 2 for a, b in zip(y, prediction)) / len(y))


def finish(raw, tr, va, core):
    return np.clip(core.shrink(np.asarray(raw, dtype=float), va), tr.sub_ec.min(), tr.sub_ec.max())


def fit_r3(tr, va, core, fold, seed, provenance, log_path):
    path = cache_path(fold, 'r3', seed)
    key = canonical_hash(provenance | {'kind': 'R3 original .6ET+.3LGB+.1MLP', 'seed': seed})
    cached = load_cache(path, key, va.row_id)
    if cached is not None:
        log(f'{fold[0]}/{fold[1]} R3{seed}: verified cache', log_path)
        return cached
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.neural_network import MLPRegressor
    started = time.monotonic()
    y = tr.sub_ec.to_numpy(dtype=float)
    prediction, train_prediction = [], []
    et = core.et(seed)
    et.steps[-1][1].set_params(n_jobs=2)
    lg = core.lg(seed, 'tweedie').set_params(n_jobs=2)
    mlp = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(),
                        MLPRegressor(hidden_layer_sizes=(128, 64), alpha=1e-2,
                                     learning_rate_init=1e-3, max_iter=800,
                                     early_stopping=True, n_iter_no_change=25,
                                     validation_fraction=.12, random_state=seed))
    for model, columns in [(et, core.FULL), (lg, core.BASE), (mlp, core.BASE)]:
        with threadpool_limits(limits=2):
            model.fit(tr[columns], y)
            prediction.append(np.asarray(model.predict(va[columns]), dtype=float))
            train_prediction.append(np.asarray(model.predict(tr[columns]), dtype=float))
        del model
    raw = .6 * prediction[0] + .3 * prediction[1] + .1 * prediction[2]
    train_raw = .6 * train_prediction[0] + .3 * train_prediction[1] + .1 * train_prediction[2]
    arrays = {'row_id': va.row_id.to_numpy(dtype=str), 'sub_ec': va.sub_ec.to_numpy(dtype=float),
              'raw_r3': raw, 'raw_et': prediction[0], 'raw_lgb': prediction[1],
              'raw_mlp': prediction[2], 'train_row_id': tr.row_id.to_numpy(dtype=str),
              'train_raw_r3': train_raw}
    metadata = {'provenance': provenance, 'kind': 'r3', 'seed': seed,
                'fold': [fold[0], fold[1]], 'elapsed_seconds': time.monotonic() - started,
                'train_raw_rmse': rmse(y, train_raw),
                'train_post_rmse': rmse(y, finish(train_raw, tr, tr, core)),
                'raw_validation_rmse': rmse(va.sub_ec, raw),
                'post_validation_rmse': rmse(va.sub_ec, finish(raw, tr, va, core))}
    stored = write_cache(path, key, arrays, metadata)
    log(f'{fold[0]}/{fold[1]} R3{seed}: saved {metadata["elapsed_seconds"]:.1f}s', log_path)
    gc.collect()
    return stored


def fit_pfn(tr, va, core, fold, seed, provenance, log_path):
    path = cache_path(fold, 'pfn', seed)
    key = canonical_hash(provenance | {'kind': 'TabPFN V2 CPU float32 context2000 estimator4', 'seed': seed})
    cached = load_cache(path, key, va.row_id)
    if cached is not None:
        log(f'{fold[0]}/{fold[1]} PFN{seed}: verified cache', log_path)
        return cached
    import torch
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion
    torch.set_num_threads(2)
    started = time.monotonic()
    index = np.random.default_rng(seed).choice(len(tr), size=min(2000, len(tr)), replace=False)
    model = TabPFNRegressor.create_default_for_version(
        ModelVersion.V2, model_path=str(CKPT), device='cpu', n_estimators=4,
        random_state=seed, ignore_pretraining_limits=True,
        inference_precision=torch.float32, n_preprocessing_jobs=1,
    )
    xtr = tr[core.FULL].to_numpy(dtype=np.float32)
    xva = va[core.FULL].to_numpy(dtype=np.float32)
    with threadpool_limits(limits=2):
        model.fit(xtr[index], tr.sub_ec.to_numpy(dtype=float)[index])
        raw = np.asarray(model.predict(xva), dtype=float)
        repeated = np.asarray(model.predict(xva[:8]), dtype=float)
    assert np.array_equal(raw[:len(repeated)], repeated), 'PFN first8 repeat prediction mismatch'
    assert raw.shape == (len(va),) and np.isfinite(raw).all()
    arrays = {'row_id': va.row_id.to_numpy(dtype=str), 'sub_ec': va.sub_ec.to_numpy(dtype=float),
              'raw_pfn': raw, 'context_index': index,
              'context_row_id': tr.row_id.iloc[index].to_numpy(dtype=str)}
    metadata = {'provenance': provenance, 'kind': 'pfn', 'seed': seed,
                'fold': [fold[0], fold[1]], 'elapsed_seconds': time.monotonic() - started,
                'raw_validation_rmse': rmse(va.sub_ec, raw), 'repeat_first8': 'PASS',
                'sample_row_ids_hash': canonical_hash(arrays['context_row_id'].tolist())}
    stored = write_cache(path, key, arrays, metadata)
    log(f'{fold[0]}/{fold[1]} PFN{seed}: saved {metadata["elapsed_seconds"]:.1f}s', log_path)
    del model, xtr, xva
    gc.collect()
    return stored


def verify_fold(tr, va, core, fold, provenance):
    r3, pfn = {}, {}
    for kind, seeds, description, target in [
        ('r3', R3_SEEDS, 'R3 original .6ET+.3LGB+.1MLP', r3),
        ('pfn', PFN_SEEDS, 'TabPFN V2 CPU float32 context2000 estimator4', pfn),
    ]:
        for seed in seeds:
            key = canonical_hash(provenance | {'kind': description, 'seed': seed})
            cached = load_cache(cache_path(fold, kind, seed), key, va.row_id)
            assert cached is not None, f'22-fold collection incomplete: {fold[0]}/{fold[1]} {kind}{seed}'
            arrays, manifest = cached
            assert np.array_equal(arrays['sub_ec'], va.sub_ec.to_numpy(dtype=float))
            if kind == 'r3':
                assert arrays['train_row_id'].tolist() == tr.row_id.tolist()
                assert np.array_equal(arrays['raw_r3'], .6 * arrays['raw_et'] + .3 * arrays['raw_lgb'] + .1 * arrays['raw_mlp'])
            else:
                expected = np.random.default_rng(seed).choice(len(tr), size=min(2000, len(tr)), replace=False)
                assert np.array_equal(arrays['context_index'], expected)
                assert arrays['context_row_id'].tolist() == tr.row_id.iloc[expected].tolist()
            target[seed] = arrays
    return r3, pfn


def fixed_baseline(common, va, fold):
    name, index, _ = fold
    reference = [common.baseline(va, name, index, seed) for seed in REFERENCE_SEEDS]
    assert len({item['cache_hash'] for item in reference}) == 1
    path = common.CACHE / f'{name}_{index}.npz'
    meta = json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
    assert meta['prediction_sha256'] == reference[0]['cache_hash']
    with np.load(path, allow_pickle=False) as stored:
        assert stored['row_id'].tolist() == va.row_id.tolist()
    return np.mean([item['v2'] for item in reference], axis=0), reference[0]['cache_hash']


def bootstrap(group, reference):
    units = group.groupby(['farm', 'block'], sort=True)
    sizes = units.size()
    assert len(sizes) == 80 and sizes.sum() == 8640
    n = sizes.to_numpy(dtype=float)
    base = units.apply(lambda g: float(((g[reference] - g.sub_ec) ** 2).sum()), include_groups=False).to_numpy()
    candidate = units.apply(lambda g: float(((g.E40_8 - g.sub_ec) ** 2).sum()), include_groups=False).to_numpy()
    generator = np.random.default_rng(BOOT_SEED)
    draws = []
    for farm in sorted(set(sizes.index.get_level_values('farm'))):
        choices = np.flatnonzero(sizes.index.get_level_values('farm') == farm)
        draws.append(choices[generator.integers(0, len(choices), size=(NBOOT, len(choices)))])
    indices = np.concatenate(draws, axis=1)
    denominator = n[indices].sum(axis=1)
    delta_mse = (candidate[indices] - base[indices]).sum(axis=1) / denominator
    delta_rmse = np.sqrt(candidate[indices].sum(axis=1) / denominator) - np.sqrt(base[indices].sum(axis=1) / denominator)
    interval = np.quantile(delta_mse, [ALPHA, 1 - ALPHA]).tolist()
    p_worse = float(np.mean(delta_mse >= 0))
    return {'seed': int(group.seed.iloc[0]), 'reference': reference, 'arm': 'E40_8',
            'n_blocks': 80, 'n_rows': len(group), 'iterations': NBOOT, 'bootstrap_seed': BOOT_SEED,
            'alpha': ALPHA, 'ci_mse': interval,
            'ci_rmse': np.quantile(delta_rmse, [ALPHA, 1 - ALPHA]).tolist(),
            'p_worse': p_worse, 'passed': bool(p_worse < ALPHA and interval[1] < 0)}


def evaluate(frame):
    expected = {(v, seed) for v in VALIDATORS for seed in R3_SEEDS}
    assert set(frame[['validator', 'seed']].itertuples(index=False, name=None)) == expected
    assert not frame.duplicated(['validator', 'validation_fold', 'seed', 'row_id']).any()
    assert np.isfinite(frame[['sub_ec', 'E40_8', 'v2_fresh', 'v2_fixed']].to_numpy(dtype=float)).all()
    scores, boots = [], []
    for (name, seed), group in frame.groupby(['validator', 'seed'], sort=True):
        unique = group.groupby('row_id')[['sub_ec', 'E40_8', 'v2_fresh', 'v2_fixed']].mean()
        for reference in ['v2_fresh', 'v2_fixed']:
            value = rmse(group.sub_ec, group.E40_8)
            base = rmse(group.sub_ec, group[reference])
            fold_values = [rmse(g.sub_ec, g.E40_8) for _, g in group.groupby('validation_fold')]
            base_fold = [rmse(g.sub_ec, g[reference]) for _, g in group.groupby('validation_fold')]
            scores.append({'validator': name, 'seed': int(seed), 'reference': reference,
                           'candidate_rmse': value, 'reference_rmse': base, 'delta_rmse': value - base,
                           'delta_pct': 100 * (value / base - 1), 'improved': value < base,
                           'candidate_unique_rmse': rmse(unique.sub_ec, unique.E40_8),
                           'reference_unique_rmse': rmse(unique.sub_ec, unique[reference]),
                           'row_occurrences': len(group), 'unique_rows': len(unique),
                           'fold_rmse_mean': float(np.mean(fold_values)),
                           'fold_rmse_std': float(np.std(fold_values, ddof=1)) if len(fold_values) > 1 else None,
                           'reference_fold_rmse_mean': float(np.mean(base_fold)),
                           'reference_fold_rmse_std': float(np.std(base_fold, ddof=1)) if len(base_fold) > 1 else None})
            if name == 'DIAG10':
                assert len(group) == 8640 and group.row_id.is_unique
                boots.append(bootstrap(group, reference))
    assert len(scores) == 30 and len(boots) == 6
    direction = all(row['improved'] for row in scores)
    confidence = all(row['passed'] for row in boots)
    averaged = frame.groupby(['validator', 'validation_fold', 'row_id'], as_index=False)[
        ['sub_ec', 'E40_8', 'v2_fresh', 'v2_fixed']].mean()
    ensemble = []
    for name, group in averaged.groupby('validator', sort=True):
        ensemble.append({'validator': name, **{model: rmse(group.sub_ec, group[model])
                                             for model in ['E40_8', 'v2_fresh', 'v2_fixed']}})
    return {'scores': scores, 'bootstrap': boots, 'ensemble_diagnostic_only': ensemble,
            'direction_pass': direction, 'confidence_pass': confidence,
            'improving_seed_validator_reference_cells': sum(row['improved'] for row in scores),
            'total_cells': 30, 'public_selection_gate_pass': direction and confidence,
            'candidate_adopted': False, 'independent_verification_pending': True,
            'K3_is_data_output_only': True, 'final_lock_scored': False, 'submission_created': False}


def collect(common, data, lab, folds, locks, core, base):
    completion_path = HERE / 'completion.json'
    if completion_path.exists():
        completed = json.loads(completion_path.read_text(encoding='utf-8'))
        assert completed['base_provenance_hash'] == canonical_hash(base)
        for filename, digest in completed['artifacts'].items():
            assert sha(Path(filename)) == digest
        print('기존 collect 완료 산출물 해시 확인; 변경하지 않았습니다.', flush=True)
        return
    wide_records, long_records, baseline_hashes = [], [], {}
    for fold in folds:
        tr, va = common.split_fold(data, lab, fold, locks)
        provenance = fold_provenance(tr, va, core, base)
        r3, pfn = verify_fold(tr, va, core, fold, provenance)
        fixed, reference_hash = fixed_baseline(common, va, fold)
        baseline_hashes[f'{fold[0]}:{fold[1]}'] = reference_hash
        wide = va[['row_id', 'farm', 'day', 'hour', 'block', 'sub_ec']].copy()
        wide['validator'], wide['validation_fold'] = fold[:2]
        for seed in PFN_SEEDS:
            wide[f'raw_pfn_{seed}'] = pfn[seed]['raw_pfn']
        bag4 = np.mean([pfn[seed]['raw_pfn'] for seed in PFN_SEEDS[:4]], axis=0)
        bag8 = np.mean([pfn[seed]['raw_pfn'] for seed in PFN_SEEDS], axis=0)
        wide['raw_pfn_bag4'], wide['raw_pfn_bag8'] = bag4, bag8
        wide['v2_fixed'] = fixed
        for seed in R3_SEEDS:
            raw = r3[seed]['raw_r3']
            fresh = finish(.8 * raw + .2 * bag4, tr, va, core)
            candidate = finish(.6 * raw + .4 * bag8, tr, va, core)
            wide[f'raw_r3_{seed}'] = raw
            wide[f'v2_fresh_{seed}'] = fresh
            wide[f'E40_8_{seed}'] = candidate
            long = va[['row_id', 'farm', 'day', 'hour', 'block', 'sub_ec']].copy()
            long['validator'], long['validation_fold'], long['seed'] = fold[0], fold[1], seed
            long['raw_r3'], long['raw_pfn_bag4'], long['raw_pfn_bag8'] = raw, bag4, bag8
            long['raw_E40_8'] = .6 * raw + .4 * bag8
            long['v2_fixed'], long['v2_fresh'], long['E40_8'] = fixed, fresh, candidate
            long_records.append(long)
        wide_records.append(wide)
    wide = pd.concat(wide_records, ignore_index=True)
    long = pd.concat(long_records, ignore_index=True)
    assert len(wide) == 27720 and len(long) == 27720 * 3
    assert not wide.duplicated(['validator', 'validation_fold', 'row_id']).any()
    expected_folds = {(f[0], f[1]) for f in folds}
    assert set(wide[['validator', 'validation_fold']].itertuples(index=False, name=None)) == expected_folds
    result = evaluate(long)
    artifacts = {}
    for path, frame in [(OUT / 'oof_wide.csv', wide), (OUT / 'oof_long.csv', long),
                        (HERE / 'summary_scores.csv', pd.DataFrame(result['scores'])),
                        (HERE / 'bootstrap_DIAG10.csv', pd.DataFrame(result['bootstrap']))]:
        atomic_csv(path, frame)
        artifacts[str(path)] = sha(path)
    arrays = {column: wide[column].to_numpy(dtype=str if wide[column].dtype.kind in 'OU' else None)
              for column in wide.columns}
    atomic_npz(OUT / 'oof_wide.npz', arrays)
    artifacts[str(OUT / 'oof_wide.npz')] = sha(OUT / 'oof_wide.npz')
    atomic_json(HERE / 'result.json', result)
    artifacts[str(HERE / 'result.json')] = sha(HERE / 'result.json')
    atomic_json(OUT / 'collection_manifest.json', {'base_provenance': base, 'baseline_hashes': baseline_hashes,
                                                 'pfn_context_models': 176, 'r3_models': 66,
                                                 'wide_rows': len(wide), 'long_rows': len(long)})
    artifacts[str(OUT / 'collection_manifest.json')] = sha(OUT / 'collection_manifest.json')
    atomic_json(completion_path, {'status': 'COMPLETE', 'pfn_context_models': 176, 'r3_models': 66,
                                  'wide_rows': len(wide), 'long_rows': len(long),
                                  'base_provenance_hash': canonical_hash(base), 'artifacts': artifacts,
                                  'public_selection_gate_pass': result['public_selection_gate_pass'],
                                  'candidate_adopted': False, 'independent_verification_pending': True,
                                  'final_lock_scored': False, 'submission_created': False})
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--run', action='store_true')
    mode.add_argument('--collect', action='store_true')
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument('--shard', type=int, choices=range(N_SHARDS))
    selection.add_argument('--fold', help='Recovery only: fixed NAME:INDEX')
    args = parser.parse_args()
    assert not args.collect or args.shard is None and args.fold is None
    assert not args.run or args.shard is not None or args.fold is not None, 'Choose a non-overlapping fixed shard'
    OUT.mkdir(parents=True, exist_ok=True)
    common = load_common()
    data, lab, folds, locks, core = common.prepare()
    assert len(folds) == 22 and len(lab) == 8640
    environment = environment_manifest()
    base = base_provenance(common, environment)
    if args.collect:
        collect(common, data, lab, folds, locks, core, base)
        return
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    selected = [fold for ordinal, fold in enumerate(folds) if ordinal % N_SHARDS == args.shard]
    tag = f'shard{args.shard}'
    if args.fold is not None:
        name, index = args.fold.split(':')
        selected = [fold for fold in folds if fold[:2] == (name, int(index))]
        assert len(selected) == 1, 'Unknown fixed fold'
        tag = f'{name}_{index}'
    checks = common.feature_checks(data)
    assert all(checks[name] for name in ['order_invariance', 'future_input_invariance',
                                       'other_farm_input_invariance', 'past_only_invariance',
                                       'missing_midnight_preserved'])
    env_path = OUT / f'{tag}_environment.json'
    env_record = {'base_provenance': base, 'feature_checks': checks,
                  'fixed_fold_group': [[fold[0], fold[1]] for fold in selected]}
    if env_path.exists():
        assert json.loads(env_path.read_text(encoding='utf-8')) == env_record
    else:
        atomic_json(env_path, env_record)
    log_path = OUT / f'{tag}_progress.log'
    for fold in selected:
        tr, va = common.split_fold(data, lab, fold, locks)
        assert not set(tr[['farm', 'day']].itertuples(index=False, name=None)) & locks
        assert not set(va[['farm', 'day']].itertuples(index=False, name=None)) & locks
        provenance = fold_provenance(tr, va, core, base)
        with fold_lock(fold):
            for seed in R3_SEEDS:
                fit_r3(tr, va, core, fold, seed, provenance, log_path)
            for seed in PFN_SEEDS:
                fit_pfn(tr, va, core, fold, seed, provenance, log_path)
            verify_fold(tr, va, core, fold, provenance)
        log(f'{fold[0]}/{fold[1]}: all3 R3/all8 PFN caches verified', log_path)
    completed_path = OUT / f'{tag}_completion.json'
    record = {'status': 'COMPLETE', 'base_provenance_hash': canonical_hash(base),
              'folds': [[fold[0], fold[1]] for fold in selected],
              'final_lock_scored': False, 'submission_created': False}
    if completed_path.exists():
        assert json.loads(completed_path.read_text(encoding='utf-8')) == record
    else:
        atomic_json(completed_path, record)


if __name__ == '__main__':
    main()
