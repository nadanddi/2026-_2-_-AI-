"""Pre-registered, single-arm CatBoost EC comparison; no submission creation."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / '집' / '클로드' / 'research'))
import env  # first project import, read-only common environment bootstrap
sys.path.insert(0, str(ROOT / '집' / '코덱스' / 'local' / 'ec_model_packages_20261002_v1'))

import argparse
import hashlib
import importlib.util
import json
import math
import platform
import time

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
COMMON_PATH = HERE.parent / 'ec_model_common_20261002_v1' / 'common.py'
OUT = ROOT / '집' / '코덱스' / 'local' / 'ec_catboost_20261002_v1'
SEEDS = [7, 101, 2024]
VALIDATORS = ['DIAG10', 'A', 'B', 'EXT10', 'EXT12']
BOOTSTRAP_DRAWS = 20000
BOOTSTRAP_SEED = 918
BONFERRONI_ARMS = 5
PARAMETERS = dict(
    loss_function='RMSE', eval_metric='RMSE', iterations=1200,
    depth=6, learning_rate=0.03, l2_leaf_reg=3,
    bootstrap_type='Bernoulli', subsample=0.8, rsm=0.8,
    nan_mode='Min', task_type='CPU', thread_count=2,
    use_best_model=False, allow_writing_files=False, verbose=False,
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def rmse(y, prediction):
    return math.sqrt(math.fsum((float(a) - float(b)) ** 2
                             for a, b in zip(y, prediction)) / len(y))


def log(message):
    print(message, flush=True)
    with (OUT / 'progress.log').open('a', encoding='utf-8') as stream:
        stream.write(message + '\n')


def load_common():
    spec = importlib.util.spec_from_file_location('ec_fixed_common_catboost', COMMON_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fingerprint(frame, columns):
    values = pd.util.hash_pandas_object(frame[columns], index=False).values
    return hashlib.sha256(values.tobytes()).hexdigest()


def finish(raw, tr, va, core):
    return np.clip(core.shrink(np.asarray(raw, dtype=float), va),
                   tr.sub_ec.min(), tr.sub_ec.max())


def fold_identity(fold):
    if isinstance(fold, dict):
        return str(fold['validator']), int(fold['index'])
    return str(fold[0]), int(fold[1])


def fit_one(common, data, lab, fold, locks, core, seed, CatBoostRegressor):
    validator, index = fold_identity(fold)
    tr, va = common.split_fold(data, lab, fold, locks)
    assert len(tr) and len(va)
    assert tr.row_id.is_unique and va.row_id.is_unique
    assert not set(tr.row_id) & set(va.row_id)
    assert len(core.FULL) == 38 and 'farm' not in core.FULL
    assert not set(['in_rad', 'act_side', 'act_valve', 'act_cool', 'act_pump']) & set(core.FULL)
    for frame in [tr, va]:
        assert all((str(f), int(d)) not in locks for f, d in zip(frame.farm, frame.day))
        assert np.isfinite(frame.sub_ec.to_numpy(dtype=float)).all()
        assert not np.isinf(frame[core.FULL].to_numpy(dtype=float)).any()
    path = OUT / f'{validator}_{index}_seed{seed}.npz'
    baseline_info = common.baseline(va, validator, index, seed)
    baseline_v2 = np.asarray(baseline_info['v2'], dtype=float)
    assert baseline_v2.shape == (len(va),) and np.isfinite(baseline_v2).all()
    provenance = {
        'script': sha(__file__), 'protocol': sha(HERE / 'PROTOCOL.md'),
        'common': sha(COMMON_PATH),
        'core': sha(Path(core.__file__)),
        'parameters': PARAMETERS | {'random_seed': seed},
        'train': fingerprint(tr, ['row_id', 'sub_ec'] + core.FULL),
        'validation': fingerprint(va, ['row_id', 'sub_ec'] + core.FULL),
        'baseline_cache_sha256': baseline_info['cache_hash'],
    }
    key = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()
    manifest_path = path.with_suffix('.json')
    if path.exists():
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        assert manifest['key'] == key, f'checkpoint provenance mismatch: {path}'
        assert manifest['prediction_sha256'] == sha(path)
        with np.load(path, allow_pickle=False) as cached:
            assert cached['row_id'].tolist() == va.row_id.tolist()
            assert np.array_equal(cached['baseline_v2'], baseline_v2)
            pred = cached['catboost'].copy()
            raw = cached['raw_catboost'].copy()
        assert np.array_equal(pred, finish(raw, tr, va, core))
        log(f'{validator}/{index} seed{seed}: verified cached prediction')
    else:
        started = time.monotonic()
        model = CatBoostRegressor(**PARAMETERS, random_seed=seed)
        model.fit(tr[core.FULL], tr.sub_ec.to_numpy(dtype=float))
        assert model.tree_count_ == PARAMETERS['iterations']
        raw = np.asarray(model.predict(va[core.FULL]), dtype=float)
        pred = finish(raw, tr, va, core)
        assert raw.shape == pred.shape == (len(va),)
        assert np.isfinite(raw).all() and np.isfinite(pred).all()
        repeated = np.asarray(model.predict(va.iloc[:8][core.FULL]), dtype=float)
        assert np.array_equal(raw[:len(repeated)], repeated), 'single/repeated prediction differs'
        train_raw = np.asarray(model.predict(tr[core.FULL]), dtype=float)
        train_final = finish(train_raw, tr, tr, core)
        np.savez_compressed(path, row_id=va.row_id.to_numpy(dtype=str),
                            raw_catboost=raw, catboost=pred, baseline_v2=baseline_v2,
                            sub_ec=va.sub_ec.to_numpy(dtype=float))
        manifest = {
            'key': key, 'provenance': provenance, 'validator': validator,
            'fold': index, 'seed': seed, 'train_rows': len(tr), 'val_rows': len(va),
            'train_days': len(tr[['farm', 'day']].drop_duplicates()),
            'validation_days': len(va[['farm', 'day']].drop_duplicates()),
            'train_raw_rmse': rmse(tr.sub_ec, train_raw),
            'train_postprocessed_rmse': rmse(tr.sub_ec, train_final),
            'validation_rmse': rmse(va.sub_ec, pred),
            'baseline_v2_rmse': rmse(va.sub_ec, baseline_v2),
            'elapsed_seconds': time.monotonic() - started,
            'prediction_sha256': sha(path), 'repeat_prediction': 'PASS',
        }
        save_json(manifest_path, manifest)
        log(f'{validator}/{index} seed{seed}: CatBoost {manifest["validation_rmse"]:.9f}; '
            f'v2 {manifest["baseline_v2_rmse"]:.9f}; {manifest["elapsed_seconds"]:.1f}s')
    result = va[['row_id', 'farm', 'day', 'hour', 'block', 'sub_ec']].copy()
    result['validator'] = validator
    result['validation_fold'] = index
    result['seed'] = seed
    result['raw_catboost'] = raw
    result['catboost'] = pred
    result['baseline_v2'] = baseline_v2
    result['v2'] = baseline_v2
    return result, manifest


def paired_bootstrap(diag):
    # Fixed, farm-stratified five-day blocks match phase3's original block column.
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    sums = np.zeros((BOOTSTRAP_DRAWS, 3), dtype=float)
    for _, group in diag.groupby('farm', sort=True):
        table = pd.DataFrame({
            'block': group.block, 'n': np.ones(len(group)),
            'baseline': (group.baseline_v2 - group.sub_ec) ** 2,
            'candidate': (group.catboost - group.sub_ec) ** 2,
        }).groupby('block')[['n', 'baseline', 'candidate']].sum().to_numpy(dtype=float)
        draws = rng.integers(0, len(table), size=(BOOTSTRAP_DRAWS, len(table)))
        sums += table[draws].sum(axis=1)
    delta = np.sqrt(sums[:, 2] / sums[:, 0]) - np.sqrt(sums[:, 1] / sums[:, 0])
    alpha = .025 / BONFERRONI_ARMS
    interval = np.quantile(delta, [alpha, 1 - alpha]).tolist()
    p_worse = float(np.mean(delta >= 0))
    return {
        'draws': BOOTSTRAP_DRAWS, 'rng_seed': BOOTSTRAP_SEED,
        'bootstrap_unit': 'farm-stratified original five-day block; all hours retained',
        'difference_rmse': rmse(diag.sub_ec, diag.catboost) - rmse(diag.sub_ec, diag.baseline_v2),
        'ci_bonferroni5': interval, 'p_worse': p_worse, 'threshold': alpha,
        'statistical_gate_pass': bool(p_worse < alpha and interval[1] < 0),
    }


def summarize(predictions, manifests, common):
    summaries = []
    for (validator, seed), group in predictions.groupby(['validator', 'seed'], sort=True):
        unique = group.groupby('row_id')[['sub_ec', 'catboost', 'baseline_v2']].mean()
        candidates = [m['validation_rmse'] for m in manifests
                      if m['validator'] == validator and m['seed'] == seed]
        bases = [m['baseline_v2_rmse'] for m in manifests
                 if m['validator'] == validator and m['seed'] == seed]
        candidate_rmse = rmse(group.sub_ec, group.catboost)
        baseline_rmse = rmse(group.sub_ec, group.baseline_v2)
        summaries.append({
            'validator': validator, 'seed': seed,
            'candidate_rmse_occurrence': candidate_rmse,
            'baseline_rmse_occurrence': baseline_rmse,
            'relative_change': candidate_rmse / baseline_rmse - 1,
            'strict_improvement': candidate_rmse < baseline_rmse,
            'candidate_rmse_unique_average': rmse(unique.sub_ec, unique.catboost),
            'baseline_rmse_unique_average': rmse(unique.sub_ec, unique.baseline_v2),
            'row_occurrences': len(group), 'unique_rows': len(unique),
            'candidate_fold_rmse_mean': float(np.mean(candidates)),
            'candidate_fold_rmse_std': float(np.std(candidates)),
            'baseline_fold_rmse_mean': float(np.mean(bases)),
            'baseline_fold_rmse_std': float(np.std(bases)),
        })
    assert len(summaries) == len(SEEDS) * len(VALIDATORS)
    boots = {}
    for seed in SEEDS:
        diag = predictions[predictions.validator.eq('DIAG10') & predictions.seed.eq(seed)]
        assert len(diag) == 8640 and diag.row_id.is_unique
        assert len(diag[['farm', 'day']].drop_duplicates()) == 360
        boots[str(seed)] = paired_bootstrap(diag)
    averaged = predictions.groupby(
        ['validator', 'validation_fold', 'row_id', 'farm', 'day', 'hour', 'block'], as_index=False
    )[['sub_ec', 'catboost', 'baseline_v2']].mean()
    ensemble = []
    for validator, group in averaged.groupby('validator', sort=True):
        ensemble.append({'validator': validator,
                         'candidate_rmse': rmse(group.sub_ec, group.catboost),
                         'baseline_v2_rmse': rmse(group.sub_ec, group.baseline_v2)})
    boots['seed_average_diagnostic_only'] = paired_bootstrap(
        averaged[averaged.validator.eq('DIAG10')]
    )
    score_gate = all(row['strict_improvement'] for row in summaries)
    statistical_gate = all(boots[str(seed)]['statistical_gate_pass'] for seed in SEEDS)
    common_evaluation = common.evaluate(predictions, ['catboost'])
    save_json(HERE / 'common_evaluation.json', common_evaluation)
    result = {
        'status': 'COMPLETE', 'one_arm': 'CatBoost standalone fixed parameters',
        'score_gate_pass': score_gate, 'statistical_gate_pass': statistical_gate,
        'public_selection_gate_pass': bool(score_gate and statistical_gate),
        'candidate_adopted': False, 'independent_verification_pending': True,
        'final_lock_scored': False, 'submission_created': False,
        'bootstrap_DIAG10': boots, 'seed_average_diagnostic_only': ensemble,
    }
    pd.DataFrame(summaries).to_csv(HERE / 'summary_scores.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(manifests).drop(columns=['provenance']).to_csv(
        HERE / 'fold_scores.csv', index=False, encoding='utf-8-sig')
    save_json(HERE / 'result.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true', help='Execute after protocol/code pre-registration.')
    arguments = parser.parse_args()
    if not arguments.run:
        raise SystemExit('실행하지 않았습니다. 사전등록 커밋 후 --run으로 실행하세요.')
    try:
        import catboost
        from catboost import CatBoostRegressor
    except ImportError as error:
        raise SystemExit('CatBoost 미설치: 모델 실행 전 환경을 준비해야 합니다. 자동 설치하지 않습니다.') from error
    common = load_common()
    data, lab, folds, locks, core = common.prepare()
    assert len(folds) == 22
    assert set(fold_identity(fold)[0] for fold in folds) == set(VALIDATORS)
    assert len(lab) == 8640 and lab.row_id.is_unique
    OUT.mkdir(parents=True, exist_ok=True)
    save_json(OUT / 'environment.json', {
        'python': platform.python_version(), 'numpy': np.__version__,
        'pandas': pd.__version__, 'catboost': catboost.__version__,
        'parameters': PARAMETERS, 'seeds': SEEDS,
        'script_sha256': sha(__file__), 'protocol_sha256': sha(HERE / 'PROTOCOL.md'),
        'common_sha256': sha(COMMON_PATH),
    })
    records = []
    manifests = []
    for fold in folds:
        for seed in SEEDS:
            result, manifest = fit_one(common, data, lab, fold, locks, core, seed, CatBoostRegressor)
            records.append(result)
            manifests.append(manifest)
    predictions = pd.concat(records, ignore_index=True)
    predictions.to_csv(OUT / 'oof_predictions.csv', index=False, encoding='utf-8-sig')
    report = summarize(predictions, manifests, common)
    save_json(HERE / 'completion.json', {
        'status': 'COMPLETE', 'fold_seed_models': len(manifests),
        'oof_sha256': sha(OUT / 'oof_predictions.csv'),
        'script_sha256': sha(__file__), 'protocol_sha256': sha(HERE / 'PROTOCOL.md'),
        'public_selection_gate_pass': report['public_selection_gate_pass'],
        'candidate_adopted': False, 'independent_verification_pending': True,
    })
    log(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
