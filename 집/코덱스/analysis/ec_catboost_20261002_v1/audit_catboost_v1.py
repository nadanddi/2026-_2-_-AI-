"""Read completed artifacts; independently verify arithmetic and describe EC errors."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / '집' / '클로드' / 'research'))
import env

import csv
import hashlib
import json
import math

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = ROOT / '집' / '코덱스' / 'local' / 'ec_catboost_20261002_v1'
SEEDS = [7, 101, 2024]
NBOOT = 20000


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rmse(y, prediction):
    return math.sqrt(math.fsum((float(a) - float(b)) ** 2 for a, b in zip(y, prediction)) / len(y))


def independent_bootstrap(group):
    units = {}
    for row in group.itertuples(index=False):
        key = (row.farm, int(row.block))
        units.setdefault(key, []).append((float(row.sub_ec), float(row.baseline_v2), float(row.catboost)))
    keys = sorted(units)
    assert len(keys) == 80
    table = np.array([[len(units[key]),
                       math.fsum((b - y) ** 2 for y, b, c in units[key]),
                       math.fsum((c - y) ** 2 for y, b, c in units[key])]
                      for key in keys], dtype=float)
    generator = np.random.default_rng(918)
    draws = []
    for farm in sorted({key[0] for key in keys}):
        choices = np.array([i for i, key in enumerate(keys) if key[0] == farm])
        draws.append(choices[generator.integers(0, len(choices), size=(NBOOT, len(choices)))])
    index = np.concatenate(draws, axis=1)
    sums = table[index].sum(axis=1)
    dmse = (sums[:, 2] - sums[:, 1]) / sums[:, 0]
    drmse = np.sqrt(sums[:, 2] / sums[:, 0]) - np.sqrt(sums[:, 1] / sums[:, 0])
    return {'p_worse': float(np.mean(dmse >= 0)),
            'ci_mse': np.quantile(dmse, [.005, .995]).tolist(),
            'ci_rmse': np.quantile(drmse, [.005, .995]).tolist(), 'n_blocks': 80}


def main():
    assert (HERE / 'completion.json').exists(), 'CatBoost 全22fold×3seed 완료 후 실행'
    assert not (HERE / 'audit_verified_v1.json').exists(), '기존 검산 산출물을 덮어쓰지 않음'
    completion = json.loads((HERE / 'completion.json').read_text(encoding='utf-8'))
    assert completion['status'] == 'COMPLETE' and completion['fold_seed_models'] == 66
    assert completion['oof_sha256'] == sha(OUT / 'oof_predictions.csv')
    d = pd.read_csv(OUT / 'oof_predictions.csv')
    assert set(d.seed) == set(SEEDS)
    assert not d.duplicated(['validator', 'validation_fold', 'seed', 'row_id']).any()
    locks = {(r['farm'], int(r['day'])) for r in
             json.loads((Path(env.CODEX) / 'ec_final_lock' / 'locked_days.json').read_text(encoding='utf-8'))['selected']}
    truth = {}
    with (Path(env.DATA) / 'train_y.csv').open(newline='', encoding='utf-8-sig') as stream:
        for record in csv.DictReader(stream):
            farm, day, _ = record['row_id'].split('_')
            if farm not in ['F13', 'F47'] or (farm, int(day)) in locks:
                continue
            truth[record['row_id']] = float(record['sub_ec'])
    assert len(truth) == 8640
    for row in d.itertuples(index=False):
        assert math.isclose(float(row.sub_ec), truth[row.row_id], abs_tol=5e-15, rel_tol=0)
    manifests = []
    max_post_difference = 0.0
    for (validator, index, seed), group in d.groupby(['validator', 'validation_fold', 'seed'], sort=True):
        path = OUT / f'{validator}_{index}_seed{seed}.npz'
        manifest = json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
        assert manifest['prediction_sha256'] == sha(path)
        assert manifest['provenance']['script'] == sha(HERE / 'run_catboost.py')
        assert manifest['provenance']['protocol'] == sha(HERE / 'PROTOCOL.md')
        baseline_path = ROOT / '집' / '코덱스' / 'local' / 'ec_restart_phase3_20261001_v1' / f'{validator}_{index}.npz'
        baseline_manifest = json.loads(baseline_path.with_suffix('.json').read_text(encoding='utf-8'))
        assert baseline_manifest['prediction_sha256'] == sha(baseline_path)
        assert manifest['provenance']['baseline_cache_sha256'] == sha(baseline_path)
        validation_days = set(group[['farm', 'day']].itertuples(index=False, name=None))
        banned = {(f, int(day) + offset) for f, day in validation_days | locks for offset in [-1, 0, 1]}
        train_labels = [value for rid, value in truth.items()
                        if (rid[:3], int(rid[4:7])) not in banned]
        assert len(train_labels) == manifest['train_rows']
        lo, hi = min(train_labels), max(train_labels)
        with np.load(path, allow_pickle=False) as z, np.load(baseline_path, allow_pickle=False) as base:
            assert z['row_id'].tolist() == group.row_id.tolist() == base['row_id'].tolist()
            assert np.allclose(z['sub_ec'], group.sub_ec, rtol=0, atol=5e-15)
            assert np.allclose(z['catboost'], group.catboost, rtol=0, atol=5e-15)
            assert np.allclose(z['raw_catboost'], group.raw_catboost, rtol=0, atol=5e-15)
            assert np.array_equal(z['baseline_v2'], base[f'v2_{seed}'])
        for _, day in group.groupby(['farm', 'day'], sort=True):
            raw_values = []
            for row in day.sort_values('hour').itertuples(index=False):
                raw_values.append(float(row.raw_catboost))
                expected = max(lo, min(hi, .5 * row.raw_catboost + .5 * math.fsum(raw_values) / len(raw_values)))
                max_post_difference = max(max_post_difference, abs(expected - row.catboost))
        value = rmse(group.sub_ec, group.catboost)
        assert math.isclose(value, manifest['validation_rmse'], rel_tol=0, abs_tol=1e-12)
        assert math.isclose(rmse(group.sub_ec, group.baseline_v2), manifest['baseline_v2_rmse'], rel_tol=0, abs_tol=1e-12)
        manifests.append(manifest)
    assert len(manifests) == 66 and max_post_difference < 1e-12
    score_rows = []
    saved_scores = pd.read_csv(HERE / 'summary_scores.csv')
    for (validator, seed), group in d.groupby(['validator', 'seed'], sort=True):
        raw = rmse(group.sub_ec, group.raw_catboost)
        post = rmse(group.sub_ec, group.catboost)
        base = rmse(group.sub_ec, group.baseline_v2)
        saved = saved_scores[saved_scores.validator.eq(validator) & saved_scores.seed.eq(seed)].iloc[0]
        assert math.isclose(post, saved.candidate_rmse_occurrence, rel_tol=0, abs_tol=1e-12)
        assert math.isclose(base, saved.baseline_rmse_occurrence, rel_tol=0, abs_tol=1e-12)
        score_rows.append({'validator': validator, 'seed': int(seed), 'raw_rmse': raw,
                           'post_rmse': post, 'v2_rmse': base,
                           'post_change_pct': 100 * (post / raw - 1),
                           'relative_vs_v2_pct': 100 * (post / base - 1)})
    common_result = json.loads((HERE / 'common_evaluation.json').read_text(encoding='utf-8'))
    boots = {}
    for seed in SEEDS:
        diag = d[d.validator.eq('DIAG10') & d.seed.eq(seed)]
        assert len(diag) == 8640 and diag.row_id.is_unique and set(diag.row_id) == set(truth)
        independent = independent_bootstrap(diag)
        saved = next(row for row in common_result['bootstrap'] if row['seed'] == seed)
        assert independent['p_worse'] == saved['p_worse']
        assert np.allclose(independent['ci_rmse'], [saved['ci_rmse_low'], saved['ci_rmse_high']], rtol=0, atol=1e-12)
        assert np.allclose(independent['ci_mse'], [saved['ci_mse_low'], saved['ci_mse_high']], rtol=0, atol=1e-12)
        boots[str(seed)] = independent
    averaged = d[d.validator.eq('DIAG10')].groupby(
        ['row_id', 'farm', 'day', 'hour', 'block'], as_index=False
    )[['sub_ec', 'raw_catboost', 'catboost', 'baseline_v2']].mean()
    x = pd.read_csv(Path(env.DATA) / 'train_X.csv', usecols=['row_id', 'act_circfan', 'act_vent'])
    averaged = averaged.merge(x, on='row_id', validate='one_to_one')
    keys = ['farm', 'day']
    daily_true = averaged.groupby(keys).sub_ec.transform('mean')
    fan = averaged.groupby(keys).act_circfan.transform('mean')
    vent0 = averaged.groupby(keys).act_vent.transform(lambda v: (v == 0).mean())
    high = daily_true.ge(1.2)
    closed = fan.lt(10) & vent0.gt(.85)
    segments = {'all': np.ones(len(averaged), dtype=bool),
                'F13': averaged.farm.eq('F13'), 'F47': averaged.farm.eq('F47'),
                'early': averaged.day.lt(179), 'late': averaged.day.ge(179),
                'hours_0_6': averaged.hour.le(6), 'hours_7_23': averaged.hour.ge(7),
                'high_ec': high, 'normal_ec': ~high, 'sealed': closed,
                'sealed_and_high': closed & high, 'sealed_not_high': closed & ~high}
    segment_rows = []
    for name, mask in segments.items():
        group = averaged[mask]
        for model in ['raw_catboost', 'catboost', 'baseline_v2']:
            error = group[model] - group.sub_ec
            day_error = error.groupby([group.farm, group.day]).transform('mean')
            mse = math.fsum(float(v) ** 2 for v in error) / len(error)
            level = math.fsum(float(v) ** 2 for v in day_error) / len(error)
            shape = math.fsum(float(v) ** 2 for v in error - day_error) / len(error)
            assert math.isclose(mse, level + shape, rel_tol=0, abs_tol=1e-12)
            segment_rows.append({'segment': name, 'model': model, 'rows': len(group),
                                 'days': len(group[keys].drop_duplicates()), 'rmse': math.sqrt(mse),
                                 'bias': float(error.mean()), 'level_rmse': math.sqrt(level),
                                 'shape_rmse': math.sqrt(shape), 'level_fraction': level / mse})
    train_rows = []
    for validator in sorted(set(d.validator)):
        for seed in SEEDS:
            selected = [m for m in manifests if m['validator'] == validator and m['seed'] == seed]
            train_rows.append({'validator': validator, 'seed': seed, 'folds': len(selected),
                               'train_raw_rmse_mean_saved': float(np.mean([m['train_raw_rmse'] for m in selected])),
                               'train_post_rmse_mean_saved': float(np.mean([m['train_postprocessed_rmse'] for m in selected])),
                               'validation_fold_rmse_mean': float(np.mean([m['validation_rmse'] for m in selected])),
                               'training_score_source': 'stored training manifests; no training rerun'})
    pd.DataFrame(score_rows).to_csv(HERE / 'audit_raw_post_scores_v1.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(segment_rows).to_csv(HERE / 'audit_segments_v1.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(train_rows).to_csv(HERE / 'audit_training_gap_v1.csv', index=False, encoding='utf-8-sig')
    verified = {'status': 'PASS', 'source_labels': 'csv.DictReader; lock excluded before EC numeric conversion',
                'locked_labels_scored': False, 'evaluated_test_values': False,
                'checkpoints_verified': len(manifests), 'rmse_cells_verified': len(score_rows),
                'max_independent_postprocess_difference': max_post_difference,
                'bootstrap_independent_crosscheck': boots,
                'source_oof_sha256': sha(OUT / 'oof_predictions.csv'),
                'code_sha256': sha(__file__), 'decisions': common_result['decisions'],
                'training_predictions_replayed': False, 'new_model_trained': False}
    verified['segment_scope'] = (
        'Full-day input sealed grouping and label-derived high_ec are diagnostic only; '
        'no feature, gate, threshold, weight, hyperparameter or candidate selection added.'
    )
    (HERE / 'audit_verified_v1.json').write_text(json.dumps(verified, ensure_ascii=False, indent=2), encoding='utf-8')
    print(pd.DataFrame(score_rows).to_string(index=False))
    print(pd.DataFrame(segment_rows).query("model != 'raw_catboost'").to_string(index=False))
    print(json.dumps(verified, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
