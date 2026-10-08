"""Register Windows path aggregation fix with unchanged raw3 producer sources."""
from pathlib import Path, PureWindowsPath
import hashlib
import json
import sys

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    if not __debug__ or sys.flags.optimize:
        raise RuntimeError('Python -O prohibited')
    regpath = HERE / 'DOMAIN24_original_raw_fit_registration_v3.json'
    raw = json.loads(regpath.read_text(encoding='utf-8'))
    assert raw['status'] == 'REGISTERED_ORIGINAL66_R3_DOMAIN24_RAW_FIT_BEFORE_FIT'
    assert all(sha(p) == v for p, v in raw['source_sha256'].items())
    assert raw['baseline_fit_count'] + raw['candidate_fit_count'] == 5346
    registry = json.loads((HERE / 'original_validator_endpoint_registry_v1.json').read_text(encoding='utf-8'))
    assert len(registry['folds']) == 66
    # Exercise path normalization without fitting or opening predictions.
    outputs = {}
    old_count = None
    for fold in registry['folds']:
        folder = f'{fold["validator"]}_fold{fold["fold"]}'
        cells = [f'BASELINE_{member}_seed{seed}.json' for seed in raw['seeds'] for member in ('ET', 'LGB', 'MLP')]
        cells += [f'{cid}_seed{seed}.json' for seed in raw['seeds'] for cid in sorted(raw['family_map'])]
        assert len(cells) == 81
        paths = [PureWindowsPath(folder) / name for name in cells]
        if old_count is None:
            old_count = sum(str(p).startswith(folder + '/') for p in paths)
            assert old_count == 0
        for path in paths: outputs[path.as_posix()] = True
        assert sum(p.startswith(folder + '/') for p in outputs) == 81
    assert len(outputs) == 5346
    driver = HERE / 'run_domain_original_raw_driver_v4.py'
    producer = HERE / 'run_domain_original_raw_v3.py'
    pins = dict(raw['source_sha256'])
    for p in (regpath, driver, Path(__file__)):
        p = p.resolve()
        value = sha(p)
        if str(p) in pins: assert pins[str(p)] == value
        pins[str(p)] = value
    result = {'status': 'REGISTERED_RAW_DRIVER_PATH_FIX_NO_MODEL_CHANGE',
              'driver_sha256': sha(driver), 'producer_sha256': sha(producer),
              'producer_registration_sha256': sha(regpath), 'sources_sha256': pins,
              'total_fits': 5346, 'folds': 66, 'fits_per_fold': 81,
              'models_changed': False, 'raw_prediction_contract_changed': False,
              'old_windows_first_fold_count': old_count,
              'synthetic_windows_folds_checked': 66,
              'heldout_truth_read': False, 'score_computed': False,
              'changes': ['driver relative output paths normalized with as_posix',
                          'additional driver/source registration guard before producer3 reuse'],
              'limits': ['No model completeness or full score gate is asserted by this registration']}
    with (HERE / 'DOMAIN24_original_raw_driver_registration_v4.json').open('x', encoding='utf-8') as out:
        json.dump(result, out, ensure_ascii=False, indent=2)
    print(f'Raw driver4 registered: {len(pins)} pins, Windows66/5346 path checks; no fit/score')


if __name__ == '__main__': main()
