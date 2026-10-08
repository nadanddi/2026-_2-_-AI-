"""Pin PFN completion path normalization before first original context fit."""
from pathlib import Path, PureWindowsPath
import hashlib
import json
import sys

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    if not __debug__ or sys.flags.optimize: raise RuntimeError('Python -O prohibited')
    regpath = HERE / 'DOMAIN24_original_pfn_registration_v2.json'
    raw = json.loads(regpath.read_text(encoding='utf-8'))
    assert raw['status'] == 'REGISTERED_ORIGINAL66_PFN_REFERENCE_ONLY_BEFORE_FIT'
    assert raw['context_seeds'] == [5, 6, 7, 8] and raw['context_fits'] == 264
    assert all(sha(p) == v for p, v in raw['source_sha256'].items())
    registry = json.loads((HERE / 'original_validator_endpoint_registry_v1.json').read_text(encoding='utf-8'))
    assert len(registry['folds']) == 66
    paths = {}
    for fold in registry['folds']:
        folder = f'{fold["validator"]}_fold{fold["fold"]}'
        for seed in raw['context_seeds']:
            for name in (f'context{seed}.json', f'context{seed}_audit.json'):
                path = PureWindowsPath(folder) / name
                assert not str(path).startswith(folder + '/')
                paths[path.as_posix()] = True
        assert sum(p.startswith(folder + '/') for p in paths) == 8
    assert len(paths) == 528
    producer = HERE / 'run_original_pfn_cache_v2.py'
    driver = HERE / 'run_original_pfn_driver_v3.py'
    pins = dict(raw['source_sha256'])
    for p in (regpath, driver, Path(__file__)):
        p = p.resolve(); value = sha(p)
        if str(p) in pins: assert pins[str(p)] == value
        pins[str(p)] = value
    result = {'status': 'REGISTERED_PFN_DRIVER_PATH_FIX_NO_MODEL_CHANGE',
              'driver_sha256': sha(driver), 'producer_sha256': sha(producer),
              'producer_registration_sha256': sha(regpath), 'sources_sha256': pins,
              'context_fits': 264, 'files': 528, 'folds': 66,
              'models_changed': False, 'raw_prediction_contract_changed': False,
              'synthetic_windows_folds_checked': 66, 'synthetic_files_checked': 528,
              'heldout_truth_read': False, 'score_computed': False,
              'changes': ['driver relative output paths normalized with as_posix',
                          'additional driver/source registration guard before producer2 reuse'],
              'limits': ['No actual context fit or full model score gate is asserted']}
    with (HERE / 'DOMAIN24_original_pfn_driver_registration_v3.json').open('x', encoding='utf-8') as out:
        json.dump(result, out, ensure_ascii=False, indent=2)
    print(f'PFN driver3 registered: {len(pins)} pins, Windows66/528 path checks; no fit/score')


if __name__ == '__main__': main()
