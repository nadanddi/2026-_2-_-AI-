"""Strict original model completion manifests; never a numerical/model score gate."""
from pathlib import Path, PurePosixPath, PureWindowsPath
import hashlib
import json


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition: raise ValueError(message)



def contained_path(root, relative):
    root = Path(root).resolve(strict=True)
    path = (root / relative).resolve(strict=True)
    require(path.is_relative_to(root), 'resolved output path escaped root')
    return path

def verify_manifest(manifest, expected_names, read_file):
    """Validate exact canonical names before accessing any named output bytes."""
    require(type(manifest) is dict, 'file manifest must be a dict')
    require(set(manifest) == set(expected_names), 'manifest missing or extra files')
    for name, expected in manifest.items():
        require(type(name) is str and '\\' not in name and ':' not in name and not PureWindowsPath(name).drive, 'noncanonical relative name')
        path = PurePosixPath(name)
        require(not path.is_absolute() and len(path.parts) == 2 and
                all(part not in ('', '.', '..') for part in path.parts) and
                path.as_posix() == name, 'unsafe manifest name')
        require(type(expected) is str and len(expected) == 64 and
                all(c in '0123456789abcdef' for c in expected), 'invalid file SHA')
    hashes = {}
    for name in sorted(manifest):
        value = hashlib.sha256(read_file(name)).hexdigest()
        require(value == manifest[name], 'output SHA mismatch: ' + name)
        hashes[name] = value
    return hashes


def expected_fold_files(fold, rawreg, kind):
    folder = f'{fold["validator"]}_fold{fold["fold"]}'
    require(kind in ('raw', 'pfn'), 'unknown model kind')
    if kind == 'raw':
        require(rawreg['seeds'] == [47, 1414, 6464] and len(rawreg['family_map']) == 24,
                'raw recipe mismatch')
        names = [f'BASELINE_{member}_seed{s}.json' for s in rawreg['seeds'] for member in ('ET', 'LGB', 'MLP')]
        names += [f'{cid}_seed{s}.json' for s in rawreg['seeds'] for cid in sorted(rawreg['family_map'])]
        require(len(names) == len(set(names)) == 81, 'raw cell collision')
    else:
        names = [f'context{s}{suffix}.json' for s in (5, 6, 7, 8) for suffix in ('', '_audit')]
    return {folder + '/' + name for name in names}


def verify_fold(root, fold, rawreg, kind, registration_sha):
    root = Path(root).resolve()
    folder = f'{fold["validator"]}_fold{fold["fold"]}'
    require('/' not in folder and '\\' not in folder and '..' not in folder, 'invalid fold directory')
    expected = expected_fold_files(fold, rawreg, kind)
    path = root / folder / 'complete.json'
    blob = contained_path(root, path.relative_to(root)).read_bytes()
    before = hashlib.sha256(blob).hexdigest()
    complete = json.loads(blob.decode('utf-8'))
    status = ('ORIGINAL_FOLD_R3_DOMAIN_RAW_COMPLETE_NO_SCORE' if kind == 'raw' else
              'ORIGINAL_FOLD_PFN4_RAW_AND_AUDITS_COMPLETE_NO_SCORE')
    require(complete['status'] == status and complete['registration_sha256'] == registration_sha,
            'wrong fold completion contract')
    require(complete['heldout_truth_loaded'] is False, 'fold receipt read heldout truth')
    if kind == 'raw':
        require(complete['candidate_fit_count'] == 72 and complete['baseline_fit_count'] == 9,
                'wrong fold raw counts')
    existing = {folder + '/' + p.name for p in (root / folder).glob('*.json') if p.name != 'complete.json'}
    require(existing == expected, 'disk fold files missing or unexpected')
    hashes = verify_manifest(complete['files_sha256'], expected, lambda name: contained_path(root, name).read_bytes())
    require(hashlib.sha256(contained_path(root, path.relative_to(root)).read_bytes()).hexdigest() == before, 'fold completion changed during audit')
    return {'fold': folder, 'completion_sha256': before, 'files_sha256': hashes,
            'status': 'FOLD_MANIFEST_PASS_NOT_NUMERICAL_OR_FULL_MODEL_GATE'}


def verify_all(root, registry, rawreg, kind, registration_sha):
    """Require sealed66 completion before any full-manifest claim.

    This checks storage completeness, not producer audit semantics, input/label
    lineage, fresh feature matrices, fitted statistics or prediction causality.
    """
    root = Path(root).resolve()
    folds = registry['folds']
    require(len(folds) == 66 and len({(f['validator'], f['fold']) for f in folds}) == 66,
            'original66 registry required')
    path = root / 'complete.json'
    blob = contained_path(root, path.relative_to(root)).read_bytes()
    before = hashlib.sha256(blob).hexdigest()
    complete = json.loads(blob.decode('utf-8'))
    status = ('ORIGINAL66_R3_DOMAIN_RAW_COMPLETE_NO_FULL_BASELINE_OR_SCORE' if kind == 'raw' else
              'ORIGINAL66_PFN264_CONTEXTS_RAW_AND_AUDITS_COMPLETE_NO_SCORE')
    require(complete['status'] == status and complete['registration_sha256'] == registration_sha,
            'wrong global completion contract')
    require(complete['heldout_truth_loaded'] is False, 'global receipt read heldout truth')
    if kind == 'raw':
        require(complete['folds'] == 66 and complete['baseline_fit_count'] == 594 and
                complete['candidate_fit_count'] == 4752, 'wrong global raw counts')
    else:
        require(complete['whole_baseline_complete'] is False, 'PFN receipt cannot claim full baseline')
    expected = set().union(*(expected_fold_files(f, rawreg, kind) for f in folds))
    require(len(expected) == (5346 if kind == 'raw' else 528), 'global file count mismatch')
    require(set(complete['files_sha256']) == expected, 'global manifest coverage mismatch')
    receipts = [verify_fold(root, f, rawreg, kind, registration_sha) for f in folds]
    combined = {name: value for r in receipts for name, value in r['files_sha256'].items()}
    require(combined == complete['files_sha256'], 'global/fold manifests disagree')
    require(hashlib.sha256(contained_path(root, path.relative_to(root)).read_bytes()).hexdigest() == before, 'global completion changed during audit')
    return {'status': 'ORIGINAL66_COMPLETION_MANIFEST_PASS_NOT_FULL_MODEL_GATE',
            'kind': kind, 'completion_sha256': before, 'files_sha256': combined,
            'fold_completion_sha256': {r['fold']: r['completion_sha256'] for r in receipts},
            'heldout_truth_read': False, 'model_fit': False, 'whole_pipeline_gate_passed': False}

