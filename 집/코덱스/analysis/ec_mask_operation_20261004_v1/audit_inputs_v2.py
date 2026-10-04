"""Fresh input audit with finite checks, strata and causal prefix replay."""
import sys
from pathlib import Path
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'집/클로드/research'))
import env
import csv, json, importlib.util
import numpy as np
import pandas as pd
spec = importlib.util.spec_from_file_location('input_audit_v1', HERE/'audit_inputs_v1.py')
A = importlib.util.module_from_spec(spec)
spec.loader.exec_module(A)

def extra(d, fn):
    assert np.isfinite(d[A.CONTROLS].to_numpy(dtype=float)).all()
    prefix_elements = 0
    for _, g in d.groupby(['farm','day'], sort=True):
        g = g.sort_values('hour')
        full = fn(g)
        for h in [0,6,12,23]:
            part = g[g.hour <= h]
            assert np.array_equal(fn(part)[A.NEW].to_numpy(), full.loc[part.index,A.NEW].to_numpy())
            prefix_elements += len(part)*len(A.NEW)
    strata = []
    for farm in ['F13','F47']:
        for late in [False, True]:
            g = d[(d.farm == farm) & ((d.day >= 179) == late)]
            strata.append(dict(farm=farm, late=late, rows=len(g), days=len(g.day.unique()),
                               missing={c:int(g[c].isna().sum()) for c in A.CONTROLS}))
    return dict(finite_values=True, prefix_comparison_elements=prefix_elements, strata=strata)

def main():
    out = HERE/'input_audit_v2.json'
    assert not out.exists(), out
    prior = json.loads((HERE/'input_audit_v1.json').read_text(encoding='utf-8'))
    assert A.sha(A.INPUT) == prior['input_sha256']
    assert A.sha(A.SOURCE) == prior['source_sha256']
    assert A.sha(A.PUBLIC) == prior['public_id_source_sha256']
    d = pd.read_csv(A.INPUT, usecols=['row_id']+A.CONTROLS, float_precision='round_trip')
    d = d[d.row_id.str[:3].isin(['F13','F47'])].copy()
    assert d.row_id.is_unique
    d['farm'] = d.row_id.str[:3]
    d['day'] = d.row_id.str[4:7].astype(int)
    d['hour'] = d.row_id.str[8:10].astype(int)
    ids = set()
    with A.PUBLIC.open(encoding='utf-8-sig', newline='') as handle:
        for row in csv.DictReader(handle):
            if row['validator'] == 'DIAG10' and row['seed'] == '7':
                assert row['row_id'] not in ids
                ids.add(row['row_id'])
    assert len(ids) == 8640
    public = d[d.row_id.isin(ids)]
    assert set(public.row_id) == ids
    fn = A.extracted()
    fresh_full, fresh_public = A.profile(d, fn), A.profile(public, fn)
    assert fresh_full == prior['full_input']
    assert fresh_public == prior['public_input']
    result = dict(status='PASS_FRESH_INPUT_REPLAY', input_sha256=A.sha(A.INPUT),
                  dp1_source_sha256=A.sha(A.SOURCE), audit_v1_sha256=A.sha(HERE/'audit_inputs_v1.py'),
                  audit_v2_sha256=A.sha(Path(__file__)), full_input=fresh_full, public_input=fresh_public,
                  full_extra=extra(d, fn), public_extra=extra(public, fn),
                  numeric_targets_used=0, target_fields_queried=0, test_reads=0, fit=0, predict=0, scores=0,
                  limitations=['Public CSV lines are parsed as strings, but target fields are never queried or converted.',
                               'Same-implementation rerun confirms reproducibility; v1 scalar and CSV parser are separate numerical checks.',
                               'Prefix replay covers these operation features only, not all model features or predictions.',
                               'No original DP1 training-frame replay, unseen holdout or causal effect test.'])
    with out.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps(dict(status=result['status'], full_extra=result['full_extra'],
                         public_extra=result['public_extra']), ensure_ascii=False))

if __name__ == '__main__':
    main()
