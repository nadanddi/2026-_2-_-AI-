"""Input-only DP1 replay. No labels, model imports, fit, test or score."""
import sys
from pathlib import Path
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / '집/클로드/research'))
import env
import ast, csv, hashlib, json, math
from collections import Counter
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SOURCE = ROOT / '집/클로드/research/ec3_DP1_daily_operation_pattern_v1.py'
INPUT = Path(env.DATA) / 'train_X.csv'
PUBLIC = ROOT / '집/코덱스/local/ec_dc4_integration_20261002_v1/v2_integration_oof.csv'
CONTROLS = ['act_vent', 'act_thermal', 'act_shade', 'act_heating', 'act_co2']
NEW = ['seal_run', 'vent_open_hours', 'first_open_hour', 'thermal_switches',
       'shade_switches', 'since_curtain_change', 'heat_run', 'co2_hours', 'vent_max']

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def extracted():
    tree = ast.parse(SOURCE.read_text(encoding='utf-8-sig'))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in {'run_len', 'day_feats'}]
    assert len(nodes) == 2
    namespace = {'np': np, 'pd': pd}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), 'exec'), namespace)
    return namespace['day_feats']

def scalar(g):
    # Independent loop reproduces the original zero-fill and row-position semantics.
    result = []
    seal = opened = heat = co2 = ts = ss = 0
    first = 24
    vmax = -math.inf
    previous = None
    last = -1
    for k, row in enumerate(g.sort_values('hour').itertuples()):
        values = [getattr(row, c) for c in CONTROLS]
        v, th, sh, he, co = [0. if pd.isna(x) else float(x) for x in values]
        seal = seal + 1 if v == 0 else 0
        opened += v > 0
        if first == 24 and v > 0:
            first = int(row.hour)
        state = (th > 0, sh > 0)
        if previous is not None:
            changed_t, changed_s = state[0] != previous[0], state[1] != previous[1]
            ts += changed_t
            ss += changed_s
            if changed_t or changed_s:
                last = k
        previous = state
        heat = heat + 1 if he > 0 else 0
        co2 += co > 0
        vmax = max(vmax, v)
        result.append([seal, opened, first, ts, ss, k-last if last >= 0 else k+1, heat, co2, vmax])
    return np.asarray(result, dtype=float)

def profile(d, day_feats):
    columns = {}
    for c in CONTROLS:
        columns[c] = dict(missing=int(d[c].isna().sum()), observed_zero=int(d[c].eq(0).sum()),
                          observed_positive=int(d[c].gt(0).sum()), negative=int(d[c].lt(0).sum()))
    grids, affected, replay, feature_differences = [], [], 0, {c: 0 for c in NEW}
    for (farm, day), g in d.groupby(['farm', 'day'], sort=True):
        g = g.sort_values('hour')
        hours = g.hour.to_numpy()
        if not np.array_equal(hours, np.arange(24)):
            grids.append(dict(farm=farm, day=int(day), hours=hours.tolist()))
        original = day_feats(g)[NEW].to_numpy(dtype=float)
        separate = scalar(g)
        assert np.array_equal(original, separate)
        replay += original.size
        # Two extreme completions measure sensitivity to truly missing controls only.
        zero = g.copy()
        positive = g.copy()
        zero[CONTROLS] = zero[CONTROLS].fillna(0.)
        positive[CONTROLS] = positive[CONTROLS].fillna(1.)
        other = day_feats(positive)[NEW].to_numpy(dtype=float)
        differences = original != other
        for i, name in enumerate(NEW):
            feature_differences[name] += int(differences[:, i].sum())
        if differences.any():
            affected.append(dict(farm=farm, day=int(day), missing_cells=int(g[CONTROLS].isna().sum().sum()),
                                 changed_feature_cells=int(differences.sum())))
        assert np.array_equal(original, day_feats(zero)[NEW].to_numpy(dtype=float))
    return dict(rows=len(d), days=len(d[['farm','day']].drop_duplicates()),
                columns=columns, incomplete_hour_grids=grids,
                zero_vs_positive_missing_completion_changed_cells=feature_differences,
                affected_days=affected, independent_replay_elements=replay)

def main():
    output = HERE / ('input_audit_v2.json' if '--verify' in sys.argv else 'input_audit_v1.json')
    assert not output.exists(), output
    ids = set()
    with PUBLIC.open(encoding='utf-8-sig', newline='') as handle:
        for row in csv.DictReader(handle):
            if row['validator'] == 'DIAG10' and row['seed'] == '7':
                assert row['row_id'] not in ids
                ids.add(row['row_id'])
    assert len(ids) == 8640
    d = pd.read_csv(INPUT, usecols=['row_id']+CONTROLS, float_precision='round_trip')
    d = d[d.row_id.str[:3].isin(['F13','F47'])].copy()
    assert d.row_id.is_unique
    d['farm'] = d.row_id.str[:3]
    d['day'] = d.row_id.str[4:7].astype(int)
    d['hour'] = d.row_id.str[8:10].astype(int)
    # Different parser verifies missing/zero/positive counts, without loading a target.
    counter = {c: Counter() for c in CONTROLS}
    csv_rows = 0
    with INPUT.open(encoding='utf-8-sig', newline='') as handle:
        for row in csv.DictReader(handle):
            if row['row_id'][:3] not in {'F13','F47'}:
                continue
            csv_rows += 1
            for c in CONTROLS:
                value = float(row[c]) if row[c].strip() else math.nan
                counter[c]['missing' if math.isnan(value) else 'zero' if value == 0 else 'positive' if value > 0 else 'negative'] += 1
    assert csv_rows == len(d)
    for c in CONTROLS:
        assert counter[c]['missing'] == int(d[c].isna().sum())
        assert counter[c]['zero'] == int(d[c].eq(0).sum())
        assert counter[c]['positive'] == int(d[c].gt(0).sum())
    day_feats = extracted()
    public = d[d.row_id.isin(ids)]
    assert set(public.row_id) == ids
    result = dict(status='PASS_INPUT_ONLY', input_sha256=sha(INPUT), source_sha256=sha(SOURCE),
                  public_id_source_sha256=sha(PUBLIC), target_values_accessed=0, test_reads=0,
                  fit=0, predict=0, scores=0, cleaning_applied=0,
                  full_input=profile(d, day_feats), public_input=profile(public, day_feats),
                  limitation='Only input replay and missing-value sensitivity. No actual DP1 training-frame replay or performance/causal proof.')
    if '--verify' in sys.argv:
        prior = json.loads((HERE/'input_audit_v1.json').read_text(encoding='utf-8'))
        assert prior == result
    with output.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps({k: result[k] for k in ['status','full_input','public_input']}, ensure_ascii=False))

if __name__ == '__main__':
    main()
