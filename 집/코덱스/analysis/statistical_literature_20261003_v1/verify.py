"""Fresh, standard-library-only verification; never reads EC lock or test data."""
from pathlib import Path
import csv, json, math, hashlib

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
saved = json.loads((HERE / 'diagnostic.json').read_text(encoding='utf-8'))
tp = ROOT / '집/코덱스/local/temp_tk_season_20261003_v1/TK1_predictions.csv'
ep = ROOT / '집/코덱스/local/ec_dc4_integration_20261002_v1/v2_integration_oof.csv'
for path in (tp, ep, HERE / 'diagnose.py'):
    assert hashlib.sha256(path.read_bytes()).hexdigest() == saved['source_hashes'][path.name]
with tp.open(encoding='utf-8-sig', newline='') as f:
    temp = list(csv.DictReader(f))
with ep.open(encoding='utf-8-sig', newline='') as f:
    ec = list(csv.DictReader(f))
verified = []
for record in saved['findings']:
    label, segment = record['target'], record['segment']
    if label.startswith('TEMP_'):
        _, seed, context = label.split('_')
        members = {}
        for row in temp:
            if row['validator'] == 'DIAG10' and float(row['base_seed']) == int(seed) and row['context'] == context:
                members.setdefault(row['row_id'], {})[row['member']] = row
        rows = [(r['W30G'], [float(r[m]['prediction']) for m in ('BASE', 'CODEX', 'PFN')], float(r['W30G']['prediction']), float(r['W30G']['sub_temp'])) for r in members.values()]
    else:
        seed = int(label.split('_')[1])
        rows = [(r, [float(r['season_r3']), float(r['season_pfn'])], float(r['season_v2']), float(r['sub_ec'])) for r in ec if r['validator'] == 'DIAG10' and int(float(r['seed'])) == seed]
    selected = []
    for row, experts, prediction, truth in rows:
        late = float(row['day']) >= 179
        if segment == 'all' or segment == 'early' and not late or segment == 'late' and late or segment == 'F47_late' and late and row['farm'] == 'F47':
            selected.append((row, experts, prediction, truth))
    floor, actual, outside = [], [], 0
    for row, experts, prediction, truth in selected:
        lower, upper = sorted(experts)[0], sorted(experts)[-1]
        distance = lower - truth if truth < lower else truth - upper if truth > upper else 0.0
        floor.append(distance * distance)
        actual.append((prediction - truth) ** 2)
        outside += int(distance > 0)
    n = len(selected)
    values = dict(n=n, days=len({(r['farm'], r['day']) for r, _, _, _ in selected}), outside_n=outside, outside_pct=100 * outside / n, baseline_rmse=math.sqrt(math.fsum(actual) / n), oracle_hull_rmse=math.sqrt(math.fsum(floor) / n), unavoidable_sse_pct=100 * math.fsum(floor) / math.fsum(actual))
    for key, value in values.items():
        assert math.isclose(value, record[key], rel_tol=1e-11, abs_tol=1e-11), (label, segment, key)
    verified.append(dict(target=label, segment=segment, **values))
audit_dir = ROOT / '집/코덱스/analysis/temp_validation_structure_20261003_v1'
audit = json.loads((audit_dir / 'audit.json').read_text(encoding='utf-8'))
with (audit_dir / 'weather_groups.csv').open(encoding='utf-8-sig', newline='') as f:
    groups = list(csv.DictReader(f))
assert len(groups) == audit['days'] == 400
assert len({r['weather_group'] for r in groups}) == audit['weather_groups'] == 122
assert sum(f['validation_weather_shared_train'] for f in audit['folds']) == 291
result = dict(status='PASS', method='csv.DictReader + scalar arithmetic + math.fsum, fresh process', cells=len(verified), weather_days=400, weather_groups=122, shared_weather_days=291, findings=verified, source_hashes={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (tp, ep, HERE / 'diagnose.py', HERE / 'verify.py', audit_dir / 'audit.json', audit_dir / 'weather_groups.csv')})
(HERE / 'verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps({k: result[k] for k in ('status', 'method', 'cells', 'weather_days', 'weather_groups', 'shared_weather_days')}, ensure_ascii=False))
