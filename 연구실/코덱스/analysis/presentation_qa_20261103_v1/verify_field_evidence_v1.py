"""Read-only evidence audit; no model fitting, prediction, or source changes."""
import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
DATA = ROOT / '공용/대회자료/정형데이터/참가자_배포'
WEATHER = ['out_temp', 'out_hum', 'out_rad', 'out_wspd']
records = {}
source_hashes = {}
for name in ['train_X.csv', 'test_X.csv']:
    path = DATA / name
    source_hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    with path.open(encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f):
            farm, day, hour = row['row_id'].split('_')
            if farm not in ['F13', 'F47']:
                continue
            key = (farm, int(day))
            rec = records.setdefault(key, {'split': name, 'weather': {}})
            rec['weather'][int(hour)] = [float(row[c]) if row[c] else math.nan for c in WEATHER]
labels = {}
path = DATA / 'train_y.csv'
source_hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
with path.open(encoding='utf-8-sig', newline='') as f:
    for row in csv.DictReader(f):
        farm, day, hour = row['row_id'].split('_')
        if farm not in ['F13', 'F47']:
            continue
        labels.setdefault((farm, int(day)), {})[int(hour)] = [float(row[c]) if row[c] else math.nan for c in ['sub_ec', 'sub_temp']]
keys = sorted(k for k, r in records.items() if r['split'] == 'train_X.csv')
assert len(keys) == 400 and len(records) == 460
for k in keys:
    assert all(math.isfinite(v) for h in [0, 1, 22, 23] for v in labels[k][h])
sd = [statistics.pstdev([records[k]['weather'][h][c] for k in keys for h in range(24) if math.isfinite(records[k]['weather'][h][c])]) for c in range(4)]
def edge(table, key, end):
    a, b = (23, 22) if end else (0, 1)
    return [1.5*x - .5*y for x, y in zip(table[key][a], table[key][b])]
weather = {k: r['weather'] for k, r in records.items()}
rows = []
for target in keys:
    before = edge(labels, target, False)
    costs = []
    for prev in keys:
        if prev == target:
            continue
        after = edge(labels, prev, True)
        ce, ct = [abs(x-y) for x, y in zip(after, before)]
        costs.append((ce/.01 + ct/.3, prev, ce, ct))
    costs.sort()
    (c1, prev, ce, ct), (c2, _, _, _) = costs[:2]
    oj = sum(w*abs(x-y)/s for w, x, y, s in zip([1,1,.3,.5], edge(weather, prev, True), edge(weather, target, False), sd))
    rows.append({'target': f'{target[0]}_{target[1]:03}', 'prev': f'{prev[0]}_{prev[1]:03}', 'c1': c1, 'c2': c2, 'ec_gap': ce, 'temp_gap': ct, 'oj': oj, 'cert_initial': c1 <= 1 and c2-c1 >= 1 and oj <= 1.2, 'cross_id': prev[0] != target[0]})
dup = Counter(r['prev'] for r in rows if r['cert_initial'])
cert = [r for r in rows if r['cert_initial'] and dup[r['prev']] == 1]
cross = [r for r in cert if r['cross_id']]
groups = defaultdict(list)
for key, rec in sorted(records.items()):
    signature = tuple('nan' if not math.isfinite(v) else round(v, 1) for h in range(24) for v in rec['weather'][h])
    groups[signature].append(f'{key[0]}_{key[1]:03}')
table_path = ROOT / '집/클로드/research/pch4_v1_table.csv'
with table_path.open(encoding='utf-8-sig', newline='') as f:
    table = list(csv.DictReader(f))
ratios = {arm: [float(r[arm])/float(r['base']) for r in table] for arm in ['chainF', 'chain27']}
result = {'source_hashes': source_hashes, 'records': len(records), 'train_records': len(keys), 'cert_links': len(cert), 'cross_id_links': len(cross), 'cross_id_fraction': len(cross)/len(cert), 'directions': dict(Counter(r['prev'][:3]+'->'+r['target'][:3] for r in cert)), 'cross_id_examples': sorted(cross, key=lambda r:r['c1'])[:6], 'rounded_weather_groups': len(groups), 'weather_group_size_counts': dict(Counter(map(len, groups.values()))), 'weather_group_examples': sorted(groups.values(), key=lambda g: (-len(g),g))[:4], 'pch4_table_rows': len(table), 'pch4': {arm: {'improved_farms': sum(x<1 for x in vals), 'median_rmse_ratio': statistics.median(vals)} for arm, vals in ratios.items()}, 'scope': 'Recomputed cert links from original CSV using saved t1/t3 criteria; recomputed PCH4 summary from stored per-farm table, not model refitting.'}
assert len(cert) == 190 and len(cross) == 40
assert len(table) == 49 and result['pch4']['chainF']['improved_farms'] == 48
with (OUT/'field_evidence_recomputed_v1.json').open('x', encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False, indent=2)
with (OUT/'cross_id_cert_links_v1.csv').open('x', encoding='utf-8-sig', newline='') as f:
    writer=csv.DictWriter(f, fieldnames=list(cross[0])); writer.writeheader(); writer.writerows(cross)
print(json.dumps(result, ensure_ascii=False, indent=2))
