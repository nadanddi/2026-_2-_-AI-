"""Check saved report arithmetic only. No fit, predictions or score search."""
from pathlib import Path
import csv, json, math, hashlib
from decimal import Decimal, localcontext

ROOT = Path(__file__).resolve().parents[4]
SRC = ROOT / '집/코덱스/analysis/ec_log_partition_mean_20261004_v1'
OUT = Path(__file__).resolve().parent
with (SRC / 'scores_v1.csv').open(encoding='utf-8-sig', newline='') as f:
    scores = list(csv.DictReader(f))
with (SRC / 'segments_v1.csv').open(encoding='utf-8-sig', newline='') as f:
    segments = list(csv.DictReader(f))
result = json.loads((SRC / 'result_v1.json').read_text(encoding='utf-8'))
leaf = json.loads((SRC / 'leaf_diagnostic_v1.json').read_text(encoding='utf-8'))
assert len(scores) == 15
assert sum(int(r['n']) for r in scores) == result['rows'] == 83160
assert result['decision'] == 'REJECT' and result['public_pass'] is False
for r in scores:
    change = 100 * (float(r['candidate']) / float(r['baseline']) - 1)
    assert abs(change - float(r['change_pct'])) < 1e-11
    with localcontext() as ctx:
        ctx.prec = 40
        independent = 100 * (Decimal(r['candidate']) / Decimal(r['baseline']) - 1)
        assert abs(float(independent) - change) < 1e-11

cells = []
for seed in [7, 101, 2024]:
    overall = next(r for r in scores if r['validator'] == 'DIAG10' and int(r['seed']) == seed)
    high = next(r for r in segments if r['segment'] == 'high' and int(r['seed']) == seed)
    ordinary = next(r for r in segments if r['segment'] == 'ordinary' and int(r['seed']) == seed)
    assert int(high['n']) == 744 and int(ordinary['n']) == 7896 and int(overall['n']) == 8640
    assert int(high['days']) == 31 and int(ordinary['days']) == 329
    for col in ['baseline', 'candidate']:
        combined = math.sqrt(math.fsum([int(r['n']) * float(r[col]) ** 2 for r in [high, ordinary]]) / int(overall['n']))
        assert abs(combined - float(overall[col])) < 1e-12
        with localcontext() as ctx:
            ctx.prec = 40
            independent = ((Decimal(high['n']) * Decimal(high[col]) ** 2 + Decimal(ordinary['n']) * Decimal(ordinary[col]) ** 2) / Decimal(overall['n'])).sqrt()
            assert abs(float(independent) - combined) < 1e-12
    change_pct = {}
    for name, row in [('overall', overall), ('high', high), ('ordinary', ordinary)]:
        change_pct[name] = 100 * (float(row['candidate']) / float(row['baseline']) - 1)
        with localcontext() as ctx:
            ctx.prec = 40
            assert abs(float(100 * (Decimal(row['candidate']) / Decimal(row['baseline']) - 1)) - change_pct[name]) < 1e-11
    delta_sse = {name: int(row['n']) * (float(row['candidate']) ** 2 - float(row['baseline']) ** 2) for name, row in [('high', high), ('ordinary', ordinary)]}
    assert math.fsum(delta_sse.values()) < 0
    cells.append(dict(seed=seed, change_pct=change_pct, delta_sse=delta_sse,
                      ordinary_bias_before=float(ordinary['baseline_bias']),
                      ordinary_bias_after=float(ordinary['candidate_bias'])))

stats = leaf['stats']
assert abs(stats['multirow_query_visits'] / stats['query_visits'] - leaf['query_multirow_fraction']) < 1e-15
assert stats['varying_b_query_visits'] / stats['query_visits'] == leaf['query_varying_b_fraction'] == 0
assert result['family'] == 16 and abs(result['alpha'] - .025 / 16) < 1e-15
assert all(v['p_worse'] > result['alpha'] for v in result['bootstrap'].values())
report = dict(status='PASS', scope='Saved aggregate arithmetic and metadata only; no raw-model replay or bootstrap recomputation',
              cells=cells, improved_score_cells=sum(float(r['change_pct']) < 0 for r in scores),
              total_score_cells=len(scores), family=result['family'], threshold=result['alpha'],
              stored_p_worse={k:v['p_worse'] for k,v in result['bootstrap'].items()},
              stored_leaf_diagnostic_scope=leaf['scope'], query_multirow_fraction=leaf['query_multirow_fraction'],
              hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [SRC/'scores_v1.csv', SRC/'segments_v1.csv', SRC/'run.py', SRC/'result_v1.json', SRC/'leaf_diagnostic_v1.json']})
(OUT / 'log_report_check_v1.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k != 'hashes'}, ensure_ascii=False, indent=2))
