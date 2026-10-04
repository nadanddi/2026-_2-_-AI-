"""Re-sum fixed public prediction/shift CSVs only. No fitting or tuning."""
from pathlib import Path
import csv, json, math, hashlib
from collections import defaultdict
from decimal import Decimal, localcontext

ROOT = Path(__file__).resolve().parents[4]
H = ROOT / '집/코덱스/analysis/ec_matched_inner_calibration_20261003_v1'
SRC = ROOT / '집/코덱스/local/ec_matched_inner_calibration_20261003_v1'
OUT = Path(__file__).resolve().parent

def table(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def rowkey(r): return r['validator'], int(r['fold']), int(r['seed']), r['row_id']
def avg(values):
    values = list(values)
    return math.fsum(values) / len(values)

report = json.loads((H / 'full_verification_v1.json').read_text(encoding='utf-8'))
scores = {(r['validator'], int(r['seed'])): r for r in table(H / 'full_scores_v1.csv')}
segments = {(int(r['seed']), r['segment']): r for r in table(H / 'full_segments_v1.csv')}
groups = defaultdict(list); hashes = {}; cells = {}; scalar_maxdiff = 0.
clip_count = 0
for cell in report['split_audit']:
    v, k = cell['validator'], int(cell['fold'])
    assert v in ['DIAG10', 'A', 'B', 'EXT10', 'EXT12']
    for seed in [7, 101, 2024]:
        path = SRC / f'{v}_{k}_{seed}_pred.csv'
        rows = table(path); hashes[path.name] = sha(path)
        assert rows
        for r in rows:
            key = rowkey(r)
            assert key[:3] == (v, k, seed) and key not in cells
            cells[key] = r
            for alias, col in [('yy', 'y'), ('bb', 'baseline'), ('cc', 'candidate')]:
                r[alias] = float(r[col]); assert math.isfinite(r[alias])
            lo, hi, c = float(r['clip_lo']), float(r['clip_hi']), float(r['correction'])
            assert math.isfinite(c) and lo <= r['bb'] <= hi and lo <= r['cc'] <= hi
            reconstructed = min(hi, max(lo, r['bb'] + c))
            scalar_maxdiff = max(scalar_maxdiff, abs(reconstructed - r['cc']))
            clip_count += not lo <= r['bb'] + c <= hi
        groups[v, seed].extend(rows)
assert len(hashes) == 66 and len(cells) == 83160 and scalar_maxdiff < 1e-12

# Independently compare all saved fold files to the combined public OOF.
combined = table(SRC / 'oof.csv'); assert len(combined) == 83160
combined_seen = set(); combined_maxdiff = 0.
for r in combined:
    key = rowkey(r); assert key in cells and key not in combined_seen
    combined_seen.add(key); source = cells[key]
    for col in ['farm', 'day', 'hour']:
        assert r[col] == source[col]
    for col in ['y', 'baseline', 'candidate', 'correction', 'clip_lo', 'clip_hi']:
        combined_maxdiff = max(combined_maxdiff, abs(float(r[col]) - float(source[col])))
assert combined_seen == set(cells) and combined_maxdiff < 1e-12

def metrics(rows):
    n = len(rows); result = {}
    for alias, col in [('baseline', 'bb'), ('candidate', 'cc')]:
        e = [r[col] - r['yy'] for r in rows]
        sse = math.fsum(x*x for x in e)
        result[alias] = math.sqrt(sse/n)
        result[alias+'_bias'] = math.fsum(e)/n
        result[alias+'_sse'] = sse
        result[alias+'_error_variance'] = sse/n-result[alias+'_bias']**2
    result['change_pct'] = 100*(result['candidate']/result['baseline']-1)
    result['delta_sse'] = result['candidate_sse']-result['baseline_sse']
    return result

score_checks = []; decimal_maxdiff = 0.
for (v, seed), rows in sorted(groups.items()):
    expected = scores[v, seed]; actual = metrics(rows)
    assert len(rows) == int(expected['n'])
    for col in ['baseline', 'candidate', 'change_pct']:
        assert abs(actual[col]-float(expected[col])) < 1e-10
    # A second numerical route uses exact Decimal representations of float inputs.
    with localcontext() as ctx:
        ctx.prec = 45
        for col, alias in [('bb', 'baseline'), ('cc', 'candidate')]:
            sq = sum(((Decimal(r[col])-Decimal(r['yy']))**2 for r in rows), Decimal(0))
            dec_rmse = float((sq/Decimal(len(rows))).sqrt())
            decimal_maxdiff = max(decimal_maxdiff, abs(dec_rmse-actual[alias]))
    score_checks.append(dict(validator=v, seed=seed, n=len(rows), **actual))
assert decimal_maxdiff < 1e-12

def day_decomposition(rows):
    days = defaultdict(list)
    for r in rows: days[r['farm'], int(r['day'])].append(r)
    result = {}
    for alias, col in [('baseline', 'bb'), ('candidate', 'cc')]:
        level = []; shape = []
        for rr in days.values():
            e = [r[col]-r['yy'] for r in rr]; mean = avg(e)
            level.append(len(e)*mean**2)
            shape.append(math.fsum((x-mean)**2 for x in e))
        result[alias+'_daily_level_sse'] = math.fsum(level)
        result[alias+'_intraday_shape_sse'] = math.fsum(shape)
        total = metrics(rows)[alias+'_sse']
        assert abs(result[alias+'_daily_level_sse']+result[alias+'_intraday_shape_sse']-total) < 1e-9
    for component in ['daily_level_sse', 'intraday_shape_sse']:
        result['delta_'+component] = result['candidate_'+component]-result['baseline_'+component]
    return result

segment_checks = []; tradeoffs = []
for seed in [7, 101, 2024]:
    rows = groups['DIAG10', seed]
    assert len(rows) == len({r['row_id'] for r in rows}) == 8640
    days = defaultdict(list)
    for r in rows: days[r['farm'], int(r['day'])].append(r['yy'])
    assert len(days) == 360 and all(len(y) == 24 for y in days.values())
    high = {k for k, y in days.items() if avg(y) >= 1}
    assert len(high) == 31
    selectors = {
        'high': lambda r: (r['farm'], int(r['day'])) in high,
        'ordinary': lambda r: (r['farm'], int(r['day'])) not in high,
        'late': lambda r: int(r['day']) >= 179,
        'F13': lambda r: r['farm'] == 'F13',
        'F47': lambda r: r['farm'] == 'F47',
        'hour0': lambda r: int(r['hour']) == 0,
    }
    current = {}
    for name, select in selectors.items():
        rr = [r for r in rows if select(r)]; expected = segments[seed, name]
        actual = metrics(rr); days_n = len({(r['farm'], int(r['day'])) for r in rr})
        assert len(rr) == int(expected['n']) and days_n == int(expected['days'])
        for col in ['baseline', 'candidate', 'baseline_bias', 'candidate_bias']:
            assert abs(actual[col]-float(expected[col])) < 1e-12
        entry = dict(seed=seed, segment=name, n=len(rr), days=days_n, **actual,
                     **day_decomposition(rr))
        segment_checks.append(entry); current[name] = entry
    total = metrics(rows)['delta_sse']
    assert abs(current['high']['delta_sse']+current['ordinary']['delta_sse']-total) < 1e-10
    tradeoffs.append(dict(seed=seed, high_sse_loss=current['high']['delta_sse'],
                          ordinary_sse_gain=-current['ordinary']['delta_sse'],
                          total_sse_loss=total,
                          high_loss_to_ordinary_gain=current['high']['delta_sse']/-current['ordinary']['delta_sse']))

# Descriptive transport diagnostic: re-sum the parent's fixed CSV, not a new gate.
shift_path = SRC / 'full_shift_records_v1.csv'
shift = table(shift_path); assert len(shift) == 31236
shift_seen = set(); shift_groups = defaultdict(list)
for r in shift:
    key = r['validator'], int(r['fold']), int(r['seed']), int(r['hour']), r['domain'], r['farm'], int(r['day'])
    assert key not in shift_seen; shift_seen.add(key)
    x, y, needed, c = [float(r[col]) for col in ['x', 'yday', 'required_shift', 'learned_correction']]
    assert all(math.isfinite(z) for z in [x, y, needed, c])
    assert abs(needed-(y-x)) < 1e-12
    assert (r['high'] == 'True') == (y >= 1)
    r.update(xx=x, yy=y, needed=needed, correction=c)
    shift_groups[r['validator'], int(r['seed']), int(r['hour']), r['domain'], r['prediction_bin']].append(r)
summary = table(H / 'full_shift_summary_v1.csv'); assert len(summary) == len(shift_groups)
for r in summary:
    key = r['validator'], int(r['seed']), int(r['hour']), r['domain'], r['prediction_bin']
    rr = shift_groups[key]; assert len(rr) == int(r['n'])
    assert sum(z['yy'] >= 1 for z in rr) == int(r['high_n'])
    for col, source in [('prediction_mean', 'xx'), ('label_mean', 'yy'), ('required_shift', 'needed'), ('learned_correction', 'correction')]:
        assert abs(avg(z[source] for z in rr)-float(r[col])) < 1e-12
shift_report = json.loads((H/'full_shift_diagnostic_v1.json').read_text(encoding='utf-8'))
shift_checks = []
for expected in shift_report['selected_high_prediction']:
    seed, h, domain = expected['seed'], expected['hour'], expected['domain']
    rr = [r for r in shift if r['validator'] == 'DIAG10' and int(r['seed']) == seed
          and int(r['hour']) == h and r['domain'] == domain and r['xx'] >= .9]
    actual = dict(seed=seed, hour=h, domain=domain, occurrences=len(rr),
                  unique_days=len({(r['farm'], int(r['day'])) for r in rr}),
                  prediction_mean=avg(r['xx'] for r in rr), label_mean=avg(r['yy'] for r in rr),
                  required_shift=avg(r['needed'] for r in rr),
                  learned_correction=avg(r['correction'] for r in rr),
                  high_fraction=avg(r['yy'] >= 1 for r in rr))
    for col, value in actual.items():
        if isinstance(value, float): assert abs(value-expected[col]) < 1e-12
        else: assert value == expected[col]
    shift_checks.append(actual)

record = dict(status='PASS', scope='Fixed completed public OOF and descriptive shift CSV only; no fit, target-file access, tuning or new adoption rule',
              public_occurrence_rows=len(cells), unique_diag_rows=8640, files=len(hashes),
              scalar_maxdiff=scalar_maxdiff, scalar_clip_count=clip_count,
              combined_oof_maxdiff=combined_maxdiff, decimal_rmse_maxdiff=decimal_maxdiff,
              score_checks=score_checks, segment_checks=segment_checks, tradeoffs=tradeoffs,
              shift_record_rows=len(shift), shift_summary_cells=len(summary), shift_checks=shift_checks,
              saved_verifier_checks_not_rerun=report['checks'],
              saved_coefficient_recalculations_not_rerun=len(report['inner_audit']),
              hashes=hashes, source_sha256=sha(H/'run.py'),
              shift_csv_sha256=sha(shift_path))
path = OUT/'matched_oof_verification_v1.json'; assert not path.exists()
path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps({k:v for k,v in record.items() if k in ['status', 'public_occurrence_rows', 'unique_diag_rows', 'files', 'scalar_maxdiff', 'scalar_clip_count', 'combined_oof_maxdiff', 'decimal_rmse_maxdiff', 'tradeoffs', 'shift_checks']}, ensure_ascii=False, indent=2))
