"""Independent Decimal loss sums and explicit MT19937 block resampling."""
import csv
import hashlib
import json
import math
import random
from decimal import Decimal, getcontext
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
getcontext().prec = 60
def read(name):
    return json.loads((HERE / name).read_text(encoding='utf-8'))
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

result = read('DOMAIN24_BLK_diagnostic_results_v1.json')
stage = read('DOMAIN24_execution_score_v1.json')
assert stage['returncode'] == 0 and stage['frozen_sources_unchanged']
assert len(result['results']) == 48 and result['cumulative_current_BLK_variant_count'] == 54
gate=read('DOMAIN24_verified_gate_v1.json')
assert gate['whole_pipeline_gate_passed'] is True and gate['adoption_permitted'] is False
assert all(digest(Path(path)) == value for path,value in gate['transitive_sources_sha256'].items())

pred = read('checkpoints/DOMAIN24_BLK_ASSEMBLED_v2/predictions.json')
layout = read('BLK_layout_v2.json')
ids = pred['row_ids']
assert len(ids)==len(set(ids))==1440 and set(ids)==set(layout['query_ids'])
expected={(scope,f'D{i:02d}') for scope in ['BLK_QUERY_ROLE','BLK_RAW_PASS'] for i in range(1,25)}
assert len(result['results'])==len(expected)==48
assert {(e['scope'],e['candidate']) for e in result['results']}==expected
cell_grid={(seed,segment) for seed in [47,1414,6464] for segment in ['전체','일반','고EC','앞','가운데','뒤']}
assert all(len(e['cells'])==18 and {(c['seed'],c['segment']) for c in e['cells']}==cell_grid for e in result['results'])
assert all(len(e['block_MSE_delta_sums'])==8 and all(math.isfinite(v) for v in e['block_MSE_delta_sums']) for e in result['results'])
assert gate['predictions_sha256']==digest(HERE/'checkpoints/DOMAIN24_BLK_ASSEMBLED_v2/predictions.json')
assert gate['code_sha256']==digest(HERE/'verify_domain_BLK_v1.py')
spec=read('DOMAIN24_BLK_scorer_spec_v1.json')
assert result['scorer_spec_sha256']==digest(HERE/'DOMAIN24_BLK_scorer_spec_v1.json')
assert spec['gate_sha256']==digest(HERE/'DOMAIN24_verified_gate_v1.json')
assert spec['code_sha256']==digest(HERE/'score_domain_BLK_v1.py')
assert stage['pipeline_registration_sha256']==digest(HERE/'DOMAIN24_BLK_pipeline_registration_v2.json')
assert spec['predictions_sha256']==gate['predictions_sha256']
tags={f'{scope}_seed{seed}' for scope in ['BLK_QUERY_ROLE','BLK_RAW_PASS'] for seed in [47,1414,6464]}
assert set(pred['baseline'])==set(pred['candidate'])==tags
for tag in tags:
    assert len(pred['baseline'][tag])==1440 and all(math.isfinite(v) for v in pred['baseline'][tag])
    assert set(pred['candidate'][tag])=={f'D{i:02d}' for i in range(1,25)}
    assert all(len(v)==1440 and all(math.isfinite(x) for x in v) for v in pred['candidate'][tag].values())

source = ROOT / '공용/대회자료/정형데이터/참가자_배포/train_y.csv'
assert digest(source) == layout['source_sha256']['train_y.csv']
truth = {}
wanted = set(ids)
with source.open(encoding='utf-8-sig', newline='') as handle:
    for row in csv.DictReader(handle):
        if row['row_id'] in wanted:
            assert row['row_id'] not in truth
            truth[row['row_id']] = Decimal.from_float(float(row['sub_ec']))
assert set(truth) == set(ids)
days = {}
for rid in ids:
    days.setdefault(rid[:7], []).append(truth[rid])
assert len(days)==60 and all(len(v)==24 for v in days.values())
high = {d for d, values in days.items() if sum(values) / len(values) >= 1}
positions = {}
block_ids = []
farm_blocks = {'F13': [], 'F47': []}
for index, block in enumerate(layout['blocks']):
    farm_blocks[block['farm']].append(index)
    current = []
    for day_index, day in enumerate(block['query_days']):
        for hour in range(24):
            rid = f'{block["farm"]}_{day:03d}_{hour:02d}'
            current.append(rid)
            positions[rid] = ['앞', '가운데', '뒤'][3 * day_index // len(block['query_days'])]
    block_ids.append(current)
subsets = {'전체': ids, '일반': [r for r in ids if r[:7] not in high],
           '고EC': [r for r in ids if r[:7] in high]}
subsets.update({p: [r for r in ids if positions[r] == p] for p in ['앞', '가운데', '뒤']})
assert len(block_ids)==8 and sorted(r for block in block_ids for r in block)==sorted(ids)
assert all(len(v)==4 for v in farm_blocks.values())
assert len(subsets['고EC']) == result['high_rows'] and len(high)==result['high_days']
rng = random.Random(2026100702)
samples = []
for _ in range(20000):
    sample = []
    for farm in ['F13', 'F47']:
        for _ in range(4):
            choices = farm_blocks[farm]
            sample.append(choices[rng.randrange(len(choices))])
    samples.append(sample)
checks = []
for entry in result['results']:
    scope, method = entry['scope'], entry['candidate']
    losses, rmse_delta = [], []
    for seed in [47, 1414, 6464]:
        b = dict(zip(ids, map(Decimal.from_float, pred['baseline'][f'{scope}_seed{seed}'])))
        c = dict(zip(ids, map(Decimal.from_float, pred['candidate'][f'{scope}_seed{seed}'][method])))
        delta = {r: (c[r]-truth[r])**2 - (b[r]-truth[r])**2 for r in ids}
        losses.append(delta)
        for cell in [v for v in entry['cells'] if v['seed'] == seed]:
            selected = subsets[cell['segment']]
            assert cell['rows'] == len(selected)
            if not selected:
                assert cell['baseline_RMSE'] is None and cell['candidate_RMSE'] is None
                continue
            br = float((sum((b[r]-truth[r])**2 for r in selected) / len(selected)).sqrt())
            cr = float((sum((c[r]-truth[r])**2 for r in selected) / len(selected)).sqrt())
            assert abs(br-cell['baseline_RMSE']) < 1e-12
            assert abs(cr-cell['candidate_RMSE']) < 1e-12
            if cell['segment'] == '전체':
                rmse_delta.append(cr-br)
    block_sse = [sum(sum(v[r] for v in losses)/3 for r in rows) for rows in block_ids]
    assert max(abs(float(a)-b) for a, b in zip(block_sse, entry['block_MSE_delta_sums'])) < 1e-10
    boots = [float(sum(block_sse[b] for b in draw) / sum(len(block_ids[b]) for b in draw)) for draw in samples]
    p = (1 + sum(v >= 0 for v in boots)) / 20001
    assert p == entry['p_worse']
    ordered = sorted(boots)
    ci = []
    for q in [.025, .975]:
        point = q * (len(ordered)-1)
        low = int(point)
        ci.append(ordered[low] + (ordered[min(low+1, len(ordered)-1)]-ordered[low])*(point-low))
    assert max(abs(a-b) for a,b in zip(ci,entry['individual95_MSE_delta_CI'])) < 1e-12
    assert abs(sum(rmse_delta)/3-entry['mean_seed_delta_RMSE']) < 1e-12
    assert entry['BLK_screen_pass'] == (all(v < 0 for v in rmse_delta) and p < .025/54)
    checks.append({'scope':scope,'method':method,'cell_RMSEs':len(entry['cells'])*2,'p_worse':p,'PASS':True})
out = HERE / 'DOMAIN24_score_independent_crosscheck_v2.json'
assert not out.exists()
out.write_text(json.dumps({'status':'PASS','method':'Decimal60 direct ID loss / explicit randrange farm-block bootstrap',
    'code_sha256':digest(Path(__file__)), 'cells_checked':sum(len(e['cells']) for e in result['results']),
    'counts':{'rows':len(ids),'days':len(days),'high_days':len(high),'high_rows':len(subsets['고EC'])},
    'checks':checks,'result_sha256':digest(HERE/'DOMAIN24_BLK_diagnostic_results_v1.json'),
    'limits':['Same frozen sample and public heldout labels; not new validation data',
              'Checks score arithmetic, not independent model fit or feature legality']},ensure_ascii=False,indent=2),encoding='utf-8')
print('Independent Decimal RMSE/block bootstrap checks PASS for all48 domain variants', flush=True)
