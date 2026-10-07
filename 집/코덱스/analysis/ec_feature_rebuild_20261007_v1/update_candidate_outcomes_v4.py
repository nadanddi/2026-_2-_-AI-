import csv
import json
from pathlib import Path
HERE = Path(__file__).resolve().parent
with (HERE/'feature_candidates_v3.csv').open(encoding='utf-8-sig',newline='') as handle:
    reader = csv.DictReader(handle)
    fields = reader.fieldnames
    rows = list(reader)
result = json.loads((HERE/'BLK_diagnostic_results_v1.json').read_text(encoding='utf-8'))
lookup={(r['scope'],r['method']):r for r in result['results']}
changed=0
for row in rows:
    if not row['candidate_id'].startswith('BLK_'):
        continue
    params=json.loads(row['parameters'])
    score=lookup[params['baseline_scope'],row['family']]
    assert not score['BLK_screen_pass']
    row['status']='SCREEN_FAIL_NO_ADOPTION' if row['family']!='CHAIN_PREFIX_GUARD' else 'INACTIVE_BASELINE_IDENTICAL_NO_CHAIN_REJECTION'
    row['performance']=json.dumps({'BLK_mean_seed_delta_RMSE':score['mean_seed_delta_RMSE'],
        'p_worse':score['p_worse'],'reference':'BLK_diagnostic_results_v1.json',
        'baseline':'CPU_REFERENCE_ONLY_CACHED_POLICY_CHANGED',
        'original_validators_tested':False,'catalog':'6.375'},ensure_ascii=False)
    changed+=1
assert changed==6 and len(rows)==196
out=HERE/'feature_candidates_v4.csv'
assert not out.exists()
with out.open('w',encoding='utf-8-sig',newline='') as handle:
    writer=csv.DictWriter(handle,fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
print('Candidate registry v4 saved: six actual outcomes, 190 other candidates unchanged')
