"""Add exact result grids and source links to the independent Decimal checker."""
from pathlib import Path
HERE=Path(__file__).resolve().parent
s=(HERE/'crosscheck_DOMAIN24_score_v1.py').read_text(encoding='utf-8')
old="ids = pred['row_ids']"
new=old+'''
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
'''
assert s.count(old)==1
s=s.replace(old,new)
s=s.replace("high = {d for d, values in days.items() if sum(values) / len(values) >= 1}","assert len(days)==60 and all(len(v)==24 for v in days.values())\nhigh = {d for d, values in days.items() if sum(values) / len(values) >= 1}")
s=s.replace("assert len(subsets['고EC']) == result['high_rows']","assert len(block_ids)==8 and sorted(r for block in block_ids for r in block)==sorted(ids)\nassert all(len(v)==4 for v in farm_blocks.values())\nassert len(subsets['고EC']) == result['high_rows'] and len(high)==result['high_days']")
s=s.replace('DOMAIN24_score_independent_crosscheck_v1.json','DOMAIN24_score_independent_crosscheck_v2.json')
s=s.replace("'counts':{'rows':len(ids)","'code_sha256':digest(Path(__file__)), 'cells_checked':sum(len(e['cells']) for e in result['results']),\n    'counts':{'rows':len(ids)")
compile(s,'crosscheck_DOMAIN24_score_v2.py','exec')
out=HERE/'crosscheck_DOMAIN24_score_v2.py';assert not out.exists()
out.write_text(s,encoding='utf-8')
print('Independent Decimal checker v2 exact grids/source links created; not executed')
