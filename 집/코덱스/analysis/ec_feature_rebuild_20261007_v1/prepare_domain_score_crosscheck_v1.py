"""Create a new independent arithmetic checker without modifying prior checks."""
from pathlib import Path
HERE=Path(__file__).resolve().parent
s=(HERE/'crosscheck_BLK_score_v1.py').read_text(encoding='utf-8')
replacements={
    'BLK_diagnostic_results_v1.json':'DOMAIN24_BLK_diagnostic_results_v1.json',
    'BLK_execution_score_v1.json':'DOMAIN24_execution_score_v1.json',
    'checkpoints/BLK_ASSEMBLED_REFONLY_v1/predictions.json':'checkpoints/DOMAIN24_BLK_ASSEMBLED_v2/predictions.json',
    "entry['method']":"entry['candidate']",
    '.025/6':'.025/54',
    'BLK_score_independent_crosscheck_v1.json':'DOMAIN24_score_independent_crosscheck_v1.json',
    'all six variants':'all48 domain variants',
}
for old,new in replacements.items():
    assert old in s
    s=s.replace(old,new)
old="assert stage['returncode'] == 0 and stage['frozen_sources_unchanged']"
new=old+"""
assert len(result['results']) == 48 and result['cumulative_current_BLK_variant_count'] == 54
gate=read('DOMAIN24_verified_gate_v1.json')
assert gate['whole_pipeline_gate_passed'] is True and gate['adoption_permitted'] is False
assert all(digest(Path(path)) == value for path,value in gate['transitive_sources_sha256'].items())
"""
assert old in s
s=s.replace(old,new).replace('truth = {}','truth = {}\nwanted = set(ids)').replace('in set(ids):','in wanted:')
compile(s,'crosscheck_DOMAIN24_score_v1.py','exec')
out=HERE/'crosscheck_DOMAIN24_score_v1.py'
assert not out.exists()
out.write_text(s,encoding='utf-8')
print('DOMAIN24 Decimal checker created; no score executed',flush=True)
