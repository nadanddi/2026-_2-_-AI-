"""Reject inherited-pin overwrite and close no-target audit flags before raw fit."""
from pathlib import Path
import ast
HERE=Path(__file__).resolve().parent
source=(HERE/'register_original_raw_fit_v4.py').read_text(encoding='utf-8')
source=source.replace("    assert runtime['model_fits']==runtime['query_predictions']==0",
    "    assert runtime['model_fits']==runtime['query_predictions']==0 and runtime['target_values_read'] is False")
source=source.replace("    assert statscheck['blocks_checked']==87 and statscheck['row_membership_checked']==8640",
    "    assert statscheck['blocks_checked']==87 and statscheck['row_membership_checked']==8640\n    assert statscheck['target_values_read'] is False and statscheck['model_fit'] is False")
source=source.replace("    assert integrity['full_model_score_gate'] is False and integrity['heldout_truth_loaded'] is False",
    "    assert integrity['full_model_score_gate'] is False and integrity['heldout_truth_loaded'] is False\n    assert integrity['python_actual']==runtime['environment']['python']")
source=source.replace("    assert len(negative['cases'])==10 and all(t['PASS'] for t in negative['cases'])",
    "    assert len(negative['cases'])==10 and all(t['PASS'] for t in negative['cases'])\n    assert negative['model_fit'] is False and negative['heldout_truth_loaded'] is False and negative['full_model_score_gate'] is False")
source=source.replace("'register_original_raw_fit_v4.py','upgrade_original_registrar_v4.py',",
    "'register_original_raw_fit_v4.py','upgrade_original_registrar_v4.py','register_original_raw_fit_v5.py','upgrade_original_registrar_v5.py',")
old="        p=HERE/name;sources[str(p.resolve())]=sha(p)"
new="        p=HERE/name;key=str(p.resolve());value=sha(p)\n        assert key not in sources or sources[key]==value, 'Conflicting inherited source pin: '+key\n        sources[key]=value"
assert source.count(old)==1;source=source.replace(old,new)
ast.parse(source)
with (HERE/'register_original_raw_fit_v5.py').open('x',encoding='utf-8') as h:h.write(source)
print('Registrar5 source rejects inherited-pin overwrite/no-target flag errors; registration/fit pending')
