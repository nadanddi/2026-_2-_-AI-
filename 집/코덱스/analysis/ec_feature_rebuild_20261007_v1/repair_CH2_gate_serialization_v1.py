from pathlib import Path
import ast
here=Path(__file__).resolve().parent
source=(here/'verify_ch2_BLK_v2.py').read_text(encoding='utf-8')
assert "assert diag==pred['diagnostics'][method][i]" in source
source=source.replace("assert diag==pred['diagnostics'][method][i]","assert json.loads(json.dumps(diag,allow_nan=False))==pred['diagnostics'][method][i]")
source=source.replace('CH2_BLK_verified_gate_v2.json','CH2_BLK_verified_gate_v3.json')
ast.parse(source);out=here/'verify_ch2_BLK_v3.py';assert not out.exists();out.write_text(source,encoding='utf-8')
score=(here/'score_ch2_BLK_v1.py').read_text(encoding='utf-8').replace('CH2_BLK_verified_gate_v2.json','CH2_BLK_verified_gate_v3.json').replace('verify_ch2_BLK_v2.py','verify_ch2_BLK_v3.py').replace('CH2_BLK_score_registration_v1.json','CH2_BLK_score_registration_v2.json')
ast.parse(score);out=here/'score_ch2_BLK_v2.py';assert not out.exists();out.write_text(score,encoding='utf-8')
registration=(here/'register_ch2_scorer_v1.py').read_text(encoding='utf-8').replace('score_ch2_BLK_v1.py','score_ch2_BLK_v2.py').replace('verify_ch2_BLK_v2.py','verify_ch2_BLK_v3.py').replace('CH2_BLK_score_registration_v1.json','CH2_BLK_score_registration_v2.json')
registration=registration.replace("'created_before_inference':not (HERE/'checkpoints/CH2_BLK_v1/predictions.json').exists()", "'created_before_inference':False,'repair_scope':'JSON tuple/list equivalence only; original statistics/method/predictions unchanged; before truth score'")
ast.parse(registration);out=here/'register_ch2_scorer_v2.py';assert not out.exists();out.write_text(registration,encoding='utf-8')
checker=(here/'crosscheck_ch2_BLK_score_v1.py').read_text(encoding='utf-8').replace('CH2_BLK_verified_gate_v2.json','CH2_BLK_verified_gate_v3.json').replace('score_ch2_BLK_v1.py','score_ch2_BLK_v2.py').replace('CH2_BLK_score_registration_v1.json','CH2_BLK_score_registration_v2.json')
ast.parse(checker);out=here/'crosscheck_ch2_BLK_score_v2.py';assert not out.exists();out.write_text(checker,encoding='utf-8')
print('Serialization-only gate3/score2/checker2 created; original prediction files and statistics preserved')
