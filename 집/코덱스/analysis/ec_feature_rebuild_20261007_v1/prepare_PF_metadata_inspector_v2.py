from pathlib import Path
import ast

here=Path(__file__).resolve().parent
source=(here/'inspect_claude_PF_outputs_v1.py').read_text(encoding='utf-8')
source=source.replace("assert set(table)==expected and len(folds)==66 and tm_rows==2664\n        assert dict(counts)=={'DIAG10':8640,'EL1':1104,'P2LOO':1104}","assert set(table)<=expected\n        missing=sorted(expected-set(table))\n        complete=set(table)==expected\n        if complete:\n            assert len(folds)==66 and tm_rows==2664\n            assert dict(counts)=={'DIAG10':8640,'EL1':1104,'P2LOO':1104}")
source=source.replace("'TM_rows':tm_rows,'prediction_columns':columns", "'TM_rows':tm_rows,'complete_expected_metadata':complete,'missing_expected_rows':len(missing),'missing_expected_keys':missing,'prediction_columns':columns")
source=source.replace("for key in expected) for c in shared", "for key in set(frames['PF1'])&set(frames['PF2'])) for c in shared")
source=source.replace("'source_sha256':hashes,'outputs':info", "'source_sha256':hashes,'outputs':info,'shared_keys_compared':len(set(frames['PF1'])&set(frames['PF2']))")
source=source.replace('CLAUDE_PF_outputs_metadata_v1.json','CLAUDE_PF_outputs_metadata_v2.json')
source=source.replace("print('PF1/PF2 metadata: both10848rows/66folds/TM2664; shared members agree; no target scoring',flush=True)","print(json.dumps({'outputs':[{k:v for k,v in item.items() if k!='missing_expected_keys'} for item in info],'shared_keys_compared':result['shared_keys_compared'],'shared_max_difference':max(reused.values()),'target_scoring':False},ensure_ascii=False),flush=True)")
ast.parse(source)
out=here/'inspect_claude_PF_outputs_v2.py'
assert not out.exists()
out.write_text(source,encoding='utf-8')
print('Created v2 partial-aware inspector; v1 preserved')
