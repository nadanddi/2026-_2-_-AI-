"""Carry only unchanged feature-audit evidence across a control-flow fix."""
from pathlib import Path
import json, ast, hashlib
H=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
old=H/'n4_run_v1.py';new=H/'n4_run_v2.py'
def functions(path):
    return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(path.read_text(encoding='utf-8')).body if isinstance(n,ast.FunctionDef)}
a,b=functions(old),functions(new)
names=['save','raw_fold','world','cx','folds','audit']
assert all(a[n]==b[n] for n in names)
reg=json.loads((H/'n4_registration_v1.json').read_text(encoding='utf-8'))
assert reg['source_sha']==sha(old)
parent_sha=hashlib.sha256(json.dumps(reg,sort_keys=True).encode()).hexdigest()
report=json.loads((H/'n4_audit_v1.json').read_text(encoding='utf-8'))
assert report['recipe_sha']==parent_sha and report['new_fit']==0
reg['source_sha']=sha(new)
rs=hashlib.sha256(json.dumps(reg,sort_keys=True).encode()).hexdigest()
report.update(recipe_sha=rs,evidence_kind='UNCHANGED_FUNCTION_AST_EVIDENCE_INHERITED_NOT_V2_RERUN',parent_audit_sha=sha(H/'n4_audit_v1.json'),parent_recipe_sha=parent_sha,ast_equal_functions=names,bridge_source_sha=sha(__file__))
for name,obj in [('n4_registration_v2.json',reg),('n4_audit_v2.json',report)]:
    with (H/name).open('x',encoding='utf-8') as f:json.dump(obj,f,ensure_ascii=False,indent=2)
print('N4_AUDIT_BRIDGE',rs,report['cells'],names)
