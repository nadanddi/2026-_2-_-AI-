"""Actual pure sha/AST checks with two new synthetic JSON files only."""
from pathlib import Path
import ast,json,hashlib,sys
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[4]
H=ROOT/'집/코덱스/analysis/ec_tabdpt_20261004_v1'
OUT=Path(__file__).resolve().parent
v1=ast.parse((H/'run_v1.py').read_text(encoding='utf-8'))
v3=ast.parse((H/'run_v3.py').read_text(encoding='utf-8'))
support=ast.parse((H.parent/'statistical_experiments_20261003_v1/support.py').read_text(encoding='utf-8'))
sha=next(n for n in support.body if isinstance(n,ast.FunctionDef) and n.name=='sha')
scope={'Path':Path,'hashlib':hashlib};exec(compile(ast.Module(body=[sha],type_ignores=[]),'<actual_S_sha>','exec'),scope)
class HashOnly:
    pass
S=HashOnly();S.sha=scope['sha']
main=next(n for n in v3.body if isinstance(n,ast.FunctionDef) and n.name=='main')
guard=next(n for n in ast.walk(main) if isinstance(n,ast.Assert) and 'first_audit_sha256' in ast.unparse(n.test))
code=compile(ast.Module(body=[guard],type_ignores=[]),'<actual_digest_assert>','exec')
files=[OUT/'tabdpt_v3_synthetic_first_digest_a_v1.json',OUT/'tabdpt_v3_synthetic_first_digest_b_v1.json']
for i,path in enumerate(files):
    assert not path.exists()
    path.write_text(json.dumps(dict(status='PASS',signature={'synthetic':'same'},audit_seconds=i)),encoding='utf-8')
saved={'first_audit_sha256':S.sha(files[0])}
exec(code,dict(saved=saved,S=S,first=files[0]))
try:exec(code,dict(saved=saved,S=S,first=files[1]))
except AssertionError:changed_bytes_rejected=True
else:changed_bytes_rejected=False
assert changed_bytes_rejected and S.sha(files[0])!=S.sha(files[1])

def find(tree,name):return next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name==name)
same_nested=[]
for name in ['make','predict']:
    assert ast.dump(find(v1,name),include_attributes=False)==ast.dump(find(v3,name),include_attributes=False)
    same_nested.append(name)
constants={}
for name in ['SEEDS','RAW_ATOL','FINAL_ATOL']:
    def value(tree):
        n=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in n.targets))
        return ast.literal_eval(n.value)
    constants[name]=value(v3);assert constants[name]==value(v1)
def candidate(tree):
    n=next(n for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(ast.unparse(t)=="d['candidate']" for t in n.targets))
    return ast.dump(n.value,include_attributes=False)
assert candidate(v1)==candidate(v3)
record=dict(status='PASS_DIGEST_AND_UNCHANGED_FORMULA',same_first_bytes_accepted=True,changed_first_bytes_rejected=True,
            hashes={p.name:S.sha(p) for p in files},unchanged_nested_functions=same_nested,constants=constants,
            candidate_expression_unchanged=True,real_data_reads=0,weight_reads=0,model_imports=0,model_fits=0,model_predictions=0)
dest=OUT/'tabdpt_v3_digest_check_v1.json';assert not dest.exists()
dest.write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(record,ensure_ascii=False,indent=2))
