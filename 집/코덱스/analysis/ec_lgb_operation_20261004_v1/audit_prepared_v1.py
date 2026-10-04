"""Read-only independent saved-manifest audit; no models or prediction scores."""
from pathlib import Path
import json, hashlib, ast, sys
sys.dont_write_bytecode = True
H=Path(__file__).resolve().parent
ROOT=H.parents[3]
sha=lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
p=H/'preparation_v3.json'
a=json.loads(p.read_text(encoding='utf-8'))
assert sha(H/'run_v3.py')=='03f9f6030fe3fbe84625b1c9dfef592ba6c265a87016ef1a61ac7cbae2f7e5a4'
assert sha(p)=='6a5bf8b1f30b73d11d63cad8b52a8f2c61aa5b4299355bd825b9232c9aebdea2'
assert a['status']=='PASS' and a['family']==21 and a['alpha']==.025/21
assert [a[k] for k in ['fit_count','predict_count','score_count']]==[0,0,0]
expected=[(v,k) for v,n in [('DIAG10',10),('A',5),('B',5),('EXT10',1),('EXT12',1)] for k in range(n)]
assert [(m['validator'],m['fold']) for m in a['manifest']]==expected
assert len(a['original_r3_guard'])==66
checked=0
for m in a['manifest']:
    assert m['runtime']==a['runtime'] and m['dependencies']==a['dependencies'] and m['inputs']==a['inputs']
    assert len(m['old_lgb_features'])==14 and len(m['new_lgb_features'])==23
    assert m['new_lgb_features'][:14]==m['old_lgb_features']
    assert 'day' not in m['new_lgb_features'] and m['old_lgb_features'][-1]=='season'
    for relative,digest in m['cache_hashes'].items():
        assert sha(ROOT/relative)==digest,relative
        checked+=1
tree=ast.parse((H/'run_v3.py').read_text(encoding='utf-8-sig'))
main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in ['rmse','bootstrap'] for n in ast.walk(main))
out=dict(status='PASS_SAVED_PREPARATION_ONLY',folds=22,original_guard_records=66,cache_sha_checks=checked,fit=0,predict=0,score=0,limitations=['Does not replay features or model fits; full verifier and first actual audit still required.'])
with (H/'independent_preparation_v1.json').open('x',encoding='utf-8') as f:json.dump(out,f,indent=2)
print(json.dumps(out))
