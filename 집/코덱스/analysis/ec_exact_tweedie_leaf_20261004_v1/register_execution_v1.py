"""Root pins and final launch inventory. No fit/predict/scoring."""
from pathlib import Path
import sys,json,hashlib
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;ROOT=H.parents[3];OUT=ROOT/'집/코덱스/local'/H.name
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
expected={'run_v3.py':'74246a838fa27e2f83e5cbb6a50f705bf0669f0a3ae0f578b9a6cd1aaf4eea63',
 'preparation_v3.json':'9f25648fcdda6ee06c3414a91edba15803df5ee1f1e5e2e3cf3e7cdeb9f7ea30',
 'preregistration_v1.md':'54e75b181e9054eb479991589dd8b84cc6317b404ae88f24ee57e6a34a7311ae',
 'verify_full_v3.py':'4093283b32509ea848d2ac43fed00e91ed191e500527ab35644cb5ffd991357e',
 'whole_cpp_recheck_v1.py':'c38232636990570a125d5c6e0bfdda91cc7552d18914be99afca5e4ad3408fd4',
 'first_cpp_audit_v2.py':'fd9998282b2b2ad1e3d54c26ffac1d273d7d6eb2c4795d16da00f3f3cdcee9a4'}
for n,h in expected.items():assert sha(H/n)==h,n
assert not (H/'fit_audit_v1.json').exists() and not (H/'compiled_newton_all66_v1.json').exists()
assert not list(OUT.glob('*_pred.csv')) and not list(OUT.glob('*_model.txt')) and not list(OUT.glob('*_newton_control.npz'))
prep=json.loads((H/'preparation_v3.json').read_text(encoding='utf-8'))
assert prep['family']==24 and prep['alpha']==.025/24 and prep['fit_count']==prep['predict_count']==prep['score_count']==0
assert len(prep['manifest'])==22 and len(prep['original_r3_guard'])==66
checks=0
for m in prep['manifest']:
    assert len(m['old_lgb_features'])==len(m['new_lgb_features'])==14 and m['old_lgb_features']==m['new_lgb_features']
    assert not set(m['new_lgb_features'])&{'in_rad','act_side','act_valve','act_cool','act_pump'}
    for rel,h in m['cache_hashes'].items():assert sha(ROOT/rel)==h;checks+=1
    assert sha(Path(m['library']['dll']))==m['library']['dll_sha256']=='25806c5faae08b55142c8d7670be8a88461604865eba353f7094f008e3c94ac8'
names=list(expected)+['register_execution_v1.py','synthetic_result_v4.json','synthetic_verify_full_v3.json','synthetic_model_gate_v1.json','synthetic_cpp_whole_v1.json','independent_bootstrap_critic_result_v1.json','math_review_result_v1.json','overlay_manifest_v1.json','build_result_v3.json']
for n in ['synthetic_result_v4.json','synthetic_verify_full_v3.json','synthetic_model_gate_v1.json','synthetic_cpp_whole_v1.json']:
    assert json.loads((H/n).read_text())['status']=='PASS_SYNTHETIC_ONLY',n
record=dict(status='REGISTERED_BEFORE_ACTUAL_FIT',files={str((H/n).relative_to(ROOT)):sha(H/n) for n in names},cache_hash_checks=checks,fit_count=0,predict_count=0,score_count=0,first_control='all66 compiled Newton reproduction must pass before candidate fit',tree_budget='requested800; positive actual trees<=800; no-split terminal trace can cause audit rejection, no post-fit loosening')
with (H/'registered_execution_v1.json').open('x',encoding='utf-8') as f:json.dump(record,f,ensure_ascii=False,indent=2)
print('REGISTERED_BEFORE_ACTUAL_FIT',checks)
