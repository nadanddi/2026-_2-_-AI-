from pathlib import Path
import ast,json,hashlib
H=Path(__file__).resolve().parent;sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
s=(H.parent/'ec_lgb_operation_20261004_v1/verify_full_v3.py').read_text(encoding='utf-8-sig')
s=s.replace('family21','family24').replace('Runner import/fit/predict 없음','Runner import/fit 없음; checkpoint prediction only')
s=s.replace("PREP_FILE='preparation_v3.json'","PREP_FILE='preparation_v3.json'")
for name,value in [('RUN_SHA',sha(H/'run_v3.py')),('PREP_SHA',sha(H/'preparation_v3.json')),('PREREG_SHA','PENDING_ROOT_PIN')]:
 start=s.index(name+"='");end=s.index("'",start+len(name)+2);s=s[:start]+name+"='"+value+s[end:]
s=s.replace(" dp1='40d650c639550c09ef96a6be3aae3ef0f969fa1cbd899c787e7724a7f563e86d',\n",'')
s=s.replace('FAMILY=21','FAMILY=24');start=s.index('NEW=[');end=s.index('CSV_COLUMNS=',start);s=s[:start]+'NEW=[]\n'+s[end:]
start=s.index('def operations(');end=s.index('def scalar_candidate(',start);s=s[:start]+'''def operations(lab):return lab.copy()

def library():
    import os,ctypes,lightgbm.basic as basic
    build=readj(H/'build_result_v3.json');assert build['status']=='PASS_BUILD_ONLY'
    dll=Path(build['dll']);assert sha(dll)==build['dll_sha256']=='25806c5faae08b55142c8d7670be8a88461604865eba353f7094f008e3c94ac8'
    assert sha(H/'overlay_manifest_v1.json')==build['overlay_manifest_sha256']
    overlay=readj(H/'overlay_manifest_v1.json')
    for name,item in overlay['changes'].items():assert sha(OUT/'overlay_v1'/name)==item['overlay_sha256']
    assert sha(OUT/'overlay_v1/src/objective/farmai_leaf_audit.hpp')==overlay['extra_header_sha256']
    tool=OUT/'toolchain_v1/mingw64/bin'
    assert sha(tool/'g++.exe')==build['compiler_sha256'] and sha(tool/'cmake.exe')==build['cmake_sha256']
    assert sha(OUT/'winlibs.zip')=='d5dbafc4a170e762ca6143151ec918fb9e2c72736fb14cd704abebc6bdd5276a'
    assert Path(build['alias']).resolve()==Path(build['alias_resolved_target']).resolve()==OUT.resolve()
    assert readj(H/'synthetic_result_v4.json')['status']=='PASS_SYNTHETIC_ONLY'
    original=Path(basic._LIB._name);assert sha(original)=='7e366d2e49cd061aac3ab21676b2f99b0c7a758dc3e888d4b23812af1b7d301c'
    global DLL_HANDLES
    DLL_HANDLES=[os.add_dll_directory(str(tool)),os.add_dll_directory(str(dll.parent))]
    basic._LIB=ctypes.cdll.LoadLibrary(str(dll));basic._LIB.LGBM_GetLastError.restype=ctypes.c_char_p
    assert Path(basic._LIB._name).resolve()==dll.resolve()
    return dict(dll=str(dll),dll_sha256=sha(dll),build_sha256=sha(H/'build_result_v3.json'),overlay_sha256=sha(H/'overlay_manifest_v1.json'),synthetic_sha256=sha(H/'synthetic_result_v4.json'),cpp_audit_source_sha256=sha(H/'first_cpp_audit_v2.py'))

'''+s[end:]
s=s.replace("'atol','signature'})", "'atol','signature','cpp_audit','cpp_trace_sha256'})")
s=s.replace("    small_error(first['original_lgb_maxdiff'])", "    assert first['cpp_audit']['status']=='PASS' and first['cpp_trace_sha256']==sha(OUT/'first_exact_cpp_trace_v1.ndjson')\n    small_error(first['original_lgb_maxdiff'])")
s=s.replace("keys={'status','signature','csv_sha256','fit_predict_seconds'}", "keys={'status','signature','csv_sha256','fit_predict_seconds','first_audit_sha256','model_sha256'}")
s=s.replace("full_verification_v3.json","full_verification_v1.json").replace('full_scores_v3.csv','full_scores_v1.csv').replace('full_segments_v3.csv','full_segments_v1.csv')
s=s.replace("{'status','family','preparation','cells','score_count'}", "{'status','family','preparation','cells','score_count','compiled_newton_all66_sha256'}")
s=s.replace("    lab,core,wv,folds,outer=S.loadec()", "    lib=library()\n    controls=readj(H/'compiled_newton_all66_v1.json')\n    assert controls['status']=='PASS' and controls['atol']==ATOL and controls['score_count']==0 and len(controls['cells'])==66\n    assert fit['compiled_newton_all66_sha256']==sha(H/'compiled_newton_all66_v1.json')\n    lab,core,wv,folds,outer=S.loadec()")
s=s.replace('dp1=sha(DP1),','').replace('len(cols)==23 and len(set(cols))==23','len(cols)==14 and len(set(cols))==14')
s=s.replace("assert {p.name for p in OUT.glob('*.json')}=={f'{v}_{k}_{s}.json' for v,k,s in expected}","assert {p.name for p in OUT.glob('*.json')}==({f'{v}_{k}_{s}.json' for v,k,s in expected}|{f'{v}_{k}_{s}_newton_control.json' for v,k,s in expected})\n    assert {p.name for p in OUT.glob('*_model.txt')}=={f'{v}_{k}_{s}_model.txt' for v,k,s in expected}")
s=s.replace('sig=dict(validator=v,fold=k,','sig=dict(library=lib,validator=v,fold=k,').replace("('new23',cols)","('new14',cols)")
s=s.replace("sha(first_path) if is_first else None", "sha(first_path)")
s=s.replace("            if is_first:validate_first(first,signature,lab,q)",'''            checkpoint=OUT/f'{v}_{k}_{s}_model.txt';assert meta['model_sha256']==sha(checkpoint)
            control=readj(OUT/f'{v}_{k}_{s}_newton_control.json');npz=OUT/f'{v}_{k}_{s}_newton_control.npz'
            assert control['status']=='PASS' and control['signature']==sig and control['seed']==s and control['npz_sha256']==sha(npz)
            assert control['preparation_sha256']==PREP_SHA and control['preregistration_sha256']==PREREG_SHA
            small_error(control['max_error']);assert control==controls['cells'][i*3+SEEDS.index(s)]
            cached=dict(np.load(npz,allow_pickle=False));assert np.array_equal(cached['row_id'],q.row_id);compare(cached['raw'],r3s[s]['raw_lgb'])
            import lightgbm
            model=lightgbm.Booster(model_file=str(checkpoint));assert model.dump_model()['objective']=='tweedie_exact_leaf rho:1.5 lambda_l2:1'
            if is_first:
                validate_first(first,signature,lab,q)
                from whole_cpp_recheck_v1 import independent_cpp
                cpp_independent=independent_cpp(OUT/'first_exact_cpp_trace_v1.ndjson',model,tr,bs)
''')
s=s.replace("            compare(d.y,q.sub_ec);", "            compare(d.new_lgb_raw,model.predict(q[bs]));del model\n            compare(d.y,q.sub_ec);")
s=s.replace("verification_predict=0", "verification_predict=66,cpp_independent=cpp_independent")
s=s.replace("'Saved first-audit numeric/signature checks only; no model replay'", "'First tree trace independently checked; every model checkpoint predicted without retraining'")
# Synthetic first-record mocks deliberately exercise schema only; no actual trace reads.
s=s.replace("record=dict(status='PASS',original_lgb_maxdiff", "record=dict(cpp_audit={'status':'PASS'},cpp_trace_sha256='MOCK',status='PASS',original_lgb_maxdiff")
s=s.replace("    validate_first(record,signature,q,q);rejected=0", "    original_sha=globals()['sha'];globals()['sha']=lambda p:'MOCK'\n    validate_first(record,signature,q,q);rejected=0")
s=s.replace("    d=q.copy()", "    globals()['sha']=original_sha\n    d=q.copy()")
s=s.replace('synthetic_verify_full_v3.json','synthetic_verify_full_v1.json')
ast.parse(s)
with (H/'verify_full_v1.py').open('x',encoding='utf-8') as f:f.write(s)
print('WHOLE_V1_GENERATED_FIT0_SCORE0_PREREG_PIN_PENDING')
