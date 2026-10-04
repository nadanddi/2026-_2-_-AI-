"""Generate this one fixed experiment from preserved, reviewed family21 guards."""
from pathlib import Path
import ast
H=Path(__file__).resolve().parent
p=H.parent/'ec_lgb_operation_20261004_v1/run_v3.py'
s=p.read_text(encoding='utf-8-sig')
s=s.replace('family21 LGB_OPERATION_ONLY: one Tweedie member','family24 EXACT_TWEEDIE_LEAF: one Tweedie member')
start=s.index('SOURCE = ROOT/');end=s.index('FOLD_KEYS =',start)
s=s[:start]+"GUARD_SHA = 'e5398b7c29eae32a4efc75381f7478cb1937e28bb4e61db1c1989016d732d9d9'\nCORE_SHA = '057ff4d8da6f3af29105b049251498f79f7fcd6b88d980a8222e5629eb5ce3b2'\nNEW=[]\n"+s[end:]
s=s.replace('FAMILY=21','FAMILY=24')
start=s.index('def operations(');end=s.index('def preflight(',start)
s=s[:start]+'''def operations(frame):return frame.copy()

BUILD=json.loads((H/'build_result_v3.json').read_text(encoding='utf-8'))
DLL=Path(BUILD['dll'])
def library_guard():
    assert BUILD['status']=='PASS_BUILD_ONLY' and S.sha(DLL)==BUILD['dll_sha256']
    overlay=json.loads((H/'overlay_manifest_v1.json').read_text(encoding='utf-8'))
    assert S.sha(H/'overlay_manifest_v1.json')==BUILD['overlay_manifest_sha256']
    source=OUT/'overlay_v1'
    for name,item in overlay['changes'].items():assert S.sha(source/name)==item['overlay_sha256']
    assert S.sha(source/'src/objective/farmai_leaf_audit.hpp')==overlay['extra_header_sha256']
    return dict(dll=str(DLL),dll_sha256=S.sha(DLL),build_sha256=S.sha(H/'build_result_v3.json'),overlay_sha256=S.sha(H/'overlay_manifest_v1.json'),synthetic_sha256=S.sha(H/'synthetic_result_v4.json'))
def bind_library():
    import ctypes,lightgbm.basic as basic
    global DLL_HANDLES
    DLL_HANDLES=[os.add_dll_directory(str(OUT/'toolchain_v1/mingw64/bin')),os.add_dll_directory(str(DLL.parent))]
    original=Path(basic._LIB._name)
    assert S.sha(original)=='7e366d2e49cd061aac3ab21676b2f99b0c7a758dc3e888d4b23812af1b7d301c'
    basic._LIB=ctypes.cdll.LoadLibrary(str(DLL));basic._LIB.LGBM_GetLastError.restype=ctypes.c_char_p
    assert Path(basic._LIB._name).resolve()==DLL.resolve()
    return original

'''+s[end:]
s=s.replace("len(cols)==23","len(cols)==14").replace(',dp1=SOURCE_SHA','')
s=s.replace("signature=dict(validator=v,fold=k,", "signature=dict(library=library_guard(),validator=v,fold=k,")
s=s.replace("('new23',cols)","('new14',cols)")
s=s.replace("original_guard=H.parent", "original_guard=H.parent").replace(',dp1=SOURCE','')
s=s.replace("    assert runtime_core()==signature['runtime']", "    assert library_guard()==signature['library']\n    assert runtime_core()==signature['runtime']")
s=s.replace('preparation_v3.json','preparation_v1.json')
s=s.replace("    active_sig=None", "    bind_library()\n    active_sig=None\n    fit_mode='tweedie'")
s=s.replace("model=core.lg(seed,'tweedie');assert model.get_params()==prepared['params'][str(seed)]", "model=core.lg(seed,'tweedie');assert model.get_params()==prepared['params'][str(seed)]\n        if fit_mode!='tweedie':model.set_params(objective=fit_mode)")
anchor="    OUT.mkdir(parents=True,exist_ok=True);outputs=[];audits=[];first=H/'first_fold_verification_v1.json'"
s=s.replace(anchor,'''    OUT.mkdir(parents=True,exist_ok=True)
    controls=[]
    for v,k in FOLD_KEYS:
        tr,q,old,r3s,sig=refs[v,k];active_sig=sig
        for seed in SEEDS:
            dest=OUT/f'{v}_{k}_{seed}_newton_control.npz';meta=OUT/f'{v}_{k}_{seed}_newton_control.json'
            present=[dest.exists(),meta.exists()];assert not any(present) or all(present),'Partial Newton control preserved'
            if all(present):
                record=json.loads(meta.read_text(encoding='utf-8'));assert record['signature']==sig and record['npz_sha256']==S.sha(dest)
                raw=dict(np.load(dest,allow_pickle=False));assert np.array_equal(raw['row_id'],q.row_id)
                compare(raw['raw'],r3s[seed]['raw_lgb'])
            else:
                model=make(tr,bs,seed);raw=predict(model,q,bs)
                try:error=compare(raw,r3s[seed]['raw_lgb'])
                except BaseException:preserve_failed_raw('compiled_newton',v,k,seed,q,raw);raise
                with dest.open('xb') as handle:np.savez_compressed(handle,row_id=q.row_id.to_numpy(str),raw=raw)
                record=dict(status='PASS',signature=sig,seed=seed,npz_sha256=S.sha(dest),max_error=error);savej(meta,record)
                del model;gc.collect()
            controls.append(record);print(v,k,seed,'NEWTON_REPRODUCE_PASS',flush=True)
    assert len(controls)==66
    control_path=H/'compiled_newton_all66_v1.json'
    control_summary=dict(status='PASS',cells=controls,atol=1e-12,score_count=0)
    if control_path.exists():assert json.loads(control_path.read_text(encoding='utf-8'))==control_summary
    else:savej(control_path,control_summary)
    fit_mode='tweedie_exact_leaf'
    outputs=[];audits=[];first=H/'first_fold_verification_v1.json'
''')
start=s.index("                if (v,k,seed)==('DIAG10',0,7):\n                    original=make")
end=s.index('                model=make(tr,cols,seed)',start)
s=s[:start]+"                reproduction=controls[0]['max_error']\n"+s[end:]
s=s.replace("                model=make(tr,cols,seed);raw=predict(model,q,cols)","                model=make(tr,cols,seed);raw=predict(model,q,cols)")
s=s.replace("preparation=prepared,cells=audits,score_count=0", "preparation=prepared,compiled_newton_all66_sha256=S.sha(control_path),cells=audits,score_count=0")
# Causal check uses core38 only; operation variables are deliberately absent.
s=s.replace("    raw=pd.read_csv", "    raw=pd.read_csv")
ast.parse(s)
with (H/'run_v1.py').open('x',encoding='utf-8') as f:f.write(s)
print('RUNNER_GENERATED_NO_FIT')
