"""Exercise new verifier guard on existing permitted R3 caches; no TabDPT result/score reads."""
from pathlib import Path
import importlib.util, json, sys
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent
dest=H/'original_r3_precheck_v1.json'
assert not dest.exists()
sp=importlib.util.spec_from_file_location('tabdpt_whole_guard',H/'verify_full_v2.py')
V=importlib.util.module_from_spec(sp);sp.loader.exec_module(V)
V.bootstrap_runtime(with_support=True)
lab,core,wv,folds,outer=V.S.loadec()
labels=lab.set_index('row_id').sub_ec
records=[]
for v,k,tm,vm in folds:
    tr,q=V.S.seasonal(lab[tm],lab[vm],wv)
    tr,q=tr.reset_index(drop=True),q.reset_index(drop=True)
    for seed in V.SEEDS:
        path=V.OLD/f'{v}_{k}_r3_{seed}.npz'
        meta=V.readj(path.with_suffix('.json'))
        arrays=dict(V.np.load(path,allow_pickle=False))
        record=V.guard_original_r3(arrays,meta,V.sha(path),tr,q,seed,v,k,labels)
        record['metadata_sha256']=V.sha(path.with_suffix('.json'))
        records.append(record)
assert len(records)==66
result=dict(status='PASS_EXISTING_R3_ONLY',cells=66,checks=V.CHECKS,
            verifier_sha256=V.sha(H/'verify_full_v2.py'),records=records,
            tabdpt_result_reads=0,new_score_calculations=0,fit=0,predict=0,
            limitations=['Existing cache arithmetic/provenance check, not member refit or full original feature-hash replay.'])
with dest.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
print(json.dumps({k:v for k,v in result.items() if k!='records'},ensure_ascii=False))
