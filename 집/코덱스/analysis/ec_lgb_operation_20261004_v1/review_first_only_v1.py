"""First audit schema/signatures/features only; no prediction arrays or scores."""
from pathlib import Path
import sys, importlib.util, json
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('whole_first_only',H/'verify_full_v3.py')
V=importlib.util.module_from_spec(s);s.loader.exec_module(V)
V.bootstrap(True)
first=V.readj(H/'first_fold_verification_v1.json');prep=V.readj(H/'preparation_v3.json')
assert V.sha(H/'run_v3.py')==V.RUN_SHA and V.sha(H/'preparation_v3.json')==V.PREP_SHA and V.sha(H/'preregistration_v1.md')==V.PREREG_SHA
sig=dict(**prep['manifest'][0],seed=7,preparation_sha256=V.PREP_SHA,preregistration_sha256=V.PREREG_SHA)
lab,core,wv,folds,_=V.S.loadec();lab=V.operations(lab)
v,k,tm,vm=folds[0];assert (v,k)==('DIAG10',0)
tr,q=V.S.seasonal(lab[tm],lab[vm],wv);tr,q=tr.reset_index(drop=True),q.reset_index(drop=True)
assert sig['train_ids']==V.ids_sha(tr.row_id) and sig['query_ids']==V.ids_sha(q.row_id)
for part,frame in [('train',tr),('query',q)]:
    for key,cols in [('full38',sig['full_features']),('old14',sig['old_lgb_features']),('new23',sig['new_lgb_features'])]:
        assert sig[part+'_'+key]==V.array_sha(frame[cols].to_numpy(float))
for relative,digest in sig['cache_hashes'].items():assert V.sha(V.ROOT/relative)==digest
V.validate_first(first,sig,lab,q)
meta=V.readj(V.OUT/'DIAG10_0_7.json')
V.validate_meta(meta,sig,V.sha(V.OUT/'DIAG10_0_7_pred.csv'),V.sha(H/'first_fold_verification_v1.json'))
# Read only ordered metadata columns, without materializing predictions or targets.
d=V.pd.read_csv(V.OUT/'DIAG10_0_7_pred.csv',usecols=['row_id','farm','day','hour'])
for col in d:assert V.np.array_equal(d[col],q[col])
result=dict(status='PASS_FIRST_RECORD_ONLY',train_rows=len(tr),query_rows=len(q),source_sha=V.RUN_SHA,prep_sha=V.PREP_SHA,first_sha=V.sha(H/'first_fold_verification_v1.json'),feature_hash_comparisons=6,ordered_metadata_columns=4,cache_hash_comparisons=len(sig['cache_hashes']),fit=0,predict=0,scores=0,actual_prediction_arrays_used=0,limitation='Saved audit numbers checked; models not replayed. No whole-run or performance conclusion.')
with (H/'first_only_independent_v1.json').open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
print(json.dumps(result,ensure_ascii=False))
