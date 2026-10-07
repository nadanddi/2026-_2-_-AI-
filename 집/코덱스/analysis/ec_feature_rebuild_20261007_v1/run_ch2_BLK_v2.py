"""Registered fixed-flank CH2 inference using the verified cached baseline; no target scoring."""
from pathlib import Path
from collections import Counter
import json,hashlib,math,sys
import ch2_prefix_sources_v3 as prefix_module
import ch2_reference_loader_v1 as reference_loader
from ch2_query_context_v1 import CH2QueryContext

HERE=Path(__file__).resolve().parent
SCOPES=('BLK_QUERY_ROLE','BLK_RAW_PASS');SEEDS=(47,1414,6464)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(name):return json.loads((HERE/name).read_text(encoding='utf-8'))
def write_new(path,data):
    assert not path.exists()
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def baseline_preflight():
    gate=read('BLK_verified_baseline_receipt_v4.json')
    assert gate['status']=='VERIFIED_BASELINE_FOR_BLK_DIAGNOSTIC' and gate['whole_pipeline_gate_passed']
    assert not gate['heldout_truth_loaded']
    assert gate['code_sha256']==sha(HERE/'verify_BLK_baseline_v4.py')
    for name in ['verifier_lineage_sha256','verified_evidence_sha256']:
        assert all(sha(HERE/p)==v for p,v in gate[name].items())
    assert len(gate['transitive_source_sha256'])>=40
    assert all(Path(p).is_absolute() and sha(p)==v for p,v in gate['transitive_source_sha256'].items())
    path=HERE/'checkpoints/BLK_ASSEMBLED_REFONLY_v1/predictions.json'
    assert sha(path)==gate['predictions_sha256']
    predictions=json.loads(path.read_text(encoding='utf-8'))
    assert not predictions['hidden_truth_loaded']
    assert gate['method_registration_sha256']==predictions['registration_sha256']==sha(HERE/'BLK_method_registration_v4.json')
    assert gate['cached_PFN_receipt_sha256']==sha(HERE/'BLK_cached_PFN_receipt_v2.json')
    return predictions,gate

def main():
    if not __debug__ or sys.flags.optimize:raise RuntimeError('Python -O prohibited')
    registration=read('CH2_BLK_registration_v2.json')
    assert registration['status']=='REGISTERED_BEFORE_REAL_QUERY_INFERENCE'
    assert registration['methods']==list(prefix_module.METHODS)
    assert registration['scopes']==list(SCOPES) and registration['seeds']==list(SEEDS)
    assert registration['statistics']['comparison_alpha']==.025/60
    assert all(sha(p)==value for p,value in registration['source_sha256'].items())
    model,layout,refdetails=reference_loader.load_reference()
    base,basegate=baseline_preflight();ids=base['row_ids']
    assert ids==registration['ordered_query_ids'] and set(ids)==set(layout['query_ids'])
    ctx=CH2QueryContext(layout,model)
    outputs={f'{scope}_seed{seed}':{m:[] for m in prefix_module.METHODS} for scope in SCOPES for seed in SEEDS}
    diagnostics={m:[] for m in prefix_module.METHODS};checks=[]
    prefixhash=[]
    probes={f'{b["farm"]}_{b["query_days"][i]:03d}_{h:02d}' for b in layout['blocks']
            for i in [0,len(b['query_days'])//2,len(b['query_days'])-1] for h in [0,3,6,12,23]}
    for i,rid in enumerate(ids):
        packet=ctx.query_prefix(rid)
        current={m:model.choose(rid,packet,m) for m in prefix_module.METHODS}
        prefixhash.append(hashlib.sha256(json.dumps(packet,sort_keys=True,allow_nan=False).encode()).hexdigest())
        for method in prefix_module.METHODS:diagnostics[method].append(current[method][1])
        for scope in SCOPES:
            for seed in SEEDS:
                name=f'{scope}_seed{seed}';baseline=base['baseline'][name][i]
                for method in prefix_module.METHODS:
                    value,diag=model.predict(rid,baseline,packet,method)
                    assert math.isfinite(value) and model.bounds[0]<=value<=model.bounds[1]
                    outputs[name][method].append(value)
        if rid in probes:
            # Poison every forbidden query string before prefix generation: no conversion is allowed.
            forbidden=set(ctx.query_ids)-set(packet)
            saved={k:ctx._query[k] for k in forbidden}
            try:
                for k in forbidden:ctx._query[k]={c:'FORBIDDEN_FLOAT_CONVERSION' for c in prefix_module.RAW}
                assert ctx.query_prefix(rid)==packet
                for method in prefix_module.METHODS:
                    assert model.choose(rid,dict(reversed(list(packet.items()))),method)==current[method]
                    checks.append({'rid':rid,'method':method,'forbidden_poison_and_reverse_replay':True})
            finally:ctx._query.update(saved)
    assert len(probes)==120 and len(checks)==360
    assert all(len(v)==1440 for scope in outputs.values() for v in scope.values())
    assert all(sha(p)==value for p,value in registration['source_sha256'].items())
    # Fresh reference file/source replay at end; never refit or mutate inference reference.
    _,_,after=reference_loader.load_reference();assert after==refdetails
    folder=HERE/'checkpoints/CH2_BLK_v1';folder.mkdir(parents=True,exist_ok=True)
    prediction={'row_ids':ids,'baseline':base['baseline'],'candidate':outputs,
                'diagnostics':diagnostics,'registration_sha256':sha(HERE/'CH2_BLK_registration_v2.json'),
                'hidden_truth_loaded':False}
    write_new(folder/'predictions.json',prediction)
    audit={'status':'CH2_INFERENCE_BOUNDARIES_PASS_NOT_SCORE_GATE','predictions_sha256':sha(folder/'predictions.json'),
           'registration_sha256':prediction['registration_sha256'],'runner_sha256':sha(__file__),
           'source_sha256':registration['source_sha256'],'reference_replay':refdetails,
           'baseline_gate_sha256':sha(HERE/'BLK_verified_baseline_receipt_v4.json'),
           'ordered_query_prefix_sha256':prefixhash,'consumption_log':ctx.consumed,'boundary_checks':checks,
           'boundary_check_count':len(checks),'query_rows':len(ids),'heldout_truth_loaded':False,
           'gap_values_parsed':False,'query_preprocessing_fit':False,'GPU_used':False,'score_evaluated':False,
           'reason_counts':{m:dict(Counter(d['reason'] for d in rows)) for m,rows in diagnostics.items()}}
    write_new(folder/'audit.json',audit)
    print('CH2 inference1440 rows x3methods x2scopes x3seeds;360 boundary checks; no score',flush=True)

if __name__=='__main__':main()
