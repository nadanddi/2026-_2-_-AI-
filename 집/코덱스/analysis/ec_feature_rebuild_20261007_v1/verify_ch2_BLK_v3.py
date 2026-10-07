"""Fresh all-row replay and independent endpoint arithmetic before any CH2 target scoring."""
from pathlib import Path
import json,hashlib,math,sys
import run_ch2_BLK_v3 as runner
import ch2_query_context_v2 as context_module
import ch2_reference_loader_v1 as reference_loader
import ch2_prefix_sources_v3 as prefix_module

HERE=Path(__file__).resolve().parent
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(name):return json.loads((HERE/name).read_text(encoding='utf-8'))

def main():
    if not __debug__ or sys.flags.optimize:raise RuntimeError('Python -O prohibited')
    for module,name in [(runner,'run_ch2_BLK_v3.py'),(context_module,'ch2_query_context_v2.py'),
                        (reference_loader,'ch2_reference_loader_v1.py'),(prefix_module,'ch2_prefix_sources_v3.py')]:
        assert Path(module.__file__).resolve()==(HERE/name).resolve()
    regpath=HERE/'CH2_BLK_registration_v3.json';reg=read(regpath.name)
    assert all(sha(p)==v for p,v in reg['source_sha256'].items())
    folder=HERE/'checkpoints/CH2_BLK_v1'
    pred=json.loads((folder/'predictions.json').read_text(encoding='utf-8'))
    audit=json.loads((folder/'audit.json').read_text(encoding='utf-8'))
    assert audit['status']=='CH2_INFERENCE_BOUNDARIES_PASS_NOT_SCORE_GATE'
    assert not audit['heldout_truth_loaded'] and not pred['hidden_truth_loaded']
    assert not audit['gap_values_parsed'] and not audit['query_preprocessing_fit'] and not audit['score_evaluated']
    assert audit['runner_sha256']==sha(HERE/'run_ch2_BLK_v3.py')
    assert audit['registration_sha256']==pred['registration_sha256']==sha(regpath)
    assert audit['source_sha256']==reg['source_sha256'] and audit['predictions_sha256']==sha(folder/'predictions.json')
    baseline,gate=runner.baseline_preflight()
    assert pred['baseline']==baseline['baseline'] and pred['row_ids']==baseline['row_ids']==reg['ordered_query_ids']
    assert audit['baseline_gate_sha256']==sha(HERE/'BLK_verified_baseline_receipt_v4.json')
    model,layout,refdetails=reference_loader.load_reference()
    assert audit['reference_replay']==refdetails
    ctx=context_module.CH2QueryContext(layout,model);ids=pred['row_ids']
    assert len(ids)==len(set(ids))==audit['query_rows']==1440
    assert set(pred['candidate'])=={f'{scope}_seed{s}' for scope in runner.SCOPES for s in runner.SEEDS}
    assert all(set(v)==set(prefix_module.METHODS) for v in pred['candidate'].values())
    assert all(len(v)==1440 and all(math.isfinite(x) for x in v) for group in pred['candidate'].values() for v in group.values())
    assert set(pred['diagnostics'])==set(prefix_module.METHODS) and all(len(v)==1440 for v in pred['diagnostics'].values())
    assert len(audit['ordered_query_prefix_sha256'])==1440
    probes={f'{b["farm"]}_{b["query_days"][i]:03d}_{h:02d}' for b in layout['blocks']
            for i in [0,len(b['query_days'])//2,len(b['query_days'])-1] for h in [0,3,6,12,23]}
    expected_checks={(rid,m) for rid in probes for m in prefix_module.METHODS}
    checks=audit['boundary_checks']
    assert len(checks)==audit['boundary_check_count']==360
    assert {(v['rid'],v['method']) for v in checks}==expected_checks
    assert all(v['forbidden_poison_and_reverse_replay'] is True for v in checks)
    # Verify every recorded numeric-consumption set, including audit calls, independently.
    seen={}
    for log in audit['consumption_log']:
        rid=log['rid'];assert rid in ids
        farm,day,hour=prefix_module.key(rid);block=model.blocks[model.query_to_block[rid]]
        wanted=sorted(f'{farm}_{d:03d}_{h:02d}' for d in block['query_days'] if d<=day for h in range(24 if d<day else hour+1))
        assert log['ordered_ids']==wanted;seen[rid]=seen.get(rid,0)+1
    assert set(seen)==set(ids)
    assert all(seen[rid]==(2 if rid in probes else 1) for rid in ids)
    maximum=0.;scalar_checks=0;active={m:0 for m in prefix_module.METHODS}
    for i,rid in enumerate(ids):
        packet=ctx.query_prefix(rid)
        encoded=json.dumps(packet,sort_keys=True,allow_nan=False).encode()
        assert hashlib.sha256(encoded).hexdigest()==audit['ordered_query_prefix_sha256'][i]
        farm,day,hour=prefix_module.key(rid)
        for method in prefix_module.METHODS:
            anchors,diag=model.choose(rid,packet,method)
            assert json.loads(json.dumps(diag,allow_nan=False))==pred['diagnostics'][method][i]
            active[method]+=anchors is not None
            if anchors is not None:
                f,left,right=anchors
                # Independently express interpolation as distances to each fixed endpoint.
                dl=24*(day-left)+hour-23;dr=24*(right-day)-hour
                assert dl>0 and dr>0
                level=(dr*model.records[f,left][23]['sub_ec']+dl*model.records[f,right][0]['sub_ec'])/(dl+dr)
            for scope in runner.SCOPES:
                for seed in runner.SEEDS:
                    name=f'{scope}_seed{seed}';base=baseline['baseline'][name][i]
                    expected=base if anchors is None else max(model.bounds[0],min(model.bounds[1],.8*base+.2*level))
                    actual=pred['candidate'][name][method][i]
                    delta=abs(actual-expected);maximum=max(maximum,delta);scalar_checks+=1
                    assert delta<1e-12
                    if anchors is None:assert actual==base
    assert scalar_checks==25920 and all(sha(p)==v for p,v in reg['source_sha256'].items())
    pins=dict(reg['source_sha256'])
    for path in [regpath,folder/'audit.json',folder/'predictions.json',Path(__file__)]:pins[str(path.resolve())]=sha(path)
    result={'status':'CH2_FULL_INFERENCE_GATE_VERIFIED_FOR_DIAGNOSTIC_ONLY','source_sha256':pins,
            'registration_sha256':sha(regpath),'predictions_sha256':sha(folder/'predictions.json'),
            'verifier_sha256':sha(__file__),'rows':1440,'independent_scalar_checks':scalar_checks,
            'independent_arithmetic_max_difference':maximum,'supported_rows':active,
            'actual_consumption_rows_checked':len(audit['consumption_log']),
            'all_query_prefix_replay_checked':1440,'boundary_checks':360,'heldout_truth_loaded':False,
            'diagnostic_scoring_permitted':True,'adoption_permitted':False,
            'limits':['Source/schema/finite empirical audit, not exhaustive mathematical proof',
                      'Cached CPU baseline not historical GPU identity; pass1 BLK generalization limited',
                      'Original validators/new-seed-layout/literature/data/full report remain required']}
    output=HERE/'CH2_BLK_verified_gate_v3.json';assert not output.exists()
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'CH2 full inference gate25920 scalar PASS max{maximum}; no heldout score',flush=True)

if __name__=='__main__':main()
