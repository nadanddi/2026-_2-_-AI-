"""Combine all domain ET additions; strict raw gate and full postprocess audit."""
from blk_baseline_data_v1 import *
from domain_raw_receipt_v1 import verify_raw
from domain_sg2_plan_v1 import SG2InputPlan,SCOPES
from checkpoint_v1 import atomic,digest
import json,math

def main():
    regpath=HERE/'DOMAIN24_BLK_fit_registration_v3.json'
    reg=json.loads(regpath.read_text(encoding='utf-8'))
    pipelinepath=HERE/'DOMAIN24_BLK_pipeline_registration_v1.json'
    pipeline=json.loads(pipelinepath.read_text(encoding='utf-8'))
    assert pipeline['status']=='REGISTERED_BEFORE_DOMAIN_SCORE'
    assert all(sha(Path(path))==value for path,value in pipeline['sources_sha256'].items())
    receipt=verify_raw()
    receipt_path=HERE/'DOMAIN24_raw_receipt_v1.json'
    assert json.loads(receipt_path.read_text(encoding='utf-8'))==receipt
    rawfolder=HERE/'checkpoints/DOMAIN24_BLK_RAW_v3'
    oldfolder=HERE/'checkpoints/BLK_ASSEMBLED_REFONLY_v1'
    old=json.loads((oldfolder/'predictions.json').read_text(encoding='utf-8'))
    layout=json.loads((HERE/'BLK_layout_v2.json').read_text(encoding='utf-8'))
    ctx=BLKContext(layout);ids=old['row_ids']
    assert ids==json.loads((HERE/'checkpoints/BLK_R3_v1/registration.json').read_text(encoding='utf-8'))['ordered_query_ids']
    def load(path):
        record=json.loads(path.read_text(encoding='utf-8'))
        assert record['row_ids']==ids and digest(record['pred'])==record['pred_sha256']
        assert not record['hidden_truth_loaded'] and len(record['pred'])==1440
        return np.asarray(record['pred'],float)
    pfn=np.mean([load(HERE/f'checkpoints/BLK_PFN_CPU_REFONLY_v3/context{s}.json') for s in [5,6,7,8]],axis=0)
    lo=min(label['sub_ec'] for label in ctx.reference_labels.values())
    hi=max(label['sub_ec'] for label in ctx.reference_labels.values())
    meta=ns['identify'](pd.DataFrame({'row_id':ids}))
    plan=SG2InputPlan(ctx,ids)
    probes=[f'{b["farm"]}_{b["query_days"][i]:03d}_{h:02d}' for b in layout['blocks']
            for i in [0,len(b['query_days'])-1] for h in [0,6,23]]
    source_checks=0;maximum=0.;shrink_max=0.
    # All-row source equivalence on the existing baseline pre-SG2 predictions.
    for seed in reg['seeds']:
        values=old['stage_outputs'][str(seed)]['pre_SG2_clip']
        mapping=dict(zip(ids,values))
        for scope in SCOPES:
            for i,rid in enumerate(ids):
                f,d,h=key(rid)
                prefix={f'{f}_{d:03d}_{j:02d}':mapping[f'{f}_{d:03d}_{j:02d}'] for j in range(h+1)}
                maximum=max(maximum,plan.audit_source(rid,prefix,scope))
                assert abs(float(np.clip(plan.apply(rid,prefix,scope),lo,hi))-old['baseline'][f'{scope}_seed{seed}'][i])<1e-12
                source_checks+=1
    saved={'row_ids':ids,'baseline':old['baseline'],'candidate':{},'stage_outputs':{},
           'hidden_truth_loaded':False,'registration_sha256':sha(regpath),'pipeline_registration_sha256':sha(pipelinepath)}
    for seed in reg['seeds']:
        lgb=load(HERE/f'checkpoints/BLK_R3_v1/LGB_seed{seed}.json')
        mlp=load(HERE/f'checkpoints/BLK_R3_v1/MLP_seed{seed}.json')
        for scope in SCOPES:saved['candidate'][f'{scope}_seed{seed}']={}
        for candidate in sorted(reg['family_map']):
            et=load(rawfolder/f'{candidate}_seed{seed}.json')
            mix=.6*(.6*et+.3*lgb+.1*mlp)+.4*pfn
            shrunk=ns['shrink'](mix,meta)
            by_day={}
            for i,rid in enumerate(ids):
                f,d,h=key(rid)
                by_day.setdefault((f,d),[]).append(float(mix[i]))
                independent=.5*mix[i]+.5*math.fsum(by_day[f,d])/len(by_day[f,d])
                difference=abs(float(shrunk[i])-independent)
                assert difference<1e-12
                shrink_max=max(shrink_max,difference)
            pre=np.clip(shrunk,lo,hi);mapping=dict(zip(ids,pre))
            saved['stage_outputs'][f'{candidate}_seed{seed}']={'raw_mix':mix.tolist(),'one_shrink':shrunk.tolist(),'pre_SG2_clip':pre.tolist()}
            for scope in SCOPES:
                final=[]
                for rid in ids:
                    f,d,h=key(rid)
                    prefix={f'{f}_{d:03d}_{j:02d}':float(mapping[f'{f}_{d:03d}_{j:02d}']) for j in range(h+1)}
                    final.append(float(np.clip(plan.apply(rid,prefix,scope),lo,hi)))
                saved['candidate'][f'{scope}_seed{seed}'][candidate]=final
                for rid in probes:
                    f,d,h=key(rid)
                    prefix={f'{f}_{d:03d}_{j:02d}':float(mapping[f'{f}_{d:03d}_{j:02d}']) for j in range(h+1)}
                    maximum=max(maximum,plan.audit_source(rid,prefix,scope));source_checks+=1
                    future=[r for r in ctx._query if key(r)[0]!=f or key(r)[1:]>(d,h)]
                    previous={r:ctx._query[r] for r in future}
                    try:
                        for r in future:ctx._query[r]={column:77777. for column in ctx._query[r]}
                        # Recompute original input-based selection, not only the cached plan.
                        source,_=plan.sg.predict_one(rid,prefix,scope)
                        assert abs(float(np.clip(source,lo,hi))-final[ids.index(rid)])<1e-12
                    finally:ctx._query.update(previous)
                    source_checks+=1
            print(f'{candidate} seed{seed} mixed and SG2 source/future audited',flush=True)
    assert all(sha(Path(path))==value for path,value in pipeline['sources_sha256'].items())
    assert all(sha(rawfolder/name)==value for name,value in receipt['files_sha256'].items())
    folder=HERE/'checkpoints/DOMAIN24_BLK_ASSEMBLED_v1';folder.mkdir(parents=True,exist_ok=True)
    path=folder/'predictions.json';assert not path.exists()
    atomic(path,json.dumps(saved,ensure_ascii=False,allow_nan=False))
    audit={'status':'DOMAIN24_ASSEMBLED_SOURCE_BOUNDARY_PASS_NOT_SCORE_GATE','predictions_sha256':sha(path),
        'registration_sha256':sha(regpath),'pipeline_registration_sha256':sha(pipelinepath),
        'raw_receipt_sha256':sha(receipt_path),'source_equivalence_and_future_checks':source_checks,
        'source_equivalence_max_difference':maximum,'independent_shrink_max_difference':shrink_max,
        'candidate_seed_pairs':72,'scopes':list(SCOPES),'heldout_truth_loaded':False,
        'whole_pipeline_gate_passed':False,'assembler_sha256':sha(__file__),
        'limits':['Raw ET producer source/receipt verified; this assembler does not independently refit ET',
                  'All-row SG2 baseline source comparison plus48probes per candidate-seed-scope; not universal numeric proof',
                  'Final strict verifier and scorer remain separate, no adoption permitted']}
    atomic(folder/'audit.json',json.dumps(audit,ensure_ascii=False,allow_nan=False))
    print('Domain24 full assembly completed; score gate remains closed',flush=True)

if __name__=='__main__':main()
