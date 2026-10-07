"""Combine completed CPU members, then audit SG2/endpoint outputs before scoring."""
import json,math
from blk_baseline_data_v1 import *
from blk_sg2_refonly_v2 import RefOnlySG2
from blk_endpoint_methods_v2 import EndpointMethodsV2
from checkpoint_v1 import atomic,digest
from blk_cached_pfn_receipt_v2 import verify_cached_pfn

METHODS=['PAST_ENDPOINT','BOTH_ENDPOINT','CHAIN_PREFIX_GUARD']
SCOPES=['BLK_QUERY_ROLE','BLK_RAW_PASS']

def main():
    rp=HERE/'checkpoints/BLK_R3_v1'
    pp=HERE/'checkpoints/BLK_PFN_CPU_REFONLY_v3'
    assert (rp/'complete.json').exists() and (pp/'complete.json').exists(),'original CPU workers must complete; do not substitute PFN'
    cached_receipt=verify_cached_pfn()
    receipt_path=HERE/'BLK_cached_PFN_receipt_v2.json'
    assert json.loads(receipt_path.read_text(encoding='utf-8'))==cached_receipt
    assert cached_receipt['code_sha256']==sha(HERE/'blk_cached_pfn_receipt_v2.py')
    assert set(cached_receipt['context_receipts'])=={'5','6','7','8'}
    member_paths=[rp/f'{n}_seed{s}.json' for n in ['ET','LGB','MLP'] for s in [47,1414,6464]]+[pp/f'context{s}.json' for s in [5,6,7,8]]
    preg=json.loads((pp/'registration.json').read_text(encoding='utf-8'))
    lineage={'R3_registration_sha256':sha(rp/'registration.json'),'PFN_registration_sha256':sha(pp/'registration.json'),
        'member_files_sha256':{str(p.relative_to(HERE)):sha(p) for p in member_paths},'weights_sha256':preg['weights_sha256'],
        'cached_PFN_receipt_sha256':sha(receipt_path),'cached_PFN_receipt':cached_receipt,
        'assembler_sources_sha256':{n:sha(HERE/n) for n in ['blk_assemble_predictions_v1.py','blk_assemble_predictions_v4.py','blk_cached_pfn_receipt_v1.py','blk_cached_pfn_receipt_v2.py']}}
    regpath=HERE/'BLK_method_registration_v4.json'
    rules=json.loads(regpath.read_text(encoding='utf-8'))
    assert all(sha(HERE/p)==s for p,s in rules['method_code_sha256'].items())
    assert rules['PFN_reference_only_registration_sha256']==sha(pp/'registration.json')
    assert rules['layout_sha256']==sha(HERE/'BLK_layout_v2.json') and rules['context_code_sha256']==sha(HERE/'blk_context_v1.py')
    for name in ['BLK_SG2_refonly_audit_v2.json','BLK_SG2_edge_audit_v1.json','BLK_endpoint_boundary_audit_v2.json']:
        assert json.loads((HERE/name).read_text(encoding='utf-8'))['status']=='PASS'
    rreg=json.loads((rp/'registration.json').read_text(encoding='utf-8'))
    assert all(sha(ROOT/p)==s for p,s in rreg['dependencies'].items())
    assert all(sha(Path(env.DATA)/p)==s for p,s in rreg['sources'].items())
    layout=json.loads((HERE/'BLK_layout_v2.json').read_text(encoding='utf-8'))
    assert sha(HERE/'BLK_layout_v2.json')==rreg['layout_sha256']
    ctx=BLKContext(layout);ids=rreg['ordered_query_ids']
    def load(folder,file):
        data=json.loads((folder/file).read_text(encoding='utf-8'))
        assert data['registration_sha256']==sha(folder/'registration.json') and data['row_ids']==ids
        assert digest(data['pred'])==data['pred_sha256']
        assert not data['heldout_truth_loaded']
        if folder==pp:
            audit=json.loads((folder/(Path(file).stem+'_audit.json')).read_text(encoding='utf-8'))
            assert audit['status']=='PASS' and audit['prediction_file_sha256']==sha(folder/file) and audit['max_difference']<=1e-6
        else:
            assert data['single_batch_max_difference']<=1e-6
        return np.array(data['pred'],float)
    pfn=np.mean([load(pp,f'context{s}.json') for s in [5,6,7,8]],axis=0)
    meta=ns['identify'](pd.DataFrame({'row_id':ids}))
    sg=RefOnlySG2(ctx);endpoints=EndpointMethodsV2(layout,ctx.reference_inputs,ctx.reference_labels)
    lo,hi=endpoints.lo,endpoints.hi
    saved={'row_ids':ids,'baseline':{},'candidate':{},'stage_outputs':{},'SG2_coverage':{},'guard_active':[]}
    probes=[f'{b["farm"]}_{b["query_days"][i]:03d}_{h:02d}' for b in layout['blocks'] for i in [0,len(b['query_days'])-1] for h in [0,12,23]]
    audited=0
    for seed in [47,1414,6464]:
        raw_r3=.6*load(rp,f'ET_seed{seed}.json')+.3*load(rp,f'LGB_seed{seed}.json')+.1*load(rp,f'MLP_seed{seed}.json')
        mix=.6*raw_r3+.4*pfn
        shrunk=ns['shrink'](mix,meta)
        # Independent fsum prefix check across every row.
        groups={}
        for i,rid in enumerate(ids):
            f,d,h=key(rid);groups.setdefault((f,d),[]).append(float(mix[i]))
            expected=.5*mix[i]+.5*math.fsum(groups[f,d])/len(groups[f,d])
            assert abs(shrunk[i]-expected)<1e-12
        p13=np.clip(shrunk,lo,hi);predmap=dict(zip(ids,p13))
        saved['stage_outputs'][str(seed)]={'raw_r3':raw_r3.tolist(),'raw_mix':mix.tolist(),'one_shrink':shrunk.tolist(),'pre_SG2_clip':p13.tolist()}
        for scope in SCOPES:
            tag=f'{scope}_seed{seed}';base=[];cand={c:[] for c in METHODS};coverage=[];guard=[]
            for rid in ids:
                f,d,h=key(rid);prefix={f'{f}_{d:03d}_{j:02d}':float(predmap[f'{f}_{d:03d}_{j:02d}']) for j in range(h+1)}
                p,diag=sg.predict_one(rid,prefix,scope);p=float(np.clip(p,lo,hi))
                result=endpoints.predict(rid,p,ctx.query_prefix(rid))
                base.append(p);coverage.append(diag);guard.append(result['chain_guard_active'])
                for c in METHODS:cand[c].append(result[c])
            saved['baseline'][tag]=base;saved['candidate'][tag]=cand
            saved['SG2_coverage'][tag]={'activation':sum(d['active'] for d in coverage),'changed':sum(d['changed'] for d in coverage),'reference_available':sum(d['has_candidate'] for d in coverage)}
            if not saved['guard_active']:saved['guard_active']=guard
            else:assert saved['guard_active']==guard
            for rid in probes:
                f,d,h=key(rid);i=ids.index(rid)
                prefix={f'{f}_{d:03d}_{j:02d}':float(predmap[f'{f}_{d:03d}_{j:02d}']) for j in range(h+1)}
                future=[r for r in ctx._query if key(r)[0]!=f or key(r)[1:]>(d,h)]
                original={r:ctx._query[r] for r in future}
                try:
                    for r in future:ctx._query[r]={c:99999.0 for c in ctx._query[r]}
                    # Later member predictions are deliberately absent from the API.
                    p,_=sg.predict_one(rid,prefix,scope);p=float(np.clip(p,lo,hi))
                    result=endpoints.predict(rid,p,ctx.query_prefix(rid))
                    assert p==base[i]
                    assert all(result[c]==cand[c][i] for c in METHODS)
                finally:ctx._query.update(original)
                audited+=1
            print(f'{tag} assembled, future checks complete',flush=True)
    folder=HERE/'checkpoints/BLK_ASSEMBLED_REFONLY_v1';folder.mkdir(parents=True,exist_ok=True)
    out=folder/'predictions.json';assert not out.exists()
    saved.update({'registration_sha256':sha(regpath),'code_sha256':sha(__file__),'hidden_truth_loaded':False})
    atomic(out,json.dumps(saved,ensure_ascii=False,allow_nan=False))
    manifest={'lineage':lineage,'status':'ASSEMBLED_POSTPROCESS_BOUNDARY_PASS','future_other_farm_final_postprocess_checks':audited,
        'predictions_sha256':sha(out),'registration_sha256':sha(regpath),'builder_sha256':sha(__file__),
        'CPU_member_query_batch_audit':'All4 reference-only PFN receipts independently validated; full final baseline gate still separate',
        'whole_pipeline_gate_passed':False,'performance_evaluated':False,'guard_active_rows':sum(saved['guard_active']),
        'remaining':['independent integrated pipeline review','fresh whole-baseline gate','BLK scorer diagnostic only, not adoption'],
        'baseline_policy':'same recipe plus reference-only cached PFN; earlier uncached outputs excluded'}
    atomic(folder/'audit.json',json.dumps(manifest,ensure_ascii=False,allow_nan=False))
    print('Full mix/postprocess assembled; score remains gated on final audit',flush=True)

if __name__=='__main__':main()
