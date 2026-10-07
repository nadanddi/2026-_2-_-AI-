"""Verify actual baseline lineage and full-stage tests before opening score gate."""
from pathlib import Path
import json,hashlib,math
from verify_BLK_baseline_v2 import preflight_boundaries
from blk_cached_pfn_receipt_v2 import verify_cached_pfn
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def main():
    assembled=HERE/'checkpoints/BLK_ASSEMBLED_REFONLY_v1'
    manifest=read(assembled/'audit.json');preds=read(assembled/'predictions.json')
    assert manifest['status']=='ASSEMBLED_POSTPROCESS_BOUNDARY_PASS'
    assert manifest['predictions_sha256']==sha(assembled/'predictions.json')
    rules=HERE/'BLK_method_registration_v4.json'
    assert manifest['registration_sha256']==preds['registration_sha256']==sha(rules)
    registration=read(rules)
    assert all(sha(HERE/n)==s for n,s in registration['method_code_sha256'].items())
    assert manifest['builder_sha256']==sha(HERE/'blk_assemble_predictions_v4.py')
    assert all(sha(HERE/n)==s for n,s in manifest['lineage']['assembler_sources_sha256'].items())
    assert all(sha(HERE/n)==s for n,s in manifest['lineage']['member_files_sha256'].items())
    rr=HERE/'checkpoints/BLK_R3_v1/registration.json';pr=HERE/'checkpoints/BLK_PFN_CPU_REFONLY_v3/registration.json'
    rreg=read(rr);preg=read(pr)
    assert manifest['lineage']['R3_registration_sha256']==preg['R3_registration_sha256']==sha(rr)
    assert manifest['lineage']['PFN_registration_sha256']==sha(pr)
    assert all(sha(ROOT/n)==s for n,s in rreg['dependencies'].items())
    assert all(sha(ROOT/'공용/대회자료/정형데이터/참가자_배포'/n)==s for n,s in rreg['sources'].items())
    assert preg['columns']==rreg['feature_columns']['PFN_NOT_EXECUTED'] and preg['contexts']==rreg['PFN_context_ids_NOT_EXECUTED']
    assert sha(preg['weights_path'])==preg['weights_sha256']==manifest['lineage']['weights_sha256']
    assert sha(HERE/'BLK_layout_v2.json')==rreg['layout_sha256']==registration['layout_sha256']
    assert sha(HERE/'blk_context_v1.py')==registration['context_code_sha256']
    checked=[]
    raw=read(HERE/'BLK_R3_full_future_audit_v1.json')
    assert raw['status']=='PASS' and raw['members']==9 and raw['future_other_farm_model_checks']==54
    assert raw['registration_sha256']==sha(rr) and raw['code_sha256']==sha(HERE/'blk_r3_future_audit_v1.py')
    checked.append('BLK_R3_full_future_audit_v1.json')
    cached_receipt=verify_cached_pfn()
    cached_path=HERE/'BLK_cached_PFN_receipt_v2.json'
    assert read(cached_path)==cached_receipt
    assert cached_receipt['code_sha256']==sha(HERE/'blk_cached_pfn_receipt_v2.py')
    assert cached_receipt['parent_verifier_sha256']==sha(HERE/'blk_cached_pfn_receipt_v1.py')
    assert set(cached_receipt['context_receipts'])=={'5','6','7','8'}
    assert registration['PFN_reference_only_registration_sha256']==sha(pr)==cached_receipt['registration_sha256']
    assert manifest['lineage']['cached_PFN_receipt']==cached_receipt
    assert manifest['lineage']['cached_PFN_receipt_sha256']==sha(cached_path)
    assert all(sha(HERE/n)==s for n,s in cached_receipt['receipt_files_sha256'].items())
    checked.append('BLK_cached_PFN_receipt_v2.json')
    checked.extend(cached_receipt['receipt_files_sha256'])
    verifier_lineage=preflight_boundaries()
    for name in ['verify_BLK_baseline_v3.py','verify_BLK_baseline_v4.py','blk_cached_pfn_receipt_v1.py','blk_cached_pfn_receipt_v2.py','blk_assemble_predictions_v4.py','BLK_method_registration_v4.json']:
        verifier_lineage[name]=sha(HERE/name)
    for name in ['BLK_SG2_refonly_audit_v2.json','BLK_SG2_edge_audit_v1.json','BLK_endpoint_boundary_audit_v2.json']:
        assert read(HERE/name)['status']=='PASS';checked.append(name)
    assert manifest['future_other_farm_final_postprocess_checks']==288
    ids=preds['row_ids'];assert len(ids)==len(set(ids))==1440 and ids==rreg['ordered_query_ids']
    assert len(preds['baseline'])==6 and len(preds['candidate'])==6
    assert not preds['hidden_truth_loaded']
    for name,values in preds['baseline'].items():
        assert len(values)==1440 and all(math.isfinite(x) for x in values)
        assert set(preds['candidate'][name])=={'PAST_ENDPOINT','BOTH_ENDPOINT','CHAIN_PREFIX_GUARD'}
        assert all(len(v)==1440 and all(math.isfinite(x) for x in v) for v in preds['candidate'][name].values())
        if manifest['guard_active_rows']==0:assert preds['candidate'][name]['CHAIN_PREFIX_GUARD']==values
    # Independent raw member mix and one-shrink recomputation in stdlib.
    pfn=[read(pr.parent/f'context{s}.json')['pred'] for s in [5,6,7,8]]
    train_labels={}
    import csv
    allowed=set(rreg['ordered_train_ids'])
    with (ROOT/'공용/대회자료/정형데이터/참가자_배포/train_y.csv').open(encoding='utf-8-sig',newline='') as f:
        for row in csv.DictReader(f):
            if row['row_id'] in allowed:train_labels[row['row_id']]=float(row['sub_ec'])
    lo,hi=min(train_labels.values()),max(train_labels.values())
    differences=[]
    for seed in [47,1414,6464]:
        et=read(rr.parent/f'ET_seed{seed}.json')['pred'];lg=read(rr.parent/f'LGB_seed{seed}.json')['pred'];mlp=read(rr.parent/f'MLP_seed{seed}.json')['pred']
        prefix={}
        for i,rid in enumerate(ids):
            raw_r3=.6*et[i]+.3*lg[i]+.1*mlp[i]
            mix=.6*raw_r3+.4*(math.fsum(v[i] for v in pfn)/4)
            prefix.setdefault(rid[:7],[]).append(mix)
            expected=.5*mix+.5*math.fsum(prefix[rid[:7]])/len(prefix[rid[:7]])
            clipped=max(lo,min(hi,expected))
            differences.append(abs(preds['baseline'][f'BLK_RAW_PASS_seed{seed}'][i]-clipped))
    assert max(differences)<=1e-12
    transitive_paths={rules,rr,pr,assembled/'audit.json',assembled/'predictions.json',cached_path,
        HERE/'BLK_layout_v2.json',HERE/'blk_context_v1.py',HERE/'blk_cached_pfn_receipt_v2.py',Path(preg['weights_path'])}
    transitive_paths.update(HERE/n for n in registration['method_code_sha256'])
    transitive_paths.update(ROOT/n for n in rreg['dependencies'])
    transitive_paths.update(ROOT/'공용/대회자료/정형데이터/참가자_배포'/n for n in rreg['sources'])
    transitive_paths.update(ROOT/n for n in preg['library_sha256'])
    transitive_paths.update(Path(v['path']) for v in preg['runtime_module_sources'].values())
    transitive_paths.update(HERE/n for n in manifest['lineage']['member_files_sha256'])
    transitive_paths.update(HERE/n for n in manifest['lineage']['assembler_sources_sha256'])
    transitive_paths.update(HERE/n for n in checked)
    transitive_paths.update(HERE/n for n in verifier_lineage)
    transitive_source_sha256={str(p.resolve()):sha(p) for p in sorted(transitive_paths,key=lambda p:str(p.resolve()))}
    assert len(transitive_source_sha256)>=40
    receipt={'transitive_source_sha256':transitive_source_sha256,'verifier_lineage_sha256':verifier_lineage,'cached_PFN_receipt_sha256':sha(cached_path),'cached_PFN_helper_sha256':sha(HERE/'blk_cached_pfn_receipt_v2.py'),'status':'VERIFIED_BASELINE_FOR_BLK_DIAGNOSTIC','whole_pipeline_gate_passed':True,
        'predictions_sha256':sha(assembled/'predictions.json'),'method_registration_sha256':sha(rules),
        'code_sha256':sha(__file__),'verified_evidence_sha256':{n:sha(HERE/n) for n in checked},
        'raw_mix_one_shrink_clip_independent_max_difference':max(differences),'final_postprocess_causal_checks':288,
        'scope':'CPU recipe baseline with reference-only frozen internal PFN cache under BLK_QUERY_ROLE and RAW_PASS; excludes earlier uncached output; not byte identity claim against historical GPU OOF or PFN0.2 submission14',
        'heldout_truth_loaded':False,'adoption_permitted':False,
        'limits':['Finite empirical tests plus code/ID contracts, not exhaustive proof over all possible numeric inputs','original validators and new seed/layout confirmation remain','BLK pass1 only; actual pass2 generalization unproven']}
    out=HERE/'BLK_verified_baseline_receipt_v4.json';assert not out.exists()
    out.write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Actual CPU BLK baseline gate VERIFIED for diagnostic scoring only',flush=True)

if __name__=='__main__':main()
