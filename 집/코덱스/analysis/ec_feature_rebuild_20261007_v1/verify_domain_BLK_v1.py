"""Strict whole domain mixture gate, independent scalar mix/shrink/clip/SG2 arithmetic."""
from blk_baseline_data_v1 import *
from domain_raw_receipt_v1 import verify_raw
from domain_sg2_plan_v1 import SG2InputPlan,SCOPES
from checkpoint_v1 import atomic,digest
import json,math

def main():
    regpath=HERE/'DOMAIN24_BLK_fit_registration_v3.json'
    pipelinepath=HERE/'DOMAIN24_BLK_pipeline_registration_v2.json'
    reg=json.loads(regpath.read_text(encoding='utf-8'))
    pipeline=json.loads(pipelinepath.read_text(encoding='utf-8'))
    assert pipeline['status']=='REGISTERED_BEFORE_DOMAIN_SCORE'
    assert all(sha(Path(p))==value for p,value in pipeline['sources_sha256'].items())
    rawreceiptpath=HERE/'DOMAIN24_raw_receipt_v1.json'
    rawreceipt=verify_raw()
    assert json.loads(rawreceiptpath.read_text(encoding='utf-8'))==rawreceipt
    folder=HERE/'checkpoints/DOMAIN24_BLK_ASSEMBLED_v2'
    path=folder/'predictions.json';auditpath=folder/'audit.json'
    saved=json.loads(path.read_text(encoding='utf-8'))
    audit=json.loads(auditpath.read_text(encoding='utf-8'))
    assert audit['status']=='DOMAIN24_ASSEMBLED_SOURCE_BOUNDARY_PASS_NOT_SCORE_GATE'
    assert audit['predictions_sha256']==sha(path)
    assert audit['assembler_sha256']==sha(HERE/'assemble_domain_BLK_v2.py')
    assert audit['registration_sha256']==saved['registration_sha256']==sha(regpath)
    assert audit['pipeline_registration_sha256']==saved['pipeline_registration_sha256']==sha(pipelinepath)
    assert audit['raw_receipt_sha256']==sha(rawreceiptpath)
    assert audit['candidate_seed_pairs']==72 and audit['scopes']==list(SCOPES)
    assert audit['source_equivalence_and_future_checks']==22464
    assert 0<=audit['source_equivalence_max_difference']<1e-12
    assert 0<=audit['independent_shrink_max_difference']<1e-12
    assert saved['hidden_truth_loaded'] is False and audit['heldout_truth_loaded'] is False
    layout=json.loads((HERE/'BLK_layout_v2.json').read_text(encoding='utf-8'))
    ctx=BLKContext(layout)
    old=json.loads((HERE/'checkpoints/BLK_ASSEMBLED_REFONLY_v1/predictions.json').read_text(encoding='utf-8'))
    ids=saved['row_ids']
    assert ids==old['row_ids'] and len(ids)==len(set(ids))==1440 and set(ids)==ctx.query_ids
    assert saved['baseline']==old['baseline']
    expected_tags={f'{scope}_seed{seed}' for scope in SCOPES for seed in reg['seeds']}
    assert set(saved['candidate'])==set(saved['baseline'])==expected_tags
    assert set(saved['stage_outputs'])=={f'{candidate}_seed{seed}' for candidate in reg['family_map'] for seed in reg['seeds']}
    plan=SG2InputPlan(ctx,ids)
    lo=min(v['sub_ec'] for v in ctx.reference_labels.values())
    hi=max(v['sub_ec'] for v in ctx.reference_labels.values())
    def load(p):
        data=json.loads(p.read_text(encoding='utf-8'))
        assert data['row_ids']==ids and digest(data['pred'])==data['pred_sha256']
        assert len(data['pred'])==1440 and all(math.isfinite(v) for v in data['pred'])
        return data['pred']
    contexts=[load(HERE/f'checkpoints/BLK_PFN_CPU_REFONLY_v3/context{s}.json') for s in [5,6,7,8]]
    maximum=0.;checks=0
    for seed in reg['seeds']:
        lgb=load(HERE/f'checkpoints/BLK_R3_v1/LGB_seed{seed}.json')
        mlp=load(HERE/f'checkpoints/BLK_R3_v1/MLP_seed{seed}.json')
        for candidate in sorted(reg['family_map']):
            et=load(HERE/f'checkpoints/DOMAIN24_BLK_RAW_v3/{candidate}_seed{seed}.json')
            stage=saved['stage_outputs'][f'{candidate}_seed{seed}']
            assert set(stage)=={'raw_mix','one_shrink','pre_SG2_clip'}
            assert all(len(v)==1440 and all(math.isfinite(x) for x in v) for v in stage.values())
            accumulated={};clipped={}
            for i,rid in enumerate(ids):
                f,d,h=key(rid)
                pfn=math.fsum(context[i] for context in contexts)/4
                mix=.6*math.fsum([.6*et[i],.3*lgb[i],.1*mlp[i]])+.4*pfn
                accumulated.setdefault((f,d),[]).append(mix)
                shrunk=.5*mix+.5*math.fsum(accumulated[f,d])/len(accumulated[f,d])
                clip=max(lo,min(hi,shrunk));clipped[rid]=clip
                for actual,expected in [(stage['raw_mix'][i],mix),(stage['one_shrink'][i],shrunk),(stage['pre_SG2_clip'][i],clip)]:
                    difference=abs(actual-expected);assert difference<1e-12
                    maximum=max(maximum,difference);checks+=1
            for scope in SCOPES:
                tag=f'{scope}_seed{seed}'
                assert set(saved['candidate'][tag])==set(reg['family_map'])
                predicted=saved['candidate'][tag][candidate]
                assert len(predicted)==1440 and all(math.isfinite(v) and lo<=v<=hi for v in predicted)
                for i,rid in enumerate(ids):
                    f,d,h=key(rid)
                    # Same registered floating branch; independent fsum arithmetic inside correction.
                    original_prefix=[stage['pre_SG2_clip'][ids.index(f'{f}_{d:03d}_{j:02d}')] for j in range(h+1)]
                    independent_mean=math.fsum(clipped[f'{f}_{d:03d}_{j:02d}'] for j in range(h+1))/(h+1)
                    original_mean=float(np.mean(original_prefix))
                    assert abs(independent_mean-original_mean)<1e-12
                    chosen=plan.choices[scope,rid]
                    value=clipped[rid]
                    if chosen['has_candidate'] and abs(chosen['level']-original_mean)<=.30:
                        value+=.5*(chosen['level']-independent_mean)
                    expected=max(lo,min(hi,value))
                    difference=abs(predicted[i]-expected);assert difference<1e-12
                    maximum=max(maximum,difference);checks+=1
    transitive=dict(reg['source_sha256'])
    transitive.update(pipeline['sources_sha256'])
    for p in [regpath,pipelinepath,path,auditpath,rawreceiptpath,HERE/'checkpoints/DOMAIN24_BLK_RAW_v3/complete.json']:
        transitive[str(p.resolve())]=sha(p)
    for name,value in rawreceipt['files_sha256'].items():transitive[str((HERE/'checkpoints/DOMAIN24_BLK_RAW_v3'/name).resolve())]=value
    assert all(sha(Path(p))==value for p,value in transitive.items())
    receipt={'status':'VERIFIED_DOMAIN24_BLK_FOR_DIAGNOSTIC','whole_pipeline_gate_passed':True,
        'code_sha256':sha(__file__),'predictions_sha256':sha(path),'assembly_audit_sha256':sha(auditpath),
        'fit_registration_sha256':sha(regpath),'pipeline_registration_sha256':sha(pipelinepath),
        'raw_receipt_sha256':sha(rawreceiptpath),'raw_helper_sha256':sha(HERE/'domain_raw_receipt_v1.py'),
        'transitive_sources_sha256':transitive,'independent_scalar_checks':checks,
        'independent_scalar_max_difference':maximum,'heldout_truth_loaded':False,'adoption_permitted':False,
        'limits':['Finite source audits and independent arithmetic; not universal proof or independent ET refit',
                  'SG2 source selection plan includes query inference only, no query statistics fit',
                  'Already exposed BLK diagnostic sample; all24 original validators remain mandatory']}
    out=HERE/'DOMAIN24_verified_gate_v1.json';assert not out.exists()
    atomic(out,json.dumps(receipt,ensure_ascii=False,allow_nan=False))
    print(f'Domain24 diagnostic whole gate VERIFIED: {checks} scalar checks max{maximum}',flush=True)

if __name__=='__main__':main()
