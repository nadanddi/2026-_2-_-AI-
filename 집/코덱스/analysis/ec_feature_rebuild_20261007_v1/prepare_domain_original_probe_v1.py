"""Three original-validator feature/context probes; no model fit or query truth."""
from blk_baseline_data_v1 import *
from domain_features_v2 import build_domain
from checkpoint_v1 import atomic
import json, time

def main():
    start=time.monotonic()
    registry=HERE/'original_validator_endpoint_registry_v1.json'
    original=json.loads(registry.read_text(encoding='utf-8'))
    fit=json.loads((HERE/'DOMAIN24_BLK_fit_registration_v3.json').read_text(encoding='utf-8'))
    assert all(sha(Path(p))==value for p,value in fit['source_sha256'].items())
    results=[]
    for name in ['DIAG10','P2LOO','EL1']:
        fold=next(f for f in original['folds'] if f['validator']==name)
        # BLKContext is reused only as a filter/prefix loader; no assertion of BLK-shaped folds.
        loader={'status':'REGISTERED','train_ids':fold['ordered_train_ids'],
                'query_ids':fold['ordered_query_ids'],
                'gap_ids_REMOVE_INPUT_AND_BOTH_LABELS':fold['input_forbidden_ids']}
        ctx=BLKContext(loader)
        assert not (set(ctx.reference_labels)&ctx.query_ids)
        frame=dataframe({**ctx.reference_inputs,**ctx._query})
        domain,selected,meta=build_domain(frame)
        assert set(domain.index)==ctx.train_ids|ctx.query_ids
        x,tr,calendar=prepare_reference(ctx)
        q=prepare_query(ctx,fold['ordered_query_ids'],calendar)
        assert list(q.row_id)==fold['ordered_query_ids']
        assert set(tr.row_id)==ctx.train_ids
        assert len(domain.columns)==1187
        for cid,cols in selected.items():
            extra=[c for c in cols if c not in FULL_R3]
            assert extra==fit['family_map'][cid]['additional_ET_columns']
        checks=0
        for rid in [fold['ordered_query_ids'][0],fold['ordered_query_ids'][len(q)//2],fold['ordered_query_ids'][-1]]:
            prefix=dataframe(ctx.query_prefix(rid))
            short,again,_=build_domain(prefix)
            assert again==selected
            np.testing.assert_allclose(short.loc[rid],domain.loc[rid],atol=0,rtol=0,equal_nan=True)
            single=prepare_query(ctx,[rid],calendar).set_index('row_id')
            np.testing.assert_allclose(single.loc[rid,FULL_R3].to_numpy(float),q.set_index('row_id').loc[rid,FULL_R3].to_numpy(float),atol=0,rtol=0,equal_nan=True)
            checks+=2
        results.append({'validator':name,'fold':fold['fold'],'train_rows':len(tr),
            'query_rows':len(q),'input_forbidden_rows':len(ctx.gap_ids),'domain_columns':len(domain.columns),
            'single_prefix_checks':checks,'public_reference_only':True,'heldout_truth_loaded':False})
        print(f'Original {name} fold{fold["fold"]} feature probe PASS; no fit/score',flush=True)
    assert all(sha(Path(p))==value for p,value in fit['source_sha256'].items())
    result={'status':'THREE_ORIGINAL_FOLD_FEATURE_PROBES_PASS_NOT_FULL_VALIDATION',
        'code_sha256':sha(__file__),'registry_sha256':sha(registry),'results':results,
        'duration_seconds':time.monotonic()-start,'fit_or_score_performed':False,
        'limits':['Only first fold per validator; remaining63 contexts and every model fit remain',
                  'Current cached CPU baseline policy still needs registration for original validators',
                  'No historical GPU/submission equivalence claim or candidate adoption']}
    out=HERE/'DOMAIN24_original_feature_probe_v1.json';assert not out.exists()
    atomic(out,json.dumps(result,ensure_ascii=False,allow_nan=False))

if __name__=='__main__':main()
