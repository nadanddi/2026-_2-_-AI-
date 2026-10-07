"""Feature-only preparation for all original folds; never fit or score a model."""
from blk_baseline_data_v1 import *
from domain_features_v2 import build_domain
from checkpoint_v1 import atomic,digest
from checkpoint_v2 import start_ticks
import json,hashlib,os,time,platform
import sklearn

def matrix_sha(frame):
    values=frame.to_numpy(dtype='<f8',copy=True)
    values[np.isnan(values)]=np.nan
    return hashlib.sha256(values.tobytes()).hexdigest()

def validate_sources(reg):
    assert all(sha(Path(path))==value for path,value in reg['sources_sha256'].items())

def main():
    regpath=HERE/'DOMAIN24_original_preparation_registration_v1.json'
    reg=json.loads(regpath.read_text(encoding='utf-8'))
    assert reg['status']=='REGISTERED_ORIGINAL_FEATURE_PREPARATION_NOT_MODEL_FIT'
    assert reg['runner_sha256']==sha(__file__)
    validate_sources(reg)
    expected=reg['environment']
    assert platform.python_version()==expected['python']
    assert np.__version__==expected['numpy'] and pd.__version__==expected['pandas']
    assert sklearn.__version__==expected['sklearn']
    registry=json.loads((HERE/'original_validator_endpoint_registry_v1.json').read_text(encoding='utf-8'))
    folder=HERE/'checkpoints/DOMAIN24_ORIGINAL_PREPARATION_v1'
    folder.mkdir(parents=True,exist_ok=True)
    lock=folder/'RUN_WRITER_LOCK.json'
    token={'pid':os.getpid(),'start_ticks':start_ticks(os.getpid()),'registration_sha256':sha(regpath)}
    fd=os.open(lock,os.O_WRONLY|os.O_CREAT|os.O_EXCL)
    with os.fdopen(fd,'w',encoding='utf-8') as handle:json.dump(token,handle)
    files={}
    try:
        for fold in registry['folds']:
            validate_sources(reg)
            name=f'{fold["validator"]}_fold{fold["fold"]}.json'
            path=folder/name
            ordered_train=fold['ordered_train_ids'];ordered_query=fold['ordered_query_ids']
            if path.exists():
                saved=json.loads(path.read_text(encoding='utf-8'))
                signature=saved.pop('payload_sha256')
                assert digest(saved)==signature and saved['registration_sha256']==sha(regpath)
                assert saved['ordered_query_ids']==ordered_query and saved['ordered_train_ids']==ordered_train
                assert saved['input_forbidden_ids']==fold['input_forbidden_ids']
                assert saved['status']=='PREPARED_FEATURES_NOT_FIT_OR_SCORE'
                assert saved['prefix_checks']==6 and saved['heldout_truth_loaded'] is False
                assert saved['model_fit']==saved['performance_evaluated']==False
                assert set(saved['candidate_matrices'])==set(reg['family_map'])
                for cid,signature in saved['candidate_matrices'].items():
                    assert signature['columns']==FULL_R3+reg['family_map'][cid]['additional_ET_columns']
                    assert all(len(signature[k])==64 and set(signature[k])<=set('0123456789abcdef') for k in ['train_sha256','query_sha256'])
                # This verifies saved preparation metadata only. The model runner must reconstruct matrices.
                files[name]=sha(path)
                print(f'Preparation receipt resumed {name}; no model reuse',flush=True)
                continue
            started=time.monotonic()
            loader={'status':'REGISTERED','train_ids':ordered_train,'query_ids':ordered_query,
                    'gap_ids_REMOVE_INPUT_AND_BOTH_LABELS':fold['input_forbidden_ids']}
            ctx=BLKContext(loader)
            assert not (set(ctx.reference_labels)&ctx.query_ids)
            _,tr,calendar=prepare_reference(ctx)
            q=prepare_query(ctx,ordered_query,calendar)
            assert tr.row_id.tolist()==ordered_train and q.row_id.tolist()==ordered_query
            domain,selected,_=build_domain(dataframe({**ctx.reference_inputs,**ctx._query}))
            assert set(domain.index)==ctx.train_ids|ctx.query_ids and len(domain.columns)==1187
            tb=tr[FULL_R3].copy();tb.index=tr.row_id
            qb=q[FULL_R3].copy();qb.index=q.row_id
            signatures={}
            for cid,columns in selected.items():
                extra=[c for c in columns if c not in FULL_R3]
                assert extra==reg['family_map'][cid]['additional_ET_columns']
                tx=tb.join(domain[extra]);qx=qb.join(domain[extra])
                signatures[cid]={'columns':list(tx.columns),'train_sha256':matrix_sha(tx),
                    'query_sha256':matrix_sha(qx),'all_missing_train_columns':[c for c in tx if tx[c].isna().all()]}
            checks=0
            qindex=q.set_index('row_id')
            for rid in [ordered_query[0],ordered_query[len(q)//2],ordered_query[-1]]:
                short,again,_=build_domain(dataframe(ctx.query_prefix(rid)))
                assert again==selected
                np.testing.assert_allclose(short.loc[rid],domain.loc[rid],atol=0,rtol=0,equal_nan=True)
                single=prepare_query(ctx,[rid],calendar).set_index('row_id')
                np.testing.assert_allclose(single.loc[rid,FULL_R3].to_numpy(float),qindex.loc[rid,FULL_R3].to_numpy(float),atol=0,rtol=0,equal_nan=True)
                checks+=2
            validate_sources(reg)
            result={'status':'PREPARED_FEATURES_NOT_FIT_OR_SCORE','registration_sha256':sha(regpath),
                'validator':fold['validator'],'fold':fold['fold'],'ordered_train_ids':ordered_train,
                'ordered_query_ids':ordered_query,'input_forbidden_ids':fold['input_forbidden_ids'],
                'candidate_matrices':signatures,'prefix_checks':checks,'domain_columns':1187,
                'heldout_truth_loaded':False,'model_fit':False,'performance_evaluated':False,
                'duration_seconds':time.monotonic()-started}
            result['payload_sha256']=digest(result)
            atomic(path,json.dumps(result,ensure_ascii=False,allow_nan=False))
            files[name]=sha(path)
            print(f'Original feature preparation {name} PASS ({len(q)} query rows), no model fit/score',flush=True)
        validate_sources(reg)
        assert len(files)==66
        complete=folder/'complete.json';assert not complete.exists()
        atomic(complete,json.dumps({'status':'ORIGINAL66_FEATURES_PREPARED_NOT_MODELS_VALIDATED',
            'registration_sha256':sha(regpath),'files_sha256':files,'folds':66,'prefix_checks':396,
            'heldout_truth_loaded':False,'model_fit':False,'performance_evaluated':False,
            'limits':['Each fold has three sampled prefix probes, not universal proof',
                      'Model runner must reconstruct each saved matrix and fit all24 candidates across3seeds',
                      'TM111 scoring subset preserved in the original registry; no scoring here']},ensure_ascii=False))
    finally:
        if lock.exists() and json.loads(lock.read_text(encoding='utf-8'))==token:lock.unlink()

if __name__=='__main__':main()
