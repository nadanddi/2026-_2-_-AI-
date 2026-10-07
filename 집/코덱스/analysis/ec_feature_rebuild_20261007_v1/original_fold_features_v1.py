"""Fresh original-fold reconstruction, exact prepared-matrix checks; never fit or score."""
from pathlib import Path
import json,hashlib,sys,platform
from blk_baseline_data_v1 import *
from domain_features_v2 import build_domain
from checkpoint_v1 import digest
import sklearn

def matrix_sha(frame):
    values=frame.to_numpy(dtype='<f8',copy=True)
    values[np.isnan(values)]=np.nan
    return hashlib.sha256(values.tobytes()).hexdigest()

def load_fold(validator,fold_number,require_all66=True):
    if not __debug__ or sys.flags.optimize:raise RuntimeError('Python -O prohibited')
    regpath=HERE/'DOMAIN24_original_preparation_registration_v2.json'
    reg=json.loads(regpath.read_text(encoding='utf-8'))
    assert reg['status']=='REGISTERED_ORIGINAL_FEATURE_PREPARATION_NOT_MODEL_FIT'
    assert all(sha(path)==value for path,value in reg['sources_sha256'].items())
    assert platform.python_version()==reg['environment']['python']
    assert np.__version__==reg['environment']['numpy'] and pd.__version__==reg['environment']['pandas']
    assert sklearn.__version__==reg['environment']['sklearn']
    registry=json.loads((HERE/'original_validator_endpoint_registry_v1.json').read_text(encoding='utf-8'))
    assert sha(HERE/'original_validator_endpoint_registry_v1.json')==reg['registry_sha256']
    selected=[f for f in registry['folds'] if f['validator']==validator and f['fold']==fold_number]
    assert len(selected)==1;fold=selected[0]
    folder=HERE/'checkpoints/DOMAIN24_ORIGINAL_PREPARATION_v2'
    name=f'{validator}_fold{fold_number}.json';path=folder/name
    if require_all66:
        complete=json.loads((folder/'complete.json').read_text(encoding='utf-8'))
        assert complete['status']=='ORIGINAL66_FEATURES_PREPARED_NOT_MODELS_VALIDATED'
        assert complete['folds']==66 and complete['prefix_checks']==396
        assert complete['registration_sha256']==sha(regpath)
        expected={f'{f["validator"]}_fold{f["fold"]}.json' for f in registry['folds']}
        assert set(complete['files_sha256'])==expected
        assert all(sha(folder/n)==s for n,s in complete['files_sha256'].items())
    saved=json.loads(path.read_text(encoding='utf-8'))
    payload=dict(saved);signature=payload.pop('payload_sha256')
    assert digest(payload)==signature and saved['registration_sha256']==sha(regpath)
    assert saved['status']=='PREPARED_FEATURES_NOT_FIT_OR_SCORE'
    assert saved['validator']==validator and saved['fold']==fold_number
    assert saved['ordered_train_ids']==fold['ordered_train_ids'] and saved['ordered_query_ids']==fold['ordered_query_ids']
    assert saved['input_forbidden_ids']==fold['input_forbidden_ids']
    assert not saved['heldout_truth_loaded'] and not saved['model_fit'] and not saved['performance_evaluated']
    loader={'status':'REGISTERED','train_ids':fold['ordered_train_ids'],'query_ids':fold['ordered_query_ids'],
            'gap_ids_REMOVE_INPUT_AND_BOTH_LABELS':fold['input_forbidden_ids']}
    ctx=BLKContext(loader)
    assert set(ctx.reference_inputs)==set(ctx.reference_labels)==set(fold['ordered_train_ids'])
    assert not(set(ctx.reference_labels)&(ctx.query_ids|ctx.gap_ids))
    _,tr,calendar=prepare_reference(ctx);q=prepare_query(ctx,fold['ordered_query_ids'],calendar)
    assert tr.row_id.tolist()==fold['ordered_train_ids'] and q.row_id.tolist()==fold['ordered_query_ids']
    domain,selected,_=build_domain(dataframe({**ctx.reference_inputs,**ctx._query}))
    assert len(domain.columns)==saved['domain_columns']==1187
    assert set(domain.index)==ctx.train_ids|ctx.query_ids
    assert set(selected)==set(saved['candidate_matrices'])==set(reg['family_map'])
    tb=tr[FULL_R3].copy();tb.index=tr.row_id
    qb=q[FULL_R3].copy();qb.index=q.row_id
    matrices={}
    for cid,columns in selected.items():
        extra=[c for c in columns if c not in FULL_R3]
        assert extra==reg['family_map'][cid]['additional_ET_columns']
        tx=tb.join(domain[extra]);qx=qb.join(domain[extra])
        actual={'columns':list(tx.columns),'train_sha256':matrix_sha(tx),'query_sha256':matrix_sha(qx),
                'all_missing_train_columns':[c for c in tx if tx[c].isna().all()]}
        assert actual==saved['candidate_matrices'][cid],('Prepared matrix mismatch',validator,fold_number,cid)
        matrices[cid]=(tx,qx)
    assert len(matrices)==24 and all(sha(p)==v for p,v in reg['sources_sha256'].items())
    details={'status':'ORIGINAL_FOLD_MATRICES_FRESH_RECONSTRUCTED_NOT_FIT_OR_SCORE',
             'validator':validator,'fold':fold_number,'preparation_registration_sha256':sha(regpath),
             'preparation_receipt_sha256':sha(path),'loader_source_sha256':sha(__file__),
             'train_rows':len(tr),'query_rows':len(q),'candidate_matrices_checked':24,
             'baseline_train_sha256':matrix_sha(tb),'baseline_query_sha256':matrix_sha(qb),
             'train_label_sha256':hashlib.sha256(tr.sub_ec.to_numpy(dtype='<f8').tobytes()).hexdigest(),
             'heldout_truth_loaded':False,'model_fit':False,'require_all66':require_all66}
    return ctx,tr,q,calendar,domain,matrices,details

if __name__=='__main__':
    # A reconstruction probe is preparation only; model runners must use require_all66=True.
    assert sys.argv[1:]==['--probe','DIAG10','0']
    _,_,_,_,_,_,details=load_fold('DIAG10',0,require_all66=False)
    out=HERE/'DOMAIN24_original_matrix_reconstruction_probe_v1.json';assert not out.exists()
    out.write_text(json.dumps(details,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Original DIAG10fold0 all24 matrices fresh reconstructed and SHA matched; no fit/score')
