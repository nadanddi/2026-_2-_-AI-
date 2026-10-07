"""All66 original folds: R3 nine members and domain ET24x3; no PFN/postprocess/truth score."""
from pathlib import Path
import json,os,time,hashlib,sys,platform
import original_fold_features_v2 as features_module
import blk_r3_baseline_v1 as model_module
from blk_baseline_data_v1 import *
from domain_features_v2 import build_domain
from threadpoolctl import threadpool_limits
from checkpoint_v1 import atomic,digest
from checkpoint_v2 import start_ticks
import sklearn,lightgbm

def verify_sources(reg):
    assert all(sha(p)==v for p,v in reg['source_sha256'].items())

def prefix_matrices(ctx,q,calendar,matrices,family_map):
    ids=q.row_id.tolist();indices=sorted(set([0,len(ids)//2,len(ids)-1]));result={}
    for i in indices:
        rid=ids[i];packet=ctx.query_prefix(rid)
        forbidden=set(ctx.query_ids)-set(packet);saved={k:ctx._query[k] for k in forbidden}
        try:
            for k in forbidden:ctx._query[k]={c:99999. for c in saved[k]}
            assert ctx.query_prefix(rid)==packet
            short=prepare_query(ctx,[rid],calendar).set_index('row_id')
            domain,_,_=build_domain(dataframe({**ctx.reference_inputs,**packet}))
            full=short[FULL_R3]
            np.testing.assert_allclose(full.to_numpy(float),q.iloc[[i]][FULL_R3].to_numpy(float),atol=0,rtol=0,equal_nan=True)
            px={'BASELINE_ET':full,'BASELINE_LGB':short[BASE_R3],'BASELINE_MLP':short[BASE_R3]}
            for cid,(_,query) in matrices.items():
                px[cid]=full.join(domain[family_map[cid]['additional_ET_columns']]).loc[[rid]]
                assert list(px[cid].columns)==list(query.columns)
                np.testing.assert_allclose(px[cid].to_numpy(float),query.iloc[[i]].to_numpy(float),atol=0,rtol=0,equal_nan=True)
            result[i]=px
        finally:ctx._query.update(saved)
    return result

def fit_one(reg,folder,validator,fold,seed,name,candidate,tx,qx,y,ids,probes,details):
    verify_sources(reg)
    assert details['require_all66'] is True
    assert details['loader_source_sha256']==reg['source_sha256'][str(Path(features_module.__file__).resolve())]
    assert sha(HERE/'checkpoints/DOMAIN24_ORIGINAL_PREPARATION_v2/complete.json')==reg['preparation_complete_sha256']
    signature={'train_matrix_sha256':features_module.matrix_sha(tx),'query_matrix_sha256':features_module.matrix_sha(qx),
               'train_labels_sha256':details['train_label_sha256'],'columns':list(tx.columns)}
    contract={'registration_sha256':sha(HERE/'DOMAIN24_original_raw_fit_registration_v1.json'),
              'validator':validator,'fold':fold,'seed':seed,'member':name,'candidate':candidate,
              'preparation_receipt_sha256':details['preparation_receipt_sha256'],'matrices':signature}
    path=folder/f'{candidate}_seed{seed}.json'
    if path.exists():
        saved=json.loads(path.read_text(encoding='utf-8'))
        assert saved['status']=='ORIGINAL_RAW_MODEL_PREDICTION_PASS_NO_SCORE'
        assert saved['contract']==contract and saved['row_ids']==ids and digest(saved['pred'])==saved['pred_sha256']
        assert len(saved['pred'])==len(ids) and np.isfinite(saved['pred']).all()
        assert not saved['heldout_truth_loaded'] and saved['audit']['prefix_prediction_checks']==len(probes)
        assert saved['audit']['all_errors_max']<=1e-6
        return path
    started=time.monotonic();m=model_module.model(name,seed)
    with threadpool_limits(limits=1):
        m.fit(tx,y)
        if name=='ET':m.steps[-1][1].n_jobs=1
        predicted=np.asarray(m.predict(qx),float)
        reverse=np.asarray(m.predict(qx.iloc[::-1]),float)[::-1]
        indices=np.linspace(0,len(qx)-1,min(32,len(qx)),dtype=int)
        scattered=np.asarray(m.predict(qx.iloc[indices]),float)
        differences=[float(np.max(np.abs(predicted-reverse))),float(np.max(np.abs(predicted[indices]-scattered)))]
        for i,px in probes.items():
            fresh=np.asarray(m.predict(px[candidate]),float)
            differences.append(float(abs(fresh[0]-predicted[i])))
    assert np.isfinite(predicted).all() and max(differences)<=1e-6
    imputer={}
    if name in ['ET','MLP']:
        transformer=m.steps[0][1]
        with np.errstate(invalid='ignore'):
            direct=np.nanmedian(tx.to_numpy(float),axis=0)
        np.testing.assert_allclose(transformer.statistics_,direct,atol=0,rtol=0,equal_nan=True)
        empty=[c for c in tx if tx[c].isna().all()]
        effective=transformer.get_feature_names_out().tolist()
        assert effective==[c for c in tx if c not in empty]
        imputer={'statistics_sha256':hashlib.sha256(transformer.statistics_.tobytes()).hexdigest(),
                 'independent_train_median_sha256':hashlib.sha256(direct.tobytes()).hexdigest(),
                 'all_missing_train_columns':empty,'effective_columns':effective,'fit_reference_only':True}
    verify_sources(reg)
    record={'status':'ORIGINAL_RAW_MODEL_PREDICTION_PASS_NO_SCORE','contract':contract,'row_ids':ids,
            'pred':predicted.tolist(),'pred_sha256':digest(predicted.tolist()),'imputer':imputer,
            'audit':{'prefix_prediction_checks':len(probes),'all_errors_max':max(differences),
                     'reverse_scattered_prefix_max_differences':differences,'fit_rows':len(tx),'query_rows':len(qx)},
            'heldout_truth_loaded':False,'duration_seconds':time.monotonic()-started}
    assert not path.exists();atomic(path,json.dumps(record,ensure_ascii=False,allow_nan=False))
    print(f'{validator}/{fold} {candidate} seed{seed} raw PASS {record["duration_seconds"]:.1f}s',flush=True)
    return path

def main():
    if not __debug__ or sys.flags.optimize:raise RuntimeError('Python -O prohibited')
    assert Path(features_module.__file__).resolve()==(HERE/'original_fold_features_v2.py').resolve()
    assert Path(model_module.__file__).resolve()==(HERE/'blk_r3_baseline_v1.py').resolve()
    regpath=HERE/'DOMAIN24_original_raw_fit_registration_v1.json'
    reg=json.loads(regpath.read_text(encoding='utf-8'))
    assert reg['status']=='REGISTERED_ORIGINAL66_R3_DOMAIN24_RAW_FIT_BEFORE_FIT'
    assert reg['runner_sha256']==sha(__file__) and reg['seeds']==[47,1414,6464] and reg['candidate_cap']==24
    assert reg['candidate_fit_count']==4752 and reg['baseline_fit_count']==594
    verify_sources(reg)
    expected=reg['environment']
    actual={'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,
            'sklearn':sklearn.__version__,'lightgbm':lightgbm.__version__}
    assert actual==expected
    registry=json.loads((HERE/'original_validator_endpoint_registry_v1.json').read_text(encoding='utf-8'))
    assert len(registry['folds'])==66
    root=HERE/'checkpoints/DOMAIN24_ORIGINAL_RAW_v1';root.mkdir(parents=True,exist_ok=True)
    lock=root/'RUN_WRITER_LOCK.json'
    token={'pid':os.getpid(),'start_ticks':start_ticks(os.getpid()),'registration_sha256':sha(regpath)}
    with os.fdopen(os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY),'w',encoding='utf-8') as handle:json.dump(token,handle)
    outputs={}
    try:
        for fold in registry['folds']:
            validator=fold['validator'];number=fold['fold']
            bundle=features_module.load_production_fold(validator,number,reg)
            ctx,tr,q,calendar,domain,matrices,details=bundle
            assert details['require_all66'] is True
            assert tr.row_id.tolist()==fold['ordered_train_ids'] and q.row_id.tolist()==fold['ordered_query_ids']
            assert set(matrices)==set(reg['family_map'])
            probes=prefix_matrices(ctx,q,calendar,matrices,reg['family_map'])
            ids=q.row_id.tolist();y=tr.sub_ec.to_numpy(float)
            assert np.isfinite(y).all()
            folder=root/f'{validator}_fold{number}';folder.mkdir(parents=True,exist_ok=True)
            for seed in reg['seeds']:
                for member,cols in [('ET',FULL_R3),('LGB',BASE_R3),('MLP',BASE_R3)]:
                    candidate=f'BASELINE_{member}'
                    path=fit_one(reg,folder,validator,number,seed,member,candidate,tr[cols],q[cols],y,ids,probes,details)
                    outputs[str(path.relative_to(root))]=sha(path)
                for cid,(tx,qx) in sorted(matrices.items()):
                    path=fit_one(reg,folder,validator,number,seed,'ET',cid,tx,qx,y,ids,probes,details)
                    outputs[str(path.relative_to(root))]=sha(path)
            complete=folder/'complete.json'
            current={k:v for k,v in outputs.items() if k.startswith(folder.name+'/')}
            assert len(current)==81
            summary={'status':'ORIGINAL_FOLD_R3_DOMAIN_RAW_COMPLETE_NO_SCORE','files_sha256':current,
                     'registration_sha256':sha(regpath),'candidate_fit_count':72,'baseline_fit_count':9,'heldout_truth_loaded':False}
            if complete.exists():assert json.loads(complete.read_text(encoding='utf-8'))==summary
            else:atomic(complete,json.dumps(summary,ensure_ascii=False))
            print(f'{validator}/{number} all81 raw cells complete; PFN/postprocess/score pending',flush=True)
        assert len(outputs)==5346
        verify_sources(reg)
        complete=root/'complete.json';assert not complete.exists()
        atomic(complete,json.dumps({'status':'ORIGINAL66_R3_DOMAIN_RAW_COMPLETE_NO_FULL_BASELINE_OR_SCORE',
                                   'files_sha256':outputs,'registration_sha256':sha(regpath),'folds':66,
                                   'baseline_fit_count':594,'candidate_fit_count':4752,'heldout_truth_loaded':False,
                                   'remaining':['Reference-only cached PFN contexts per66fold','frozen mix/shrink/clip/SG2/full causal gates',
                                                'original TM/P2LOO/EL1 scoring and independent verification']},ensure_ascii=False))
    finally:
        if lock.exists() and json.loads(lock.read_text(encoding='utf-8'))==token:lock.unlink()

if __name__=='__main__':main()
