"""Windows-safe completion aggregation; unchanged registered raw3 fit producer.

Only driver path keys use as_posix. Raw predictions/model contracts retain
registration3 identity. Driver4 has a separate source registration before run.
"""
import run_domain_original_raw_v3 as producer
from run_domain_original_raw_v3 import *
def main():
    if not __debug__ or sys.flags.optimize:raise RuntimeError('Python -O prohibited')
    driverpath=HERE/'DOMAIN24_original_raw_driver_registration_v4.json'
    driver=json.loads(driverpath.read_text(encoding='utf-8'))
    assert driver['status']=='REGISTERED_RAW_DRIVER_PATH_FIX_NO_MODEL_CHANGE'
    assert driver['driver_sha256']==sha(__file__)
    assert all(sha(p)==v for p,v in driver['sources_sha256'].items())
    assert driver['producer_registration_sha256']==sha(HERE/'DOMAIN24_original_raw_fit_registration_v3.json')
    assert driver['producer_sha256']==sha(producer.__file__)
    assert driver['total_fits']==5346 and driver['models_changed'] is False
    assert Path(features_module.__file__).resolve()==(HERE/'original_fold_features_v2.py').resolve()
    assert Path(model_module.__file__).resolve()==(HERE/'blk_r3_baseline_v1.py').resolve()
    regpath=HERE/'DOMAIN24_original_raw_fit_registration_v3.json'
    reg=json.loads(regpath.read_text(encoding='utf-8'))
    assert reg['status']=='REGISTERED_ORIGINAL66_R3_DOMAIN24_RAW_FIT_BEFORE_FIT'
    assert reg['runner_sha256']==sha(producer.__file__) and reg['seeds']==[47,1414,6464] and reg['candidate_cap']==24
    assert reg['candidate_fit_count']==4752 and reg['baseline_fit_count']==594
    verify_sources(reg)
    assert reg['statistics_registration_sha256']==sha(HERE/'DOMAIN24_original_statistics_registration_v2.json')
    assert reg['statistics_draws_sha256']==sha(HERE/'DOMAIN24_original_bootstrap_draws_v2.bin')
    assert reg['statistics_crosscheck_sha256']==sha(HERE/'DOMAIN24_original_statistics_independent_crosscheck_v1.json')
    assert reg['performance_read_before_fit'] is False and reg['adoption_permitted'] is False
    expected=reg['environment']
    actual={'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,
            'sklearn':sklearn.__version__,'lightgbm':lightgbm.__version__}
    assert actual==expected
    registry=json.loads((HERE/'original_validator_endpoint_registry_v1.json').read_text(encoding='utf-8'))
    assert len(registry['folds'])==66
    root=HERE/'checkpoints/DOMAIN24_ORIGINAL_RAW_v3';root.mkdir(parents=True,exist_ok=True)
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
                    outputs[path.relative_to(root).as_posix()]=sha(path)
                for cid,(tx,qx) in sorted(matrices.items()):
                    path=fit_one(reg,folder,validator,number,seed,'ET',cid,tx,qx,y,ids,probes,details)
                    outputs[path.relative_to(root).as_posix()]=sha(path)
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
        complete=root/'complete.json'
        summary={'status':'ORIGINAL66_R3_DOMAIN_RAW_COMPLETE_NO_FULL_BASELINE_OR_SCORE',
                                   'files_sha256':outputs,'registration_sha256':sha(regpath),'folds':66,
                                   'baseline_fit_count':594,'candidate_fit_count':4752,'heldout_truth_loaded':False,
                                   'remaining':['Reference-only cached PFN contexts per66fold','frozen mix/shrink/clip/SG2/full causal gates',
                                                'original TM/P2LOO/EL1 scoring and independent verification']}
        if complete.exists():assert json.loads(complete.read_text(encoding='utf-8'))==summary
        else:atomic(complete,json.dumps(summary,ensure_ascii=False))
    finally:
        if lock.exists() and json.loads(lock.read_text(encoding='utf-8'))==token:lock.unlink()

if __name__=='__main__':main()

