"""Windows-safe PFN completion aggregation with unchanged cache producer2."""
import run_original_pfn_cache_v2 as producer
from run_original_pfn_cache_v2 import *
def main():
    if not __debug__ or sys.flags.optimize:raise RuntimeError('Python -O prohibited')
    driverpath=HERE/'DOMAIN24_original_pfn_driver_registration_v3.json'
    driver=json.loads(driverpath.read_text(encoding='utf-8'))
    assert driver['status']=='REGISTERED_PFN_DRIVER_PATH_FIX_NO_MODEL_CHANGE'
    assert driver['driver_sha256']==sha(__file__)
    assert all(sha(p)==v for p,v in driver['sources_sha256'].items())
    assert driver['producer_registration_sha256']==sha(HERE/'DOMAIN24_original_pfn_registration_v2.json')
    assert driver['producer_sha256']==sha(producer.__file__)
    assert driver['context_fits']==264 and driver['models_changed'] is False
    assert Path(feature_module.__file__).resolve()==(HERE/'original_fold_features_v2.py').resolve()
    assert Path(cache_module.__file__).resolve()==(HERE/'blk_pfn_refonly_cache_v3.py').resolve()
    regpath=HERE/'DOMAIN24_original_pfn_registration_v2.json';reg=json.loads(regpath.read_text(encoding='utf-8'))
    assert reg['status']=='REGISTERED_ORIGINAL66_PFN_REFERENCE_ONLY_BEFORE_FIT'
    assert reg['runner_sha256']==sha(producer.__file__) and reg['context_seeds']==[5,6,7,8] and reg['context_fits']==264
    verify_sources(reg)
    rawpath=HERE/'DOMAIN24_original_raw_fit_registration_v3.json';rawreg=json.loads(rawpath.read_text(encoding='utf-8'))
    assert reg['raw_fit_registration_sha256']==sha(rawpath)
    registry=json.loads((HERE/'original_validator_endpoint_registry_v1.json').read_text(encoding='utf-8'))
    assert len(registry['folds'])==66
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    root=HERE/'checkpoints/DOMAIN24_ORIGINAL_PFN_CACHE_v2';root.mkdir(parents=True,exist_ok=True)
    token={'pid':os.getpid(),'start_ticks':start_ticks(os.getpid()),'registration_sha256':sha(regpath)}
    lock=root/'RUN_WRITER_LOCK.json';outputs={}
    with os.fdopen(os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY),'w',encoding='utf-8') as h:json.dump(token,h)
    try:
        for fold in registry['folds']:
            bundle=feature_module.load_production_fold(fold['validator'],fold['fold'],rawreg)
            folder=root/f'{fold["validator"]}_fold{fold["fold"]}';folder.mkdir(parents=True,exist_ok=True)
            for seed in [5,6,7,8]:
                for path in predict_context(reg,rawreg,fold,bundle,seed,folder):outputs[path.relative_to(root).as_posix()]=sha(path)
            summary={'status':'ORIGINAL_FOLD_PFN4_RAW_AND_AUDITS_COMPLETE_NO_SCORE','registration_sha256':sha(regpath),
                'files_sha256':{k:v for k,v in outputs.items() if k.startswith(folder.name+'/')},'heldout_truth_loaded':False}
            assert len(summary['files_sha256'])==8
            path=folder/'complete.json'
            if path.exists():assert json.loads(path.read_text(encoding='utf-8'))==summary
            else:atomic(path,json.dumps(summary,ensure_ascii=False))
        assert len(outputs)==528;verify_sources(reg)
        summary={'status':'ORIGINAL66_PFN264_CONTEXTS_RAW_AND_AUDITS_COMPLETE_NO_SCORE',
            'registration_sha256':sha(regpath),'files_sha256':outputs,'whole_baseline_complete':False,'heldout_truth_loaded':False}
        path=root/'complete.json'
        if path.exists():assert json.loads(path.read_text(encoding='utf-8'))==summary
        else:atomic(path,json.dumps(summary,ensure_ascii=False))
    finally:
        if lock.exists() and json.loads(lock.read_text(encoding='utf-8'))==token:lock.unlink()
if __name__=='__main__':main()

