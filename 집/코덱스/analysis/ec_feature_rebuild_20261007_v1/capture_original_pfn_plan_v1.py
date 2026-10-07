"""Metadata-only original PFN context roster and actual CPU runtime bindings."""
from pathlib import Path
import importlib.util,json,hashlib,platform
HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
path=HERE/'run_original_pfn_cache_v1.py'
spec=importlib.util.spec_from_file_location('prospective_original_pfn',path)
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
assert r.torch.version.cuda is None and '+cpu' in r.torch.__version__
registry_path=HERE/'original_validator_endpoint_registry_v1.json'
registry=json.loads(registry_path.read_text(encoding='utf-8'))
assert len(registry['folds'])==66
contexts={}
for fold in registry['folds']:
    train=fold['ordered_train_ids'];query=set(fold['ordered_query_ids']);gap=set(fold['input_forbidden_ids'])
    assert len(train)==len(set(train)) and set(train).isdisjoint(query|gap)
    name=f'{fold["validator"]}_fold{fold["fold"]}'
    contexts[name]={}
    for seed in [5,6,7,8]:
        ix=r.np.random.default_rng(seed).choice(len(train),size=2000,replace=False)
        ids=[train[int(i)] for i in ix]
        assert len(ids)==len(set(ids))==2000 and set(ids)<=set(train)
        contexts[name][str(seed)]=ids
runtime=r.runtime_paths()
parent_path=HERE/'checkpoints/BLK_PFN_CPU_REFONLY_v3/registration.json'
parent=json.loads(parent_path.read_text(encoding='utf-8'))
assert parent['columns']==r.FULL and len(r.FULL)==38
assert parent['fit_mode']=='fit_with_cache' and parent['kv_cache_precision']=='auto'
assert sha(parent['weights_path'])==parent['weights_sha256']
pins={p:sha(p) for p in set(runtime.values())|{str(path),str(Path(__file__)),str(registry_path),str(parent_path),parent['weights_path']}}
result={'status':'ORIGINAL_PFN_CONTEXTS_AND_RUNTIME_PLAN_CAPTURED_NO_FIT_REGISTRATION',
        'runner_sha256':sha(path),'code_sha256':sha(__file__),'registry_sha256':sha(registry_path),
        'source_sha256':pins,'runtime_module_paths':runtime,'context_seeds':[5,6,7,8],
        'context_fits':264,'context_rows':2000,'contexts':contexts,'columns':r.FULL,
        'weights_path':parent['weights_path'],'weights_sha256':parent['weights_sha256'],
        'fit_mode':'fit_with_cache','kv_cache_precision':'auto','model_version':'V2',
        'n_estimators':4,'precision':'float32','minimum_inference_rows':8,'atol':1e-6,'rtol':0,
        'environment':{'python':platform.python_version(),'torch':r.torch.__version__,'tabpfn':r.tabpfn.__version__},
        'CPU_policy':{'torch_threads':4,'torch_interop':1,'threadpool_limit':4},
        'model_fits':0,'query_predictions':0,'target_values_read':False,
        'limits':['Input/label matrices not loaded; actual fit contract must recompute them perfold',
                  'Preparation66 and rawfitregistration3 still required before PFN fit registration']}
with (HERE/'DOMAIN24_original_pfn_runtime_plan_v1.json').open('x',encoding='utf-8') as h:
    json.dump(result,h,ensure_ascii=False,indent=2,allow_nan=False)
print('Original PFN264 context rosters + CPU bindings captured; no fit/target values or fit registration')
