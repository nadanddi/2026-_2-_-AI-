"""Capture model constructors and actual CPU scientific bindings, with no fitting."""
from pathlib import Path
import importlib.util,json,hashlib,platform,inspect
HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
path=HERE/'run_domain_original_raw_v3.py'
spec=importlib.util.spec_from_file_location('original_raw_contract',path)
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
runtime=r.runtime_paths();contracts={};classes={}
for seed in [47,1414,6464]:
    for name in ['ET','LGB','MLP']:
        model=r.model_module.model(name,seed)
        contracts[f'{name}_seed{seed}']=r.model_contract(model)
        steps=model.steps if hasattr(model,'steps') else [('estimator',model)]
        for _,step in steps:
            classes[type(step).__module__+'.'+type(step).__name__]=str(Path(inspect.getfile(type(step))).resolve())
pins={p:sha(p) for p in set(runtime.values())|set(classes.values())|{str(path)}}
env={'python':platform.python_version(),'numpy':r.np.__version__,'pandas':r.pd.__version__,
     'sklearn':r.sklearn.__version__,'lightgbm':r.lightgbm.__version__}
result={'status':'ACTUAL_ORIGINAL_MODEL_RUNTIME_CONSTRUCTORS_CAPTURED_NO_FIT',
        'runner_sha256':sha(path),'code_sha256':sha(__file__),'runtime_module_paths':runtime,
        'model_class_paths':classes,'source_sha256':pins,'environment':env,'model_contracts':contracts,
        'model_fits':0,'query_predictions':0,'target_values_read':False,'device':'CPU'}
with (HERE/'DOMAIN24_original_runtime_contract_v1.json').open('x',encoding='utf-8') as h:
    json.dump(result,h,ensure_ascii=False,indent=2,allow_nan=False)
print('Actual runtime + nine model constructor contracts captured; fit/prediction/target values0')
