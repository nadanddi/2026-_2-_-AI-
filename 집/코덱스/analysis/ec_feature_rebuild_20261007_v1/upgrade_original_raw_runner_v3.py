"""Close actual-label/runtime/model contract gaps before registering any original fit."""
from pathlib import Path
import ast
HERE=Path(__file__).resolve().parent
source=(HERE/'run_domain_original_raw_v2.py').read_text(encoding='utf-8')
source=source.replace('import sklearn,lightgbm','''import sklearn,lightgbm
import blk_context_v1 as context_module
import blk_baseline_data_v1 as baseline_module
import domain_features_v2 as domain_module
import checkpoint_v1 as checkpoint_module
import checkpoint_v2 as checkpoint2_module
import threadpoolctl as threadpool_module''')
anchor='def canonical_numeric_sha(values):'
insert='''def runtime_paths():
    modules={'features':features_module,'models':model_module,'context':context_module,
        'baseline':baseline_module,'domain':domain_module,'checkpoint':checkpoint_module,
        'checkpoint2':checkpoint2_module,'threadpoolctl':threadpool_module,
        'numpy':np,'pandas':pd,'sklearn':sklearn,'lightgbm':lightgbm,'env':env}
    return {name:str(Path(module.__file__).resolve()) for name,module in modules.items()}

def serialize_parameter(value):
    if isinstance(value,(tuple,list)):return [serialize_parameter(v) for v in value]
    if isinstance(value,dict):return {str(k):serialize_parameter(v) for k,v in value.items()}
    if isinstance(value,(float,np.floating)) and not np.isfinite(value):return 'NaN' if np.isnan(value) else str(value)
    if isinstance(value,np.generic):return value.item()
    if value is None or isinstance(value,(bool,int,float,str)):return value
    raise TypeError('Unregistered parameter type: '+str(type(value)))

def model_contract(model):
    steps=model.steps if hasattr(model,'steps') else [('estimator',model)]
    return [{'name':name,'class':type(step).__module__+'.'+type(step).__name__,
             'parameters':serialize_parameter(step.get_params(deep=False))} for name,step in steps]

'''
assert source.count(anchor)==1;source=source.replace(anchor,insert+anchor)
old="    assert all(sha(p)==v for p,v in reg['source_sha256'].items())"
new=old+"\n    assert runtime_paths()==reg['runtime_module_paths']"
assert source.count(old)==1;source=source.replace(old,new)
old="    signature={'train_matrix_sha256'"
new="""    assert len(y)==len(tx) and np.isfinite(y).all()
    assert hashlib.sha256(np.asarray(y,dtype='<f8').tobytes()).hexdigest()==details['train_label_sha256']
    m=model_module.model(name,seed)
    params=model_contract(m)
    assert params==reg['model_contracts'][f'{name}_seed{seed}']
    signature={'train_matrix_sha256'"""
assert source.count(old)==1;source=source.replace(old,new)
old="'preparation_receipt_sha256':details['preparation_receipt_sha256'],'matrices':signature}"
new="'preparation_receipt_sha256':details['preparation_receipt_sha256'],'matrices':signature,\n              'model_contract':params,'runtime_module_paths':runtime_paths()}"
assert source.count(old)==1;source=source.replace(old,new)
source=source.replace('started=time.monotonic();m=model_module.model(name,seed)','started=time.monotonic()')
source=source.replace('DOMAIN24_original_raw_fit_registration_v2.json','DOMAIN24_original_raw_fit_registration_v3.json')
source=source.replace('DOMAIN24_ORIGINAL_RAW_v2','DOMAIN24_ORIGINAL_RAW_v3')
old="    expected=reg['environment']"
new="""    assert reg['statistics_registration_sha256']==sha(HERE/'DOMAIN24_original_statistics_registration_v2.json')
    assert reg['statistics_draws_sha256']==sha(HERE/'DOMAIN24_original_bootstrap_draws_v2.bin')
    assert reg['statistics_crosscheck_sha256']==sha(HERE/'DOMAIN24_original_statistics_independent_crosscheck_v1.json')
    assert reg['performance_read_before_fit'] is False and reg['adoption_permitted'] is False
    expected=reg['environment']"""
assert source.count(old)==1;source=source.replace(old,new)
ast.parse(source)
with (HERE/'run_domain_original_raw_v3.py').open('x',encoding='utf-8') as h:h.write(source)
print('Rawrunner3 source created: actual y/model/runtime/statistics contracts; registration and fit pending')
