from pathlib import Path
import ast
HERE=Path(__file__).resolve().parent
source=(HERE/'run_original_pfn_cache_v1.py').read_text(encoding='utf-8')
source=source.replace('from checkpoint_v2 import start_ticks','''from checkpoint_v2 import start_ticks
import original_pfn_cache_rules_v1 as rules_module
import blk_baseline_data_v1 as baseline_module
import blk_context_v1 as context_module
import blk_pfn_minbatch_v1 as adapter_module
import checkpoint_v1 as checkpoint_module
import checkpoint_v2 as checkpoint2_module
import env_extra as extra_module
import threadpoolctl as pool_module''')
old="modules={'features':feature_module,'cache_template':cache_module,'torch':torch,'tabpfn':tabpfn}"
new="""modules={'features':feature_module,'cache_template':cache_module,'torch':torch,'tabpfn':tabpfn,
        'rules':rules_module,'baseline':baseline_module,'context':context_module,'adapter':adapter_module,
        'checkpoint':checkpoint_module,'checkpoint2':checkpoint2_module,'env':env,'env_extra':extra_module,
        'threadpoolctl':pool_module,'numpy':np,'pandas':pd} """
assert source.count(old)==1;source=source.replace(old,new)
start=source.index("    assert set(audit['checks'])==CHECK_KEYS",source.index('def validate_saved'))
end=source.index('\ndef predict_context',start)
source=source[:start]+"    assert audit['auditor_sha256']==sha(rules_module.__file__)\n    assert audit['code_sha256']==sha(__file__)\n    rules_module.audit_metadata(audit,len(ids),CHECK_KEYS)\n"+source[end:]
anchor='def predict_context(reg,rawreg,fold,bundle,seed,folder):'
newfn='''def cache_fingerprint(model):
    for cache in model.executor_.kv_caches:
        assert all(t.device.type=='cpu' for t in cache.feature_cache.values())
        assert cache.test_y_embedding.device.type=='cpu'
        for layer in cache.kv.values():
            assert layer.key.device.type==layer.value.device.type=='cpu'
    result=cache_module.cache_fingerprint(model)
    for cache in result:
        for value in cache['feature_statistics'].values():value['device']='cpu'
        cache['target_embedding']['device']='cpu'
        for layer in cache['kv'].values():
            for value in layer.values():value['device']='cpu'
    rules_module.fingerprint(result)
    return result

'''
assert source.count(anchor)==1;source=source.replace(anchor,newfn+anchor)
source=source.replace('cache_module.cache_fingerprint(model);trace.phase','cache_fingerprint(model);trace.phase')
source=source.replace('after=cache_module.cache_fingerprint(model)','after=cache_fingerprint(model)')
source=source.replace("'inference_precision':str(model.inference_precision)}","'inference_precision':str(model.inference_precision),'memory_saving_mode':model.memory_saving_mode}")
source=source.replace("audit={'status':status,'contract':contract", "audit={'status':status,'code_sha256':sha(__file__),'auditor_sha256':sha(rules_module.__file__),'contract':contract")
source=source.replace('DOMAIN24_original_pfn_registration_v1.json','DOMAIN24_original_pfn_registration_v2.json')
source=source.replace('DOMAIN24_ORIGINAL_PFN_CACHE_v1','DOMAIN24_ORIGINAL_PFN_CACHE_v2')
ast.parse(source)
with (HERE/'run_original_pfn_cache_v2.py').open('x',encoding='utf-8') as h:h.write(source)
print('Original PFN runner2 strict cache/trace/import contracts created; no fit registration or execution')
