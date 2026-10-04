"""Code/manifest and synthetic guard checks only. No real data or models."""
from pathlib import Path
import ast,json,hashlib,sys
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[4]
H=ROOT/'집/코덱스/analysis/ec_tabdpt_20261004_v1'
PREP=H.parent/'ec_tabdpt_preparation_20261004_v1'
OUT=Path(__file__).resolve().parent
paths=[H/'run_v1.py',PREP/'runtime_probe_v3.py',PREP/'adapter_draft_v1.py',
       ROOT/'집/코덱스/analysis/codex_independent/rl_ec_v1/run.py']
trees={p.name:ast.parse(p.read_text(encoding='utf-8')) for p in paths}
hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
names={'ACTS','INDOOR','RAW','BASE','FP','FULL'}
nodes=[]
for node in trees['run.py'].body:
    if isinstance(node,(ast.Assign,ast.AugAssign)):
        target=node.targets[0] if isinstance(node,ast.Assign) else node.target
        if isinstance(target,ast.Name) and target.id in names:nodes.append(node)
core={};exec(compile(ast.Module(body=nodes,type_ignores=[]),'<constants>','exec'),core)
names={'ACTS','INDOOR','FULL38','CONSTRUCTOR','PREDICT','SEEDS','RAW_ATOL','FINAL_ATOL'}
nodes=[n for n in trees['adapter_draft_v1.py'].body if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id in names]
spec={};exec(compile(ast.Module(body=nodes,type_ignores=[]),'<adapter_constants>','exec'),spec)
assert spec['FULL38']==tuple(c for c in core['FULL'] if c!='day')+('season',)
assert len(spec['FULL38'])==len(set(spec['FULL38']))==38
assert spec['PREDICT']==dict(context_size=512,n_ensembles=8,batch_size=8,output_type='mean')
assert spec['CONSTRUCTOR']['context_reduction']=='retrieval' and spec['CONSTRUCTOR']['missing_indicators'] is False
prep=json.loads((H/'preparation_v1.json').read_text(encoding='utf-8'))
assert prep['status']=='PASS' and prep['folds']==len(prep['manifest'])==22 and prep['fit_count']==0
assert len({(r['validator'],r['fold']) for r in prep['manifest']})==22
assert all(tuple(r['features'])==spec['FULL38'] for r in prep['manifest'])
assert (prep['manifest'][0]['validator'],prep['manifest'][0]['fold'])==('DIAG10',0)

# Actual function, synthetic arrays only. NaN equality currently stops the input audit.
sys.path.insert(0,str(ROOT/'집/클로드/research'));import env
import numpy as np
compare=next(n for n in trees['run_v1.py'].body if isinstance(n,ast.FunctionDef) and n.name=='compare')
namespace={'np':np};exec(compile(ast.Module(body=[compare],type_ignores=[]),'<compare_only>','exec'),namespace)
assert namespace['compare']([1.,2.],[1.,2.])==0
try:namespace['compare']([1.,np.nan],[1.,np.nan])
except AssertionError:nan_identical_rejected=True
else:nan_identical_rejected=False
assert nan_identical_rejected

# Replay only the existing first-audit resume statement with an in-memory JSON path.
main=next(n for n in trees['run_v1.py'].body if isinstance(n,ast.FunctionDef) and n.name=='main')
first_if=next(n for n in ast.walk(main) if isinstance(n,ast.If) and ast.unparse(n.test)=='len(files) == 3')
resume=compile(ast.Module(body=[first_if],type_ignores=[]),'<first_resume_only>','exec')
class InMemoryPath:
    def __init__(self,value):self.value=value
    def read_text(self,encoding=None):return json.dumps(self.value)
signature={'synthetic':'same'};accepted=[]
for label,body in [('PASS_missing_checks',dict(status='PASS')),('FAIL_status',dict(status='FAIL')),
                   ('PASS_exceeded_threshold',dict(status='PASS',raw_errors={'repeat':100.},raw_atol=1e-6))]:
    body['signature']=signature
    exec(resume,dict(json=json,files=[None,None,None],first=InMemoryPath(body),signature=signature))
    accepted.append(label)
assert len(accepted)==3

presence=[]
for bits in range(8):
    states=[bool(bits&(1<<i)) for i in range(3)]
    accepted_state=not any(states) or all(states)
    presence.append(dict(present=states,accepted=accepted_state))
assert sum(r['accepted'] for r in presence)==2
provenance=next(n for n in ast.walk(main) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='provenance' for t in n.targets))
keys=[k.arg for k in provenance.value.keywords]
probe=json.loads((PREP/'runtime_probe_result_v3.json').read_text(encoding='utf-8'))
assert probe['status']=='PASS_IMPORT_ONLY_FIT_PREDICT_UNTESTED' and probe['fit']==probe['predict']==probe['data_reads']==0
record=dict(status='PASS_STATIC_AND_SYNTHETIC_FINDINGS_NEED_FIX',features=38,unique_features=38,
            prepared_folds=22,prepared_fit_count=0,registered_predict=spec['PREDICT'],
            identical_nan_feature_arrays_rejected=nan_identical_rejected,
            first_resume_bad_audits_accepted=accepted,three_file_presence_cases=presence,
            provenance_keys=keys,explicit_core_season_env_hash_keys_present=any(k in keys for k in ['core_sha256','season_sha256','env_sha256']),
            stored_import_probe_status=probe['status'],stored_import_source_files=len(probe['source_sha256']),
            source_hashes=hashes,real_dataset_reads=0,weight_reads=0,model_imports=0,model_fits=0,model_predictions=0,
            limitations=['Static/synthetic checks do not establish real-data NaN frequency or runtime prediction invariance.',
                         'Preparation and import results are stored parent evidence, not re-executed here.'])
dest=OUT/'tabdpt_preexecution_check_v1.json';assert not dest.exists()
dest.write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(record,ensure_ascii=False,indent=2))
