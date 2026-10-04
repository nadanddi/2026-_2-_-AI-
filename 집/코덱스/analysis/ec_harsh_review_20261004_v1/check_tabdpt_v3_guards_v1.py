"""Extract actual v3 guards; synthetic inputs only, no models or datasets."""
from pathlib import Path
import ast,json,hashlib,copy,sys,types
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[4]
H=ROOT/'집/코덱스/analysis/ec_tabdpt_20261004_v1'
OUT=Path(__file__).resolve().parent
source=H/'run_v3.py';tree=ast.parse(source.read_text(encoding='utf-8'))
oldtree=ast.parse((H/'run_v1.py').read_text(encoding='utf-8'))
sys.path.insert(0,str(ROOT/'집/클로드/research'));import env
import numpy as np
functions={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
scope={'np':np,'RAW_ATOL':1e-6,'FINAL_ATOL':2e-7,'hashlib':hashlib}
exec(compile(ast.Module(body=[functions[n] for n in ['compare','compare_nullable','validate_first','array_sha']],type_ignores=[]),'<actual_v3_guards>','exec'),scope)
cases=[]
def expect(name,call,accepted):
    try:call()
    except (AssertionError,KeyError,TypeError,ValueError):passed=False
    else:passed=True
    assert passed==accepted,(name,passed,accepted)
    cases.append(dict(name=name,accepted=passed))
for name,a,b,okay in [('same_finite',[1.,2.],[1.,2.],True),('same_nan',[1.,np.nan],[1.,np.nan],True),
                      ('all_nan',[np.nan],[np.nan],True),('changed_nan_mask',[1.,np.nan],[1.,2.],False),
                      ('changed_finite',[1.,np.nan],[2.,np.nan],False),('inf_equal',[np.inf],[np.inf],False),
                      ('shape_difference',[[1.]], [1.],False)]:
    expect('nullable_'+name,lambda a=a,b=b:scope['compare_nullable'](a,b),okay)
signature={'synthetic':'unchanged'}
prefix=[dict(farm=f,day=8,hour=h,raw_maxdiff=0.,final_maxdiff=0.) for f in ['F13','F47'] for h in [0,6,12]]
feature=[dict(farm=f,day=8,hour=h,feature_maxdiff=0.) for f in ['F13','F47'] for h in [0,6,12]]
valid=dict(status='PASS',signature=signature,raw_atol=1e-6,final_atol=2e-7,
           raw_errors={c:0. for c in ['repeat','single','reversed','other_query','fresh_fit']},
           scalar_maxdiff=0.,prefix=prefix,feature_causal=feature)
expect('first_valid',lambda:scope['validate_first'](valid,signature),True)
mutations={
    'failed_status':lambda r:r.update(status='FAIL'),
    'changed_signature':lambda r:r.update(signature={'synthetic':'different'}),
    'changed_raw_atol':lambda r:r.update(raw_atol=.1),
    'changed_final_atol':lambda r:r.update(final_atol=.1),
    'missing_raw_key':lambda r:r['raw_errors'].pop('fresh_fit'),
    'extra_raw_key':lambda r:r['raw_errors'].update(extra=0.),
    'raw_over':lambda r:r['raw_errors'].update(repeat=.1),
    'raw_negative':lambda r:r['raw_errors'].update(repeat=-.1),
    'raw_nan':lambda r:r['raw_errors'].update(repeat=np.nan),
    'raw_inf':lambda r:r['raw_errors'].update(repeat=np.inf),
    'scalar_over':lambda r:r.update(scalar_maxdiff=.1),
    'scalar_nan':lambda r:r.update(scalar_maxdiff=np.nan),
    'prefix_missing':lambda r:r['prefix'].pop(),
    'prefix_duplicate':lambda r:r['prefix'].__setitem__(0,copy.deepcopy(r['prefix'][1])),
    'prefix_wrong_hour':lambda r:r['prefix'][0].update(hour=24),
    'prefix_raw_over':lambda r:r['prefix'][0].update(raw_maxdiff=.1),
    'prefix_final_over':lambda r:r['prefix'][0].update(final_maxdiff=.1),
    'prefix_nan':lambda r:r['prefix'][0].update(final_maxdiff=np.nan),
    'feature_missing':lambda r:r['feature_causal'].pop(),
    'feature_duplicate':lambda r:r['feature_causal'].__setitem__(0,copy.deepcopy(r['feature_causal'][1])),
    'feature_wrong_farm':lambda r:r['feature_causal'][0].update(farm='F99'),
    'feature_over':lambda r:r['feature_causal'][0].update(feature_maxdiff=.1),
    'feature_inf':lambda r:r['feature_causal'][0].update(feature_maxdiff=np.inf),
}
for name,change in mutations.items():
    r=copy.deepcopy(valid);change(r)
    expect('first_'+name,lambda r=r:scope['validate_first'](r,signature),False)

# A descriptive scope limit: day values are not checked by validate_first.
wrong_day=copy.deepcopy(valid)
for r in wrong_day['prefix']+wrong_day['feature_causal']:r['day']=999
expect('first_changed_day_scope_limit',lambda:scope['validate_first'](wrong_day,signature),True)

main=functions['main']
presence_assert=next(n for n in ast.walk(main) if isinstance(n,ast.Assert) and isinstance(n.msg,ast.Constant) and n.msg.value=='Partial cell: preserve and stop.')
digest_assert=next(n for n in ast.walk(main) if isinstance(n,ast.Assert) and 'first_audit_sha256' in ast.unparse(n.test))
preparation_assert=next(n for n in ast.walk(main) if isinstance(n,ast.Assert) and isinstance(n.msg,ast.Constant) and n.msg.value=='Prepared inputs/code changed: stop before fitting.')
presence_code=compile(ast.Module(body=[presence_assert],type_ignores=[]),'<presence_actual>','exec')
for bits in range(8):
    present=[bool(bits&(1<<i)) for i in range(3)]
    expect(f'partial3_{bits}',lambda present=present:exec(presence_code,{'present':present}),bits in [0,7])
digest_code=compile(ast.Module(body=[digest_assert],type_ignores=[]),'<digest_actual>','exec')
digest='synthetic_bytes_sha'
for same in [True,False]:
    expect('first_digest_'+str(same),lambda same=same:exec(digest_code,dict(saved={'first_audit_sha256':digest if same else 'changed'},S=types.SimpleNamespace(sha=lambda p:digest),first=None)),same)

prepared=json.loads((H/'preparation_v3.json').read_text(encoding='utf-8'))
assert prepared['status']=='PASS' and prepared['fit_count']==0 and len(prepared['manifest'])==22
assert prepared['dependencies']['run']==hashlib.sha256(source.read_bytes()).hexdigest()
assert all(r['dependencies']==prepared['dependencies'] for r in prepared['manifest'])
class InMemoryPath:
    def __truediv__(self,other):return self
    def read_text(self,encoding=None):return json.dumps(prepared)
prepare_code=compile(ast.Module(body=[preparation_assert],type_ignores=[]),'<prepare_actual>','exec')
expect('prepare_exact',lambda:exec(prepare_code,dict(json=json,H=InMemoryPath(),prepared=prepared)),True)
for key in ['core','season','env','support','adapter','runtime_probe','run']:
    altered=copy.deepcopy(prepared);altered['dependencies'][key]='changed'
    expect('prepare_dependency_'+key,lambda altered=altered:exec(prepare_code,dict(json=json,H=InMemoryPath(),prepared=altered)),False)
for key in ['input_sha256','public_cache_sha256']:
    altered=copy.deepcopy(prepared);altered[key]='changed'
    expect('prepare_'+key,lambda altered=altered:exec(prepare_code,dict(json=json,H=InMemoryPath(),prepared=altered)),False)
for key in ['train_ids','query_ids','train_features','query_features','train_targets','query_targets','baseline_sha256','bounds','features','runtime']:
    altered=copy.deepcopy(prepared);altered['manifest'][0][key]='changed'
    expect('prepare_manifest_'+key,lambda altered=altered:exec(prepare_code,dict(json=json,H=InMemoryPath(),prepared=altered)),False)

# Original R3 provenance assertions extracted from preflight, no cache opening.
provenance_asserts=[n for n in ast.walk(functions['preflight']) if isinstance(n,ast.Assert) and 'original_meta' in ast.unparse(n.test)]
assert len(provenance_asserts)==3
guard_code=compile(ast.Module(body=provenance_asserts,type_ignores=[]),'<R3_provenance_actual>','exec')
runtime=dict(python='3.12.14',numpy='2.5.3',pandas='3.0.1',sklearn='1.9.1',lightgbm='4.7.0')
original_meta=dict(provenance=dict(shared=dict(input_sha256={'train_X.csv':'same_input'},core_sha256='same_core'),environment=runtime))
def r3_check(meta):exec(guard_code,dict(original_meta=meta,input_sha='same_input',deps={'core':'same_core'},runtime=runtime))
expect('R3_provenance_exact',lambda:r3_check(original_meta),True)
bad=copy.deepcopy(original_meta);bad['provenance']['shared']['input_sha256']['train_X.csv']='other'
expect('R3_input_changed',lambda:r3_check(bad),False)
bad=copy.deepcopy(original_meta);bad['provenance']['shared']['core_sha256']='other'
expect('R3_core_changed',lambda:r3_check(bad),False)
for key in runtime:
    bad=copy.deepcopy(original_meta);bad['provenance']['environment'][key]='other'
    expect('R3_runtime_'+key,lambda bad=bad:r3_check(bad),False)

oldfn={n.name:n for n in oldtree.body if isinstance(n,ast.FunctionDef)}
unchanged=[name for name in ['compare','array_sha','ids_sha','runtime_core','first_audit'] if ast.dump(functions[name],include_attributes=False)==ast.dump(oldfn[name],include_attributes=False)]
assert len(unchanged)==5
assert preparation_assert.lineno<next(n.lineno for n in ast.walk(main) if isinstance(n,ast.Import) and any(a.name=='torch' for a in n.names))
result=dict(status='PASS_CODE_GUARDS_SYNTHETIC_ONLY',source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            synthetic_cases=len(cases),cases=cases,unchanged_helpers=unchanged,
            prepared_folds=22,prepared_dependencies=prepared['dependencies'],
            residual_scope_limits=['validate_first checks farm/hour coverage but not the day value; first artifact digest and original execution still bind the generated audit.',
                                   'Feature audit covers 37 core features on six selected prefixes; season causality is inherited, not newly demonstrated.',
                                   'Final oof.csv is rewritten if it exists without fit_audit; an immutable aggregate guard would preserve that interrupted boundary.'],
            real_data_reads=0,weight_reads=0,model_imports=0,model_fits=0,model_predictions=0)
dest=OUT/'tabdpt_v3_guard_check_v1.json';assert not dest.exists()
dest.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result,ensure_ascii=False,indent=2))
