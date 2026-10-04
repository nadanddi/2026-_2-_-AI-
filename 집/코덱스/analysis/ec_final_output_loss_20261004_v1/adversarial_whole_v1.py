"""Synthetic-only adversarial checks; no datasets or model fitting."""
from pathlib import Path
import ast, importlib.util, json, math, hashlib, copy
H=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('independent_math',H/'verification_math_v4.py')
M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)
np,pd=M.np,M.pd
ns=dict(Path=Path,json=json,math=math,np=np,pd=pd,M=M)
tree=ast.parse((H/'verify_full_v3.py').read_text(encoding='utf-8'))
nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in {'readj','first_check'}]
exec(compile(ast.Module(body=nodes,type_ignores=[]),'synthetic_extracted','exec'),ns)
tests=[]
def reject(name,fn):
    try:fn()
    except (AssertionError,ValueError,KeyError,IndexError):tests.append(dict(name=name,rejected=True));return
    raise AssertionError('corruption accepted: '+name)
rng=np.random.default_rng(1122);a=rng.normal(size=(31,14));q=rng.normal(size=(7,14))
a[:,0]=np.nan;q[:,0]=np.nan;a[0,1]=np.nan;q[0,1]=np.nan
med=np.nanmedian(a,axis=0);med=np.where(np.isnan(med),0.,med)
clean=np.where(np.isnan(a),med,a);mean=clean.mean(0);var=((clean-mean)**2).mean(0);scale=np.sqrt(var);scale[scale==0]=1
features=['x'+str(k) for k in range(13)]+['season']
z=dict(feature_names=np.array(features),imputer_statistics=med,scaler_mean=mean,scaler_var=var,scaler_scale=scale,scaler_n_samples_seen=np.array(31))
for k,ws,bs in [(0,(64,14),(64,)),(2,(32,64),(32,)),(4,(1,32),(1,))]:
    z[f'model__{k}.weight']=rng.normal(0,.02,ws);z[f'model__{k}.bias']=rng.normal(0,.02,bs)
pred=M.replay_checkpoint(z,a,q,features)
assert np.isfinite(pred).all()
mutations={
 'checkpoint_missing_key':lambda x:x.pop('model__2.bias'),
 'feature_order':lambda x:x.update(feature_names=np.array(features[::-1])),
 'weight_dtype':lambda x:x.update({'model__0.weight':x['model__0.weight'].astype('float32')}),
 'weight_nonfinite':lambda x:x['model__4.bias'].__setitem__(0,np.nan),
 'scaler_variance':lambda x:x['scaler_var'].__setitem__(2,x['scaler_var'][2]+.01),
 'scaler_zero':lambda x:x['scaler_scale'].__setitem__(2,0),
 'training_count':lambda x:x.update(scaler_n_samples_seen=np.array(32))}
for name,mutate in mutations.items():
    bad=copy.deepcopy(z);mutate(bad);reject(name,lambda:M.replay_checkpoint(bad,a,q,features))
f=pd.DataFrame(dict(farm=['F13']*24+['F47']*24,day=[1]*24+[2]*24,hour=list(range(24))*2))
raw=rng.normal(size=48);base=M.scalar_final(f,raw,-1,1)
for h in (0,6,12):
    mask=(f.farm=='F13')&(f.hour<=h);changed=raw.copy();changed[~mask]+=999
    M.near(M.scalar_final(f,changed,-1,1)[mask],base[mask])
reject('bounds_reversed',lambda:M.scalar_final(f,raw,2,1))
reject('bounds_nonfinite',lambda:M.scalar_final(f,raw,0,np.inf))
badf=f.copy();badf.loc[1,'hour']=0
reject('duplicate_group_hour',lambda:M.scalar_final(badf,raw,-1,1))
sig=dict(seed=7,initial_weights_sha256={'7':'syntheticinit'},dependencies={'run':'sourcepin'})
first=dict(status='PASS',signature=sig,atol=1e-12,epochs=400,initial_weights_sha256='syntheticinit',all_gradients_finite=True,all_parameters_finite=True,errors={k:0. for k in ['repeat','fresh','reversed','single','other_query']},zero_query_baseline_maxdiff=0.,zero_inner_baseline_maxdiff=0.,scalar_maxdiff=0.,fresh_loss_trace_maxdiff=0.,loss_initial=1.,loss_final=.5,limitations='synthetic',prefix=[dict(farm=farm,day=day,hour=h,rows=h+1,raw_delta_maxdiff=0.,final_maxdiff=0.) for farm,day in [('F13',1),('F47',2)] for h in (0,6,12)])
ns['first_check'](first,sig,f)
for name,mutate in {
 'source_signature_mismatch':lambda x:x['signature']['dependencies'].update(run='changed'),
 'first_signature_seed':lambda x:x['signature'].update(seed=101),
 'first_future_prefix':lambda x:x['prefix'][0].update(day=3),
 'first_nonfinite':lambda x:x['errors'].update(fresh=float('nan')),
 'first_tolerance_relaxed':lambda x:x.update(atol=1e-10),
 'first_missing_key':lambda x:x.pop('all_gradients_finite'),
 'first_finite_flag':lambda x:x.update(all_parameters_finite=False)}.items():
    bad=copy.deepcopy(first);mutate(bad);reject(name,lambda:ns['first_check'](bad,sig,f))
for name,content in [('duplicate_json','{"a":1,"a":2}'),('nan_json','{"a":NaN}')]:
    p=H/(name+'_fixture_v1.json')
    with p.open('x',encoding='utf-8') as out:out.write(content)
    reject(name,lambda:ns['readj'](p))
result=dict(status='PASS_SYNTHETIC_ONLY',rejections=tests,all_nan_column_supported=True,prefix_future_perturbation_pass=True,actual_fit=0,actual_predict=0,actual_score=0,actual_data_reads=0,source_sha256={p.name:M.sha(p) for p in [Path(__file__),H/'verify_full_v3.py',H/'verification_math_v4.py']},limitations='Source mismatch fixture exercises exact signature comparison, not a full actual runtime/manifest traversal. Fresh-native evidence is recorded audit verification, not fresh training replay.')
with (H/'adversarial_whole_result_v1.json').open('x',encoding='utf-8') as out:json.dump(result,out,indent=2)
print('PASS_SYNTHETIC_ONLY',len(tests),json.dumps(result['source_sha256']))
