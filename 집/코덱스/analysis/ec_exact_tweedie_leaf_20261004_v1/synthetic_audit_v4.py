"""Synthetic only: native renewal, in-bag sums and score-cache checks."""
from pathlib import Path
import os, sys, json, hashlib, ctypes, math
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent; ROOT=H.parents[3]
LOCAL=ROOT/'집/코덱스/local/ec_exact_tweedie_leaf_20261004_v1'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
result={'status':'PENDING','actual_ec_fit':0,'synthetic_fit':0}
try:
 build=json.loads((H/'build_result_v3.json').read_text());assert build['status']=='PASS_BUILD_ONLY'
 dll=Path(build['dll']);assert sha(dll)==build['dll_sha256']
 sys.path.insert(0,str(ROOT/'집/클로드/research'));import env
 import numpy as np, lightgbm as lgb
 import lightgbm.basic as basic
 assert lgb.__version__=='4.7.0'
 original=Path(basic._LIB._name)
 handles=[os.add_dll_directory(str(LOCAL/'toolchain_v1/mingw64/bin')),os.add_dll_directory(str(dll.parent))]
 replacement=ctypes.cdll.LoadLibrary(str(dll));replacement.LGBM_GetLastError.restype=ctypes.c_char_p
 basic._LIB=replacement
 result.update(dll=str(dll),dll_sha256=sha(dll),original_dll=str(original),original_sha256=sha(original))
 rng=np.random.default_rng(77);X=rng.normal(size=(160,4));y=np.exp(.8*X[:,0]+.2*X[:,1])+0.05
 params=dict(objective='tweedie',tweedie_variance_power=1.5,lambda_l2=1.,lambda_l1=0.,learning_rate=.03,num_leaves=4,min_data_in_leaf=8,bagging_fraction=.8,bagging_freq=1,bagging_seed=7,seed=7,feature_fraction=1.,num_threads=2,verbosity=-1)
 os.chdir(LOCAL)
 models={}; records={}
 for mode in ('newton','exact'):
  path=LOCAL/f'synthetic_{mode}_v4.ndjson';assert not path.exists();path.open('x').close()
  os.environ['FARMAI_LGB_AUDIT_PATH']=path.name
  p=dict(params);p['objective']='tweedie' if mode=='newton' else 'tweedie_exact_leaf'
  dataset=lgb.Dataset(X,label=y,free_raw_data=False)
  model=lgb.train(p,dataset,num_boost_round=3,keep_training_booster=True);result['synthetic_fit']+=1
  models[mode]=model;records[mode]=[json.loads(s) for s in path.read_text().splitlines()]
  raw=model.predict(X,raw_score=True)
  model._Booster__is_predicted_cur_iter[0]=False
  cache=model._Booster__inner_predict(data_idx=0)
  assert np.max(np.abs(cache-np.exp(raw)))<=1e-12
  loaded=lgb.Booster(model_str=model.model_to_string())
  assert np.max(np.abs(loaded.predict(X)-np.exp(raw)))<=1e-12
  result[mode+'_cache_max_error']=float(np.max(np.abs(cache-np.exp(raw))))
 os.environ.pop('FARMAI_LGB_AUDIT_PATH')
 bags=lambda mode:[r for r in records[mode] if r['kind']=='bag']
 assert bags('newton')==bags('exact') and len(bags('exact'))==3
 trees=models['exact'].dump_model()['tree_info'];assert len(trees)==3
 groups=[]
 for record in records['exact']:
  if record['kind']=='bag':groups.append([record,[]])
  else:assert groups;groups[-1][1].append(record)
 max_sum=max_root=max_score=0.
 for ti,(bag,leaves) in enumerate(groups):
  ids=[i for leaf in leaves for i in leaf['ids']]
  assert len(ids)==len(set(ids)) and sorted(ids)==sorted(bag['ids'])
  assert len(leaves)==trees[ti]['num_leaves']
  old=np.full(len(y),math.log(float(np.mean(y.astype(np.float32),dtype=np.float64)))) if ti==0 else models['exact'].predict(X,raw_score=True,num_iteration=ti)
  membership=models['exact'].predict(X,pred_leaf=True)[:,ti]
  def collect(node):
   if 'leaf_index' in node:return {node['leaf_index']:node['leaf_value']}
   return collect(node['left_child'])|collect(node['right_child'])
  outputs=collect(trees[ti]['tree_structure'])
  for leaf in leaves:
   ix=np.asarray(leaf['ids']);F=np.asarray(leaf['F']);yy=np.asarray(leaf['y']);w=np.asarray(leaf['w'])
   assert np.array_equal(yy,y.astype(np.float32)[ix].astype(float))
   max_score=max(max_score,float(np.max(np.abs(F-old[ix]))));assert max_score<=1e-12
   A=math.fsum(float(a*b*math.exp(-c/2)) for a,b,c in zip(w,yy,F));B=math.fsum(float(a*math.exp(c/2)) for a,c in zip(w,F))
   max_sum=max(max_sum,abs(A-leaf['A']),abs(B-leaf['B']));assert abs(A-leaf['A'])<=1e-12*max(1,A) and abs(B-leaf['B'])<=1e-12*max(1,B)
   lo,hi=-64.,64.
   for _ in range(120):
    mid=(lo+hi)/2
    if -A*math.exp(-mid/2)+B*math.exp(mid/2)+mid>0:hi=mid
    else:lo=mid
   root=(lo+hi)/2;max_root=max(max_root,abs(root-leaf['root']));assert max_root<=1e-12
   assert abs(leaf['newton']-(A-B)/(.5*(A+B)+1))<=1e-5
   indices=np.unique(membership[ix]);assert len(indices)==1
   expected=.03*root+(old[0] if ti==0 else 0)
   assert abs(outputs[int(indices[0])]-expected)<=1e-12
 result.update(status='PASS_SYNTHETIC_ONLY',trees=3,inbag_same=True,max_sum_error=max_sum,max_root_error=max_root,max_raw_score_error=max_score,audit_sha256={m:sha(LOCAL/f'synthetic_{m}_v4.ndjson') for m in models})
except BaseException as exc:
 result.update(status='FAIL_SYNTHETIC_ONLY',error=repr(exc));raise
finally:
 with (H/'synthetic_result_v4.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
print(json.dumps(result,indent=2))
