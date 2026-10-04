"""Synthetic-only probe of leaf setter versus cached training scores.
Predeclared: add .125 to every leaf of one synthetic tree; compare prediction
and cached training output before/after. No EC inputs, labels, or scores.
"""
from pathlib import Path
import sys,json,hashlib
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;ROOT=H.parents[3]
sys.path.insert(0,str(ROOT/'집/클로드/research'));import env
import numpy as np,lightgbm as lgb
assert lgb.__version__=='4.7.0'
x=np.arange(80,dtype=float).reshape(-1,1)
y=.2+np.arange(80,dtype=float)/80
params=dict(objective='tweedie',tweedie_variance_power=1.5,learning_rate=.03,
 num_leaves=4,min_data_in_leaf=10,num_threads=1,lambda_l2=1.,verbosity=-1,
 deterministic=True,force_col_wise=True,seed=7)
b=lgb.train(params,lgb.Dataset(x,label=y),num_boost_round=1,keep_training_booster=True)
before_raw=b.predict(x,raw_score=True)
before_cache=b._Booster__inner_predict(data_idx=0).copy()
leaves=[]
def visit(node):
    if 'leaf_index' in node:leaves.append(node['leaf_index'])
    else:visit(node['left_child']);visit(node['right_child'])
visit(b.dump_model()['tree_info'][0]['tree_structure'])
assert len(leaves)>1
for leaf in leaves:b.set_leaf_output(0,leaf,b.get_leaf_output(0,leaf)+.125)
after_raw=b.predict(x,raw_score=True)
b._Booster__is_predicted_cur_iter[0]=False
after_cache=b._Booster__inner_predict(data_idx=0).copy()
pred_error=float(np.max(np.abs(after_raw-before_raw-.125)))
cache_delta=float(np.max(np.abs(after_cache-before_cache)))
assert pred_error<1e-12 and cache_delta==0
result=dict(status='PASS_SYNTHETIC_CACHE_DIVERGENCE',rows=80,trees=1,
 leaf_count=len(leaves),added_raw_score=.125,inference_delta_error=pred_error,
 cached_training_output_delta=cache_delta,ec_reads=0,ec_fit=0,ec_score=0,
 synthetic_fit=1,lightgbm_version=lgb.__version__,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
 limitation='One synthetic tree; private cache accessor; does not test a custom optimizer or EC improvement.')
with (H/'leaf_score_cache_probe_v2.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
print(json.dumps(result))

