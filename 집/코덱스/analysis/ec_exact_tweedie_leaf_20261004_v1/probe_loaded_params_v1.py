"""Synthetic one-tree model: verify loaded-param canonical names; EC fit0."""
from pathlib import Path
import sys,os,json,ctypes
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;ROOT=H.parents[3];OUT=ROOT/'집/코덱스/local'/H.name
sys.path.insert(0,str(ROOT/'집/클로드/research'));import env
import numpy as np,pandas as pd,lightgbm as lgb,lightgbm.basic as basic
build=json.loads((H/'build_result_v3.json').read_text());dll=Path(build['dll'])
handles=[os.add_dll_directory(str(OUT/'toolchain_v1/mingw64/bin')),os.add_dll_directory(str(dll.parent))]
basic._LIB=ctypes.cdll.LoadLibrary(str(dll));basic._LIB.LGBM_GetLastError.restype=ctypes.c_char_p
p=json.loads((H/'preparation_v3.json').read_text())['params']['7'];p['n_estimators']=1;p['objective']='tweedie_exact_leaf'
x=pd.DataFrame({'x':np.arange(120,dtype=float)});y=np.exp(np.linspace(-1,1,120))
model=lgb.LGBMRegressor(**p).fit(x,y)
text=model.booster_.model_to_string();assert '[num_iterations: 1]' in text
text=text.replace('[num_iterations: 1]','[num_iterations: 800]')
loaded=lgb.Booster(model_str=text)
expected=dict(objective='tweedie_exact_leaf',num_iterations=800,learning_rate=.03,bagging_fraction=.8,bagging_freq=1,feature_fraction=.8,lambda_l2=1,lambda_l1=0,tweedie_variance_power=1.5,num_leaves=31,min_data_in_leaf=40,seed=7,num_threads=4,deterministic=True,force_col_wise=True)
for k,v in expected.items():assert loaded.params[k]==v,(k,loaded.params.get(k),v)
assert loaded.num_trees()==1 and loaded.feature_name()==['x']
with (H/'loaded_params_probe_v1.json').open('x',encoding='utf-8') as f:json.dump(dict(status='PASS_SYNTHETIC_ONLY',params=loaded.params,expected=expected,trees=loaded.num_trees(),feature_names=loaded.feature_name(),synthetic_fit=1,actual_ec_fit=0,real_score=0,fixture_num_iterations_text_change='1 to 800; only serialization schema tested'),f,indent=2)
print('LOADED_PARAMS_SCHEMA_PASS_ECFIT0')
