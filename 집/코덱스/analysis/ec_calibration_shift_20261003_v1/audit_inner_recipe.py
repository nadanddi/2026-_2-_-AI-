from pathlib import Path
import sys,json
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;ROOT=H.parents[3]
sys.path.insert(0,str(ROOT/'집/클로드/research'));import env
sys.path.insert(0,str(ROOT/'집/코덱스/analysis/statistical_experiments_20261003_v1'));import support as S
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
lab,core,wv,folds,outer=S.loadec();z=dict(np.load(S.OUT/'E_DIAG10_0_cpu.npz'));i=dict(np.load(S.OUT/'E_DIAG10_0_input.npz'));idx=lab.set_index('row_id');a=idx.reindex(z['inner_train_id']).reset_index();b=idx.reindex(z['row_id']).reset_index();a,b=S.seasonal(a,b,wv)
oldfull=['season' if c=='day' else c for c in core.FULL];oldbase=['season' if c=='day' else c for c in core.BASE];newfull=[c for c in core.FULL if c!='day']+['season'];newbase=[c for c in core.BASE if c!='day']+['season']
assert np.array_equal(i['train_id'],a.row_id) and np.array_equal(i['query_id'],b.row_id)
assert np.array_equal(i['x'],a[oldfull].to_numpy(np.float32),equal_nan=True) and np.array_equal(i['q'],b[oldfull].to_numpy(np.float32),equal_nan=True)
res={};preds={}
with threadpool_limits(limits=2):
 for name,full,base in [('cached_order',oldfull,oldbase),('current_order',newfull,newbase)]:
  models=[(core.et(7),full),(core.lg(7,'tweedie'),base),(make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),MLPRegressor(hidden_layer_sizes=(128,64),alpha=.01,learning_rate_init=.001,max_iter=800,early_stopping=True,n_iter_no_change=25,validation_fraction=.12,random_state=7)),base)]
  p=[]
  for m,c in models:
   if hasattr(m,'steps') and isinstance(m.steps[-1][1],core.ExtraTreesRegressor):m.steps[-1][1].n_jobs=2
   m.fit(a[c],a.sub_ec);p.append(m.predict(b[c]));print(name,c[-1],len(p),'done',flush=True)
  preds[name]=.6*p[0]+.3*p[1]+.1*p[2]
res['old_cache_replay_maxdiff']=float(np.max(np.abs(preds['cached_order']-z['r3_7'])));assert res['old_cache_replay_maxdiff']<1e-9
res['current_vs_cached_r3_maxdiff']=float(np.max(np.abs(preds['current_order']-preds['cached_order'])));res['current_vs_cached_r3_rmsdiff']=float(np.sqrt(np.mean((preds['current_order']-preds['cached_order'])**2)))
res.update(status='PASS',old_FULL=oldfull,current_FULL=newfull,old_BASE=oldbase,current_BASE=newbase,train_days=len(a)//24,valid_days=len(b)//24,limitations=['first nested fold seed7 only','PFN numerical difference not retrained','inner GPU versus current CPU PFN differs too','no candidate score or adopted improvement'])
(H/'inner_recipe_audit_v1.json').write_text(json.dumps(res,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(res,ensure_ascii=False))
