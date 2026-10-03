from prepare import *
from sklearn.linear_model import LinearRegression,Ridge
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPRegressor
from lightgbm import LGBMRegressor
from resid_reset_features import PHYSICS_COLUMNS
class Temp:
 def fit(self,tr,ct,phc,w,seed):
  self.ct=ct;self.phc=phc;S.cold_v5.SEED=seed
  self.imp=SimpleImputer(strategy='median').fit(tr[phc]);self.phys=LinearRegression().fit(self.imp.transform(tr[phc]),tr.sub_temp,sample_weight=w)
  b=self.phys.predict(self.imp.transform(tr[phc]));self.res=S.cold_v5.lgbh().fit(tr[ct],tr.sub_temp-b,sample_weight=w)
  self.ridge=S.cold_v5.ridge().fit(tr[ct],tr.sub_temp,ridge__sample_weight=w);self.nys=S.cold_v5.nys().fit(tr[ct],tr.sub_temp,ridge__sample_weight=w)
  self.lin=make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler(),Ridge(alpha=100)).fit(tr[PHYSICS_COLUMNS],tr.sub_temp,ridge__sample_weight=w)
  self.tree=LGBMRegressor(n_estimators=220,learning_rate=.035,num_leaves=12,max_depth=-1,min_child_samples=100,reg_lambda=15,verbosity=-1,n_jobs=4,random_state=726+(seed==101)).fit(tr[S.FEATURE_COLUMNS],tr.sub_temp-self.lin.predict(tr[PHYSICS_COLUMNS]),sample_weight=w)
  return self
 def predict(self,q):
  phys=self.phys.predict(self.imp.transform(q[self.phc]));residual=self.res.predict(q[self.ct]);rid=self.ridge.predict(q[self.ct]);nys=self.nys.predict(q[self.ct]);cl=self.lin.predict(q[PHYSICS_COLUMNS]);cr=self.tree.predict(q[S.FEATURE_COLUMNS])
  return dict(BASE=.65*(phys+residual)+.25*rid+.10*nys,CODEX=cl+cr,BASE_physics=phys,BASE_residual=residual,BASE_ridge=rid,BASE_nys=nys,CODEX_physics=cl,CODEX_residual=cr)
class ECModel:
 def fit(self,tr,core,seed):
  self.full=['season' if c=='day' else c for c in core.FULL];self.base=['season' if c=='day' else c for c in core.BASE];self.core=core
  self.models=[core.et(seed),core.lg(seed,'tweedie'),make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),MLPRegressor(hidden_layer_sizes=(128,64),alpha=.01,learning_rate_init=.001,max_iter=800,early_stopping=True,n_iter_no_change=25,validation_fraction=.12,random_state=seed))]
  for m,c in zip(self.models,[self.full,self.base,self.base]):m.fit(tr[c],tr.sub_ec)
  return self
 def predict(self,q):
  p=[m.predict(q[c]) for m,c in zip(self.models,[self.full,self.base,self.base])]
  return dict(raw_r3=.6*p[0]+.3*p[1]+.1*p[2],raw_ET=p[0],raw_LGB=p[1],raw_MLP=p[2])
def groups(columns,target):
 out={}
 for c in columns:
  if target=='EC':
   g=next((v for v in ['in_temp','in_hum','in_co2']+S.loadcore().ACTS if c==v or c.startswith(v+'_')),None)
   if g is None:g='season' if c=='season' else 'clock'
  else:
   if 'in_temp' in c:g='indoor_temperature'
   elif 'out_temp' in c:g='outdoor_temperature'
   elif any(s in c for s in ['in_hum','out_hum','in_co2','dew','vpd']):g='humidity_CO2'
   elif 'out_rad' in c or 'out_wspd' in c:g='radiation_wind'
   elif 'act_heating' in c:g='heating'
   elif c.startswith('act_'):g='other_actuators'
   elif c in ['day','hour','farm_id','sin','cos','second','hr_sin','hr_cos','midnight','t'] or c.startswith('farm_'):g='clock_farm_period'
   else:g='derived_relations'
  out.setdefault(g,[]).append(c)
 return out
def finished_ec(p,tr,q,core):return np.clip(core.shrink(p,q),tr.sub_ec.min(),tr.sub_ec.max())
