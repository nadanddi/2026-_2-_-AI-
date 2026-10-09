"""Frozen high-day classifier features and factories; current prefix only."""
import runtime_v1
import numpy as np,pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
RAW=['out_temp','out_hum','out_rad','out_wspd','in_temp','in_hum','in_co2','act_vent','act_shade','act_thermal','act_heating','act_circfan','act_co2','act_fog']
ACTS=RAW[7:];CONTRASTS=['act_circfan','act_shade','act_thermal'];SEEDS=(8383,1919,7171);EC_THRESHOLD=1.2
COLS=[f'{c}__{kind}' for c in RAW for kind in ('current','h0','mean','zero_share')]+['hour','hr_sin','hr_cos','farm47']+['seal_run','vent_open_hours','first_open_hour','thermal_switches','shade_switches','since_curtain_change','heat_run','co2_hours','vent_max']+[f'{c}__daynight' for c in CONTRASTS]+['daynight_available']
assert len(COLS)==73

def identify(x):
 out=x[['row_id']].copy();z=out.row_id.str.split('_',expand=True);assert z.shape[1]==3
 out['farm']=z[0];out['day']=z[1].astype(int);out['hour']=z[2].astype(int)
 assert out.row_id.is_unique and out.farm.isin(['F13','F47']).all() and out.hour.between(0,23).all()
 return out

def run_known(a,active):
 count=0
 for v in a:
  if not np.isfinite(v):count=None
  elif bool(active(v)):
   if count is not None:count+=1
  else:count=0
 return np.nan if count is None else float(count)

def features(x):
 meta=identify(x);a=meta.merge(x[['row_id']+RAW],on='row_id',validate='one_to_one').sort_values(['farm','day','hour']).reset_index(drop=True);result=[]
 for (farm,day),g in a.groupby(['farm','day'],sort=False):
  hs=g.hour.to_numpy(int);assert np.array_equal(hs,np.arange(len(hs)))
  vals={c:g[c].to_numpy(float) for c in RAW};assert all(not np.isinf(v).any() for v in vals.values())
  for j,(_,r) in enumerate(g.iterrows()):
   h=int(r.hour);z={'row_id':r.row_id,'farm':farm,'day':int(day),'hour':h}
   for c,v in vals.items():
    p=v[:j+1];ok=p[np.isfinite(p)];z.update({f'{c}__current':float(v[j]),f'{c}__h0':float(v[0]),f'{c}__mean':float(ok.mean()) if len(ok) else np.nan,f'{c}__zero_share':float((ok==0).mean()) if len(ok) else np.nan})
   v=vals['act_vent'][:j+1];he=vals['act_heating'][:j+1];co=vals['act_co2'][:j+1];th=vals['act_thermal'][:j+1];sh=vals['act_shade'][:j+1]
   def switches(b):return float((np.diff((b>0).astype(int))!=0).sum()) if np.isfinite(b).all() else np.nan
   ventok=np.isfinite(v).all();where=np.flatnonzero(v>0)
   changes=[];allcurtain=np.isfinite(th).all() and np.isfinite(sh).all()
   if allcurtain:
    changes=np.flatnonzero((np.diff((th>0).astype(int))!=0)|(np.diff((sh>0).astype(int))!=0))+1
   z.update(hour=h,hr_sin=float(np.sin(h/24*2*np.pi)),hr_cos=float(np.cos(h/24*2*np.pi)),farm47=float(farm=='F47'),seal_run=run_known(v,lambda u:u==0),vent_open_hours=float((v>0).sum()) if ventok else np.nan,first_open_hour=float(where[0]) if len(where) and (ventok or np.isfinite(v[:where[0]+1]).all()) else 24. if ventok else np.nan,thermal_switches=switches(th),shade_switches=switches(sh),since_curtain_change=float(h-changes[-1]) if allcurtain and len(changes) else float(h+1) if allcurtain else np.nan,heat_run=run_known(he,lambda u:u>0),co2_hours=float((co>0).sum()) if np.isfinite(co).all() else np.nan,vent_max=float(np.nanmax(v)) if np.isfinite(v).any() else np.nan,daynight_available=float(h>=9))
   for c in CONTRASTS:
    b=vals[c];z[f'{c}__daynight']=float(b[9:min(h,15)+1].mean()-b[:9].mean()) if h>=9 and np.isfinite(b[:min(h,15)+1]).all() else np.nan
   result.append(z)
 out=pd.DataFrame(result);assert set(out.row_id)==set(x.row_id)
 return out.set_index('row_id').loc[x.row_id].reset_index()

def factory(kind,seed=0):
 if kind=='logit':return Pipeline([('imputer',SimpleImputer(strategy='median',keep_empty_features=True)),('scaler',StandardScaler()),('clf',LogisticRegression(C=1,max_iter=2000,random_state=seed))])
 if kind=='et':return Pipeline([('imputer',SimpleImputer(strategy='median',keep_empty_features=True)),('clf',ExtraTreesClassifier(n_estimators=300,max_depth=10,min_samples_leaf=24,max_features=.7,n_jobs=2,class_weight=None,random_state=seed))])
 raise ValueError(kind)

def positive_score(model,x):
 classes=model.classes_;p=model.predict_proba(x)
 return p[:,int(np.flatnonzero(classes==1)[0])] if (classes==1).any() else np.zeros(len(x))
