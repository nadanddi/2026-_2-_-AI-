"""Standalone causal R3 EC recipe, fixed before fits; no PFN or SG2."""
import os,sys
sys.dont_write_bytecode=True
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'집/클로드/research'))
import env
import numpy as np,pandas as pd,lightgbm
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from season_transform_v1 import identify,vectors,mapping,W
RAW=['in_temp','in_hum','in_co2','act_vent','act_shade','act_thermal','act_heating','act_circfan','act_co2','act_fog']
ACTS=RAW[3:];DAY_BASE=RAW+['day','hr_sin','hr_cos','midnight']
DAY_FULL=DAY_BASE+[v+'_h0' for v in ACTS+RAW[:3]]+[n for v in ACTS for n in (v+'_tdm',v+'_tdz')]
BASE=[c for c in DAY_BASE if c!='day']+['season']
FULL=[c for c in DAY_FULL if c!='day']+['season']
OPS=['seal_run','vent_open_hours','first_open_hour','thermal_switches','shade_switches','since_curtain_change','heat_run','co2_hours','vent_max']
FULL_R3=FULL+OPS;BASE_R3=BASE+OPS
SEEDS=(7,101,2024);WEIGHTS={'et':.6,'lgb':.3,'mlp':.1}
def run_len(b):
    o=np.zeros(len(b));c=0
    for i,v in enumerate(b):c=c+1 if v else 0;o[i]=c
    return o
def ops_day(g):
    v=g.act_vent.fillna(0).to_numpy();th=g.act_thermal.fillna(0).to_numpy();sh=g.act_shade.fillna(0).to_numpy()
    he=g.act_heating.fillna(0).to_numpy();co=g.act_co2.fillna(0).to_numpy();h=g.hour.to_numpy();f=pd.DataFrame(index=g.index)
    f['seal_run']=run_len(v==0);f['vent_open_hours']=np.cumsum(v>0)
    f['first_open_hour']=np.where(np.cumsum(v>0)>0,np.minimum.accumulate(np.where(v>0,h,99)),24)
    ts=np.r_[0,np.abs(np.diff((th>0).astype(int)))];ss=np.r_[0,np.abs(np.diff((sh>0).astype(int)))]
    f['thermal_switches']=np.cumsum(ts);f['shade_switches']=np.cumsum(ss)
    last=-1;sc=[]
    for k in range(len(h)):
        if ts[k] or ss[k]:last=k
        sc.append(k-last if last>=0 else k+1)
    f['since_curtain_change']=sc;f['heat_run']=run_len(he>0);f['co2_hours']=np.cumsum(co>0);f['vent_max']=np.maximum.accumulate(v)
    return f
def features(raw):
    a=identify(raw).sort_values(['farm','day','hour']).reset_index(drop=True);assert a.row_id.is_unique
    assert ((a.hour>=0)&(a.hour<=23)).all()
    for key,g0 in a.groupby(['farm','day']):assert np.array_equal(g0.hour.to_numpy(),np.arange(g0.hour.max()+1)),key
    a['hr_sin']=np.sin(2*np.pi*a.hour/24);a['hr_cos']=np.cos(2*np.pi*a.hour/24);a['midnight']=a.hour.eq(0).astype(float)
    g=a.groupby(['farm','day'],sort=False);h0=a[a.hour.eq(0)].set_index(['farm','day']);key=pd.MultiIndex.from_arrays([a.farm,a.day])
    for v in ACTS+RAW[:3]:a[v+'_h0']=h0[v].reindex(key).values
    for v in ACTS:
        a[v+'_tdm']=g[v].transform(lambda s:s.expanding().mean())
        a[v+'_tdz']=g[v].transform(lambda s:s.eq(0).astype(float).where(s.notna()).expanding().mean())
    a=a.join(pd.concat([ops_day(gg) for _,gg in a.groupby(['farm','day'],sort=False)]))
    return a[['row_id','farm','hour']+DAY_FULL+OPS]
def shrink(p,frame):
    d=frame[['farm','day','hour']].reset_index(drop=True).copy();d['p']=p;d=d.sort_values(['farm','day','hour'])
    avg=d.groupby(['farm','day']).p.transform(lambda s:s.expanding().mean());o=np.empty(len(p));o[d.index.values]=(.5*d.p+.5*avg).values;return o
def season(t,q,raw_t):
    td=t[['farm','day']].drop_duplicates();qd=q[['farm','day']].drop_duplicates().reset_index(drop=True)
    ts,qs,notes=mapping(td,qd,vectors(raw_t));qm=dict(zip(qd.itertuples(index=False,name=None),qs))
    t=t.copy();q=q.copy();t['season']=[ts[(f,int(d))] for f,d in zip(t.farm,t.day)];q['season']=[qm[(f,int(d))] for f,d in zip(q.farm,q.day)]
    return t,q,notes
def feature_columns(kind):return FULL_R3 if kind=='et' else BASE_R3
def factory(kind,seed):
    if kind=='et':return make_pipeline(SimpleImputer(strategy='median'),ExtraTreesRegressor(n_estimators=600,max_features=1.,min_samples_leaf=1,n_jobs=2,random_state=seed))
    if kind=='lgb':return lightgbm.LGBMRegressor(n_estimators=800,learning_rate=.03,num_leaves=31,min_child_samples=40,subsample=.8,subsample_freq=1,colsample_bytree=.8,reg_lambda=1,deterministic=True,force_col_wise=True,n_jobs=2,verbose=-1,random_state=seed,objective='tweedie',tweedie_variance_power=1.5)
    if kind=='mlp':return make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),MLPRegressor(hidden_layer_sizes=(128,64),alpha=.01,learning_rate_init=.001,max_iter=800,early_stopping=True,n_iter_no_change=25,validation_fraction=.12,random_state=seed))
    raise ValueError(kind)
def frozen_season(frame,notes,training_days):
    f=frame.copy();vals=[]
    for farm,day in zip(f.farm,f.day):
        if day<179:
            days=sorted(int(d) for ff,d in training_days if ff==farm and d<179);vals.append(float(np.interp(day,days,days)))
        else:vals.append(float(np.interp(day,notes[farm]['anchor_day'],notes[farm]['anchor_season_pav'])))
    f['season']=vals;return f
