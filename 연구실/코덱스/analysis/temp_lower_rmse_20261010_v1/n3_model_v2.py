"""Train-only midnight level and same-day relative shape, fixed two-head prototype."""
from pathlib import Path
import sys
H=Path(__file__).resolve().parent;R=H.parents[3];sys.path.insert(0,str(H.parent/'temp_repair_20261009_v1'))
from runtime_v1 import np,pd,FEATURE_COLUMNS,PHYSICS_COLUMNS
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor

def keys(a):return a.farm.astype(str)+'_'+(a.t//24).astype(int).astype(str)
def at0(a):
    out=a.loc[a.hour==0].copy();out.index=keys(out);assert out.index.is_unique
    assert set(out.index)==set(keys(a));return out

def delta(a):
    anchor=at0(a)[PHYSICS_COLUMNS].loc[keys(a)].to_numpy(float)
    d=a[PHYSICS_COLUMNS].to_numpy(float)-anchor
    d[a.hour.to_numpy()==0]=0.0
    return pd.DataFrame(d,columns=PHYSICS_COLUMNS,index=a.index)

def predict(tr,va,w):
    assert 'sub_temp' not in va.columns
    for frame in [tr,va]:assert (frame.groupby(keys(frame)).size()==24).all()
    tr0=at0(tr);va0=at0(va);wm=tr.hour.to_numpy()==0;w0=w[wm]
    lc=PHYSICS_COLUMNS+['day'];assert len(lc)==20
    level=make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler(),Ridge(alpha=100.))
    level.fit(tr0[lc],tr0.sub_temp,ridge__sample_weight=w0)
    lm=LGBMRegressor(n_estimators=120,learning_rate=.03,num_leaves=8,min_child_samples=12,reg_lambda=15,n_jobs=4,random_state=9109,verbosity=-1)
    lm.fit(tr0[FEATURE_COLUMNS],tr0.sub_temp.to_numpy()-level.predict(tr0[lc]),sample_weight=w0)
    lv=level.predict(va0[lc])+lm.predict(va0[FEATURE_COLUMNS]);lmapping=pd.Series(lv,index=va0.index)
    yshape=tr.sub_temp.to_numpy()-tr0.sub_temp.loc[keys(tr)].to_numpy()
    shp=make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler(with_mean=False),Ridge(alpha=100.,fit_intercept=False))
    dt=delta(tr);dv=delta(va);shp.fit(dt,yshape,ridge__sample_weight=w)
    bp=shp.predict(dv);assert np.array_equal(bp[va.hour.to_numpy()==0],np.zeros(len(va0)))
    sm=LGBMRegressor(n_estimators=220,learning_rate=.035,num_leaves=12,min_child_samples=100,reg_lambda=15,n_jobs=4,random_state=9109,verbosity=-1)
    sm.fit(tr[FEATURE_COLUMNS],yshape-shp.predict(dt),sample_weight=w)
    raw=bp+sm.predict(va[FEATURE_COLUMNS]);zero=pd.Series(raw[va.hour.to_numpy()==0],index=va0.index)
    shape=raw-zero.loc[keys(va)].to_numpy();lev=lmapping.loc[keys(va)].to_numpy();out=lev+shape
    assert np.array_equal(out[va.hour.to_numpy()==0],lv)
    # Validation-label columns are rejected before fitting; actual negative call is recorded by the runner.
    return out,{'level':lev,'shape_raw':raw,'shape':shape,'level_rows':len(tr0),'shape_rows':len(tr),'query_label_access':False}
