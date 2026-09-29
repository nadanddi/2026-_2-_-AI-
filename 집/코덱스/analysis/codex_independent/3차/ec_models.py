# -*- coding: utf-8 -*-
"""고정된 네 EC 후보. 학습 폴드의 당일 평균 정답만 훈련 목표로 사용."""
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import SplineTransformer,StandardScaler
from sklearn.linear_model import Ridge
from ec_features import FEATURE_COLUMNS,SPLINE_COLUMNS
NAMES=['daily_level_shape','log_daily_level_shape','spline_ridge','closed_expert']
PARAMS=dict(n_estimators=350,learning_rate=.035,num_leaves=9,min_child_samples=100,reg_lambda=10.,random_state=726,n_jobs=4,verbosity=-1,deterministic=True,force_col_wise=True)

def fit_predict_all(tr,va):
    out={}; y=tr.sub_ec.to_numpy(); key=[tr.farm,tr.day]
    # 두 단계는 훈련 자료만의 서로 다른 정답 성분에 적합한다.
    import pandas as pd
    level=pd.Series(y,index=tr.index).groupby(key).transform('mean').to_numpy()
    lev=LGBMRegressor(**PARAMS).fit(tr[FEATURE_COLUMNS],level)
    shape=LGBMRegressor(**PARAMS).fit(tr[FEATURE_COLUMNS],y-level)
    sh=shape.predict(va[FEATURE_COLUMNS]);lv=lev.predict(va[FEATURE_COLUMNS])
    out['daily_level_shape']=lv+sh
    ly=np.log(y); lm=pd.Series(ly,index=tr.index).groupby(key).transform('mean').to_numpy()
    ll=LGBMRegressor(**PARAMS).fit(tr[FEATURE_COLUMNS],lm)
    ls=LGBMRegressor(**PARAMS).fit(tr[FEATURE_COLUMNS],ly-lm)
    out['log_daily_level_shape']=np.exp(ll.predict(va[FEATURE_COLUMNS])+ls.predict(va[FEATURE_COLUMNS]))
    spline=make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),SplineTransformer(n_knots=6,degree=3,knots='quantile',include_bias=False,extrapolation='linear'),StandardScaler(),Ridge(alpha=30.))
    # 직접 EC를 예측하는 가법적 비선형 회귀: 목표 변환 후보와 구분.
    sc=SPLINE_COLUMNS+['farm_id','closed_prefix']
    spline.fit(tr[sc],y);out['spline_ridge']=spline.predict(va[sc])
    m=tr.closed_prefix.to_numpy()==1
    if m.sum()>=100:
        local=LGBMRegressor(**PARAMS).fit(tr.loc[m,FEATURE_COLUMNS],level[m])
        out['closed_expert']=np.where(va.closed_prefix.to_numpy()==1,local.predict(va[FEATURE_COLUMNS])+sh,lv+sh)
    else:out['closed_expert']=lv+sh
    return {k:np.clip(v,.062,3.46) for k,v in out.items()}
