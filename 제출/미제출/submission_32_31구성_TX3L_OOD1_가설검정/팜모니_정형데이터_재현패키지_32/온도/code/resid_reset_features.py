# -*- coding: utf-8 -*-
"""resid_reset 피처: 파일 쓰기, 정답 읽기, 모델 학습 없음.
사용: features = build_features(train_X_csv, test_X_csv)
입력은 CSV 경로 또는 pandas DataFrame. 출력은 row_id와 메타데이터, 89개 모델 열.
train_y_csv는 레이블 독립성 검사용 호환 인자이며 읽지 않는다.
"""
from pathlib import Path
import numpy as np
import pandas as pd
USABLE=['out_temp','out_hum','out_rad','out_wspd','in_temp','in_hum','in_co2','act_vent','act_shade','act_thermal','act_heating','act_circfan','act_co2','act_fog']
FILTERED=['in_temp','out_temp','out_rad','act_heating']
METADATA=['row_id','farm','day','hour','t']
# 1차와 열 순서까지 동일하게 유지.
FEATURE_COLUMNS=['day','hour','farm_id','sin','cos','second']
for _c in USABLE:
    FEATURE_COLUMNS.extend([_c,_c+'_h0',_c+'_mean',_c+'_std',_c+'_diff'])
    if _c in FILTERED:
        FEATURE_COLUMNS.extend([_c+'_reset'+str(h) for h in [1,3,8]])
FEATURE_COLUMNS.append('delta')
PHYSICS_COLUMNS=['farm_id','sin','cos','in_temp','out_temp','out_rad','act_heating']+[c+'_reset'+str(h) for c in USABLE if c in FILTERED for h in [1,3,8]]

def _read(value):
    frame=pd.read_csv(value) if isinstance(value,(str,Path)) else value.copy(deep=True)
    if not {'row_id',*USABLE}.issubset(frame.columns):
        raise ValueError('row_id와 14개 입력 센서 열이 필요합니다.')
    frame=frame[['row_id']+USABLE].copy()
    if not frame.row_id.str.match(r'^F\d{2}_\d{3}_\d{2}$').all():
        raise ValueError('row_id 형식 오류')
    frame['farm']=frame.row_id.str[:3]
    frame['day']=frame.row_id.str[4:7].astype(int)
    frame['hour']=frame.row_id.str[8:10].astype(int)
    if not frame.hour.between(0,23).all(): raise ValueError('시각 범위 오류')
    frame['t']=frame.day*24+frame.hour
    return frame

def build_features(train_X_csv,test_X_csv,train_y_csv=None):
    """원본 train_X/test_X에서 F13/F47 학습·평가 행의 인과적 피처를 반환한다.
    train_y_csv는 의도적으로 사용하지 않는다. 반환 순서는 온실·시각 순서다.
    제출 파이프라인에서는 row_id로 원래 평가 순서를 복원해야 한다.
    """
    a=pd.concat([_read(train_X_csv),_read(test_X_csv)],ignore_index=True)
    a=a[a.farm.isin(['F13','F47'])].sort_values(['farm','t']).reset_index(drop=True)
    if not a.row_id.is_unique: raise ValueError('train_X/test_X에 중복 row_id')
    result={c:a[c] for c in METADATA}
    result.update(farm_id=(a.farm=='F47').astype(float),sin=np.sin(a.hour*np.pi/12),cos=np.cos(a.hour*np.pi/12),second=(a.day>=179).astype(float))
    for c in USABLE:
        v=a[c]; group=v.groupby([a.farm,a.day])
        result[c]=v
        result[c+'_h0']=v.where(a.hour==0).groupby([a.farm,a.day]).ffill()
        result[c+'_mean']=group.transform(lambda x:x.expanding().mean())
        result[c+'_std']=group.transform(lambda x:x.expanding().std())
        result[c+'_diff']=group.diff()
        if c in FILTERED:
            for h in [1,3,8]: result[c+'_reset'+str(h)]=group.transform(lambda x,h=h:x.ewm(halflife=h).mean())
    result['delta']=a.in_temp-a.out_temp
    return pd.DataFrame(result)
