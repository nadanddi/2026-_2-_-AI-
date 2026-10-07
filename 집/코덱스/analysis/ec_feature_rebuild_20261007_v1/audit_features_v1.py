"""Executable prefix, future, other-farm, order and missing-hour checks."""
from pathlib import Path
import hashlib
import json
import sys
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
sys.path.insert(0,str(ROOT/'집/클로드/research'))
import env
import numpy as np
import pandas as pd
from causal_features_v1 import RAW,annotate,build

X=pd.read_csv(Path(env.DATA)/'train_X.csv',usecols=['row_id']+RAW)
X=X[X.row_id.str.startswith(('F13_','F47_'))].copy()
Z=annotate(X)
F,metadata=build(X)
checks=[]
def compare(a,b,name):
    assert list(a.columns)==list(b.columns)
    assert a.index.equals(b.index)
    np.testing.assert_allclose(a.to_numpy(),b.to_numpy(),rtol=0,atol=0,equal_nan=True)
    checks.append({'check':name,'rows':len(a),'columns':len(a.columns),'status':'PASS'})

shuffled,_=build(X.sample(frac=1,random_state=20261007))
compare(F,shuffled,'row_order')
samples=Z[['farm','day']].drop_duplicates().groupby('farm').head(3)
for farm,day in samples.itertuples(index=False,name=None):
    G=Z[(Z.farm==farm)&(Z.day==day)]
    for hour in [0,1,2,3,6,12,23]:
        ids=G[G.hour<=hour].row_id
        prefix,_=build(G[G.hour<=hour])
        compare(F.loc[ids],prefix.loc[ids],f'prefix:{farm}:{day}:{hour}')
        poison=G.copy();poison.loc[poison.hour>hour,RAW]=99999
        altered,_=build(poison)
        compare(F.loc[ids],altered.loc[ids],f'future:{farm}:{day}:{hour}')
        one=altered.loc[[G.loc[G.hour==hour,'row_id'].iloc[0]]]
        compare(F.loc[one.index],one,f'single_query_with_prefix:{farm}:{day}:{hour}')
other=X.copy();other.loc[other.row_id.str.startswith('F47_'),RAW]=77777
OF,_=build(other)
ids=F.index[F.index.str.startswith('F13_')]
compare(F.loc[ids],OF.loc[ids],'other_farm')
G=Z[(Z.farm=='F13')&(Z.day==int(Z[Z.farm=='F13'].day.min()))]
gap,_=build(G[G.hour!=1]);row=G.loc[G.hour==2,'row_id'].iloc[0]
assert np.isnan(gap.loc[row,'in_temp__lag1'])
assert np.isnan(gap.loc[row,'in_temp__d2'])
checks.append({'check':'missing_hour_does_not_become_previous_row','status':'PASS'})
first=F.index[F.index.str.endswith('_00')]
assert F.loc[first,'in_temp__lag1'].isna().all()
checks.append({'check':'day_reset_no_cross_midnight','rows':len(first),'status':'PASS'})
assert not np.isinf(F.to_numpy()).any()
summary={'rows':len(F),'columns':len(F.columns),'checks':checks,
    'all_checks_pass':True,'label_inputs':False,'fitted_preprocessing':False,'test_inputs_read':False,
    'matrix_saved':False,'performance_tested':False,
    'limitations':['특징 생성기의 입력 인과성만 검사. 달력·이웃·모델·후처리 전체는 아직 미검사',
        '상호작용은 실측 함수율/광합성/관수 추정값이 아닌 검정할 대리 변수',
        '1시간 창은 현재값과 동일, 1차 rate는 차분과 동일. 중복 제거는 학습 내부에서만 적용',
        '짧은 prefix 부분 창의 관측 수를 별도 특징으로 남김'],
    'code_sha256':hashlib.sha256((HERE/'causal_features_v1.py').read_bytes()).hexdigest()}
for name,obj in [('feature_definitions_v1.json',metadata),('feature_audit_v1.json',summary)]:
    p=HERE/name;assert not p.exists();p.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:summary[k] for k in ['rows','columns','all_checks_pass','performance_tested']},ensure_ascii=False))
