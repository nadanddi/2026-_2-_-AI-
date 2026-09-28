import env, common
import numpy as np, pandas as pd
tX, ty, sX = common.load_raw()
tX['is_test']=False; sX['is_test']=True
A=pd.concat([tX,sX]).sort_values(['farm','t'])
A=A.merge(ty[['row_id','sub_temp','sub_ec']],on='row_id',how='left')
# 3h smoothing: causal EWM halflife 3 within contiguous farm record
A['sm3']=A.groupby('farm').in_temp.transform(lambda s:s.ewm(halflife=3,ignore_na=True).mean())
fl=pd.read_csv('local/eda_forensic_11_flags.csv').set_index('row_id').Vany
A['flag']=A.row_id.map(fl).fillna(False).astype(bool)
T=A[A.farm.isin(['F13','F47'])]
L=T[~T.is_test & T.sub_temp.notna()]; E=T[T.is_test]
print('in_temp mean train-label %.2f test %.2f; <8: %.3f %.3f; <6: %.4f(%d) %.4f'%(L.in_temp.mean(),E.in_temp.mean(),(L.in_temp<8).mean(),(E.in_temp<8).mean(),(L.in_temp<6).mean(),(L.in_temp<6).sum(),(E.in_temp<6).mean()))
q=L.in_temp.quantile(.01); print('below train 1%% (%.1f): test %.3f'%(q,(E.in_temp<q).mean()))
L=L.assign(gap=L.sub_temp-L.sm3)
b=pd.cut(L.sm3,[-9,6,8,10,12,15,99])
print(L.groupby(b,observed=True).agg(n=('gap','size'),gap=('gap','mean'),flagfrac=('flag','mean'),heat=('act_heating','mean')).round(3))
print('clean only'); Lc=L[~L.flag]; print(Lc.groupby(pd.cut(Lc.sm3,[-9,6,8,10,12,15,99]),observed=True).gap.agg(['size','mean']).round(3))
print('rows sm3<6:'); print(L[L.sm3<6][['row_id','in_temp','sm3','sub_temp','act_heating','out_temp','flag']])
print('days contributing sm3<8:', L[L.sm3<8].groupby(['farm','day']).size().to_dict())
# per-day mean gap on cold days: is gap a day offset?
# Other farms: sub - sm3 by bin (sub_temp integer-rounded)
O=A[~A.farm.isin(['F13','F47']) & A.sub_temp.notna()].copy(); O['gap']=O.sub_temp-O.sm3
tab=O.groupby(['farm',pd.cut(O.sm3,[-9,6,8,10,15,99])],observed=True).gap.agg(['size','mean']).unstack()
print(tab['mean'].loc[['F50','F38','F30','F33','F42']].round(2)); print(tab['size'].loc[['F50','F38','F30','F33','F42']])
