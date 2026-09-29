import env, common
import numpy as np, pandas as pd
tX, ty, sX = common.load_raw()
tX['is_test']=False; sX['is_test']=True
A=pd.concat([tX,sX]).sort_values(['farm','t'])
A=A[A.farm.isin(['F13','F47'])].merge(ty[['row_id','sub_temp']],on='row_id',how='left')
F=pd.read_csv('local/eda_forensic_11_flags.csv').set_index('row_id')
A=A.join(F.drop(columns='set'),on='row_id')
V=['V1','V3','V4','V5','V6','V7']
for s in [False,True]:
    X=A[A.is_test==s]
    lab = X if s else X[X.sub_temp.notna()]
    print('TEST' if s else 'TRAIN-LABEL', len(lab),'Vany n',int(lab.Vany.sum()),'rate %.4f'%lab.Vany.mean(), {v:int(lab[v].sum()) for v in V})
    print('  flagged rows by hour==0:',int(lab[lab.Vany & (lab.hour==0)].shape[0]), ' hour dist', lab[lab.Vany].hour.value_counts().sort_index().to_dict())
    print('  days', lab[lab.Vany].groupby(['farm','day']).size().to_dict())
X=A[A.is_test & A.Vany]; print(X[['row_id','in_temp','out_temp','in_co2','act_co2','in_hum']+V].to_string())
A['sm3']=A.groupby('farm').in_temp.transform(lambda s:s.ewm(halflife=3).mean())
L=A[~A.is_test & A.sub_temp.notna()].copy(); L['gap']=L.sub_temp-L.sm3; L['graw']=L.sub_temp-L.in_temp
for name,m in [('flag h0',L.Vany&(L.hour==0)),('flag not h0',L.Vany&(L.hour!=0)),('clean',~L.Vany)]:
    print(name,m.sum(),'gap_sm3 %.2f  gap_raw %.2f in_temp %.2f'%(L.gap[m].mean(),L.graw[m].mean(),L.in_temp[m].mean()))
# midnight-rule false positive: of all hour-0 rows, how many have big jumps, train vs test
for s in [False,True]:
    X=A[(A.is_test==s)&(A.hour==0)]
    print('set test' if s else 'train', 'hour0 rows',len(X),'flag rate at h0 %.3f'%X.Vany.mean())
# rule per type, excluding hour 0
for v in V:
    tr=L[L[v]]; print(v, len(tr),'h0',int((tr.hour==0).sum()),'gap_raw %.2f'%tr.graw.mean())
