"""Q3: nearest labelled training days for each evaluation day (inside sensors + actuator signature)."""
import env, common, warnings
warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
tX, ty, sX = common.load_raw()
X = pd.concat([tX,sX]).merge(ty[['row_id','sub_temp','sub_ec']],on='row_id',how='left')
X['test']=X.row_id.isin(sX.row_id)
X=X[X.farm.isin(['F13','F47'])]
A=X.groupby(['farm','day']).agg(it=('in_temp','mean'),itmin=('in_temp','min'),itmax=('in_temp','max'),ih=('in_hum','mean'),
    fan=('act_circfan','mean'),shade=('act_shade','mean'),heat=('act_heating','mean'),aco2=('act_co2','mean'),test=('test','max'),lab=('sub_temp','count'))
tr=A[(~A.test)&(A.lab>0)]; te=A[A.test]
print('eval days in_temp mean quantiles', te.it.quantile([.1,.25,.5,.75,.9]).round(2).to_dict())
print('train label days in_temp mean quantiles', tr.it.quantile([.1,.25,.5,.75,.9]).round(2).to_dict())
f1=['it','itmin','itmax']; f2=f1+['fan','shade','heat','aco2']
sd=A[f2].std()
for feats,name in [(f1,'thermal only'),(f2,'thermal+actuator signature')]:
    rows=[]
    for k,r in te.iterrows():
        d=((((tr[feats]-r[feats].astype(float))/sd[feats])**2).sum(1)**0.5).astype(float)
        rows.append((k[0],k[1],r.it,d.nsmallest(5).mean(),(d<0.5).sum(), (tr.index[np.argmin(d.values)])))
    R=pd.DataFrame(rows,columns=['farm','day','it','d5','n05','nn'])
    R['cold']=R.it<11
    print(name); print(R.groupby('cold').agg(n=('day','size'),d5=('d5','mean'),n05=('n05','median'),n05min=('n05','min')).round(2))
    print(' nearest-day of each cold eval day:', R[R.cold][['farm','day','it','nn']].to_string(index=False))
# coverage: in_temp hourly <= x
for thr in [6,8,10]:
    print('hourly in_temp <',thr,': eval share %.3f  train-label share %.3f  train-label rows %d'%((X[X.test].in_temp<thr).mean(), (X[(~X.test)&X.sub_temp.notna()].in_temp<thr).mean(), (X[(~X.test)&X.sub_temp.notna()].in_temp<thr).sum()))
