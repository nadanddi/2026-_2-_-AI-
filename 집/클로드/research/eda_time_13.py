"""Q1l: value of source id for daily EC/temp level (block holdout like test) + causal identifiability."""
import env, common, warnings
warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, GroupKFold
tX, ty, sX = common.load_raw()
X = pd.concat([tX, sX], ignore_index=True).merge(ty[['row_id','sub_temp','sub_ec']],on='row_id',how='left')
acts=['act_vent','act_shade','act_thermal','act_heating','act_circfan','act_co2','act_fog','in_co2','in_hum','in_temp']
for f in ['F13','F47']:
    L=pd.read_csv(f'local/eda_time_10_{f}.csv').set_index('day')
    lab=L.dropna(subset=['ec'])
    days=lab.index.values
    out={'ec':{'nocl':[], 'cl':[]}, 'st':{'nocl':[], 'cl':[]}}
    for s in range(days.min(), days.max()-10, 3):
        hold=[x for x in days if s<=x<s+10]
        train=[x for x in days if x<s-1 or x>=s+11]
        if len(hold)<5: continue
        for x in hold:
            tr=np.array(train); dist=np.abs(tr-x)
            for tgt in ['ec','st']:
                nn=tr[np.argsort(dist)[:4]]
                out[tgt]['nocl'].append(lab.loc[nn,tgt].mean()-lab.loc[x,tgt])
                same=tr[lab.loc[tr,'cl'].values==lab.loc[x,'cl']]
                if len(same)==0: same=tr
                nn=same[np.argsort(np.abs(same-x))[:4]]
                out[tgt]['cl'].append(lab.loc[nn,tgt].mean()-lab.loc[x,tgt])
    for tgt in out:
        print(f,tgt,'daily-level RMSE: nearest-4-days', round(np.sqrt(np.mean(np.square(out[tgt]['nocl']))),3), ' nearest-4-same-source', round(np.sqrt(np.mean(np.square(out[tgt]['cl']))),3))
    # causal identifiability: predict day cluster from inputs at hour h only (and cumulative 0..h)
    d=X[X.farm==f].copy(); d['cl']=d.day.map(L.cl)
    for h in [0,3,6,12]:
        sub=d[d.hour<=h].groupby('day')[acts].mean()
        y=L.cl.reindex(sub.index)
        rf=RandomForestClassifier(200,n_jobs=2,random_state=0,min_samples_leaf=2)
        acc=cross_val_score(rf,sub.fillna(-1),y,cv=GroupKFold(5),groups=sub.index//20).mean()
        print(f,' cluster identifiable from hours 0..%d mean inputs: acc %.3f'%(h,acc))
