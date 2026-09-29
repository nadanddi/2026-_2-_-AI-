"""Q1j: does a nearby predecessor day give 'normal' midnight continuity?"""
import env, common, warnings
warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
tX, ty, sX = common.load_raw()
X = pd.concat([tX, sX], ignore_index=True).merge(ty[['row_id','sub_temp','sub_ec']],on='row_id',how='left')
C=['in_temp','in_hum','in_co2','act_thermal','act_shade','act_circfan']
for f in ['F13','F47']:
    d=X[X.farm==f]
    P={c:d.pivot_table(index='day',columns='hour',values=c) for c in C+['sub_temp','sub_ec']}
    days=P['in_temp'].index
    P={c:v.reindex(days) for c,v in P.items()}
    sd={c:(P[c][23]-P[c][22]).abs().median()+1e-6 for c in C}
    # normal within-day distance h22->h23 (as score)
    def score(d1,h1,d2,h2):
        return np.nansum([abs(P[c].loc[d1,h1]-P[c].loc[d2,h2])/sd[c] for c in C[:3]])
    normal=[score(x,22,x,23) for x in days]
    rows=[]
    for x in days:
        cands={k:score(x-k,23,x,0) for k in range(1,9) if x-k in days}
        if not cands: continue
        k=min(cands,key=cands.get)
        rows.append((x,k,cands[k],cands.get(1,np.nan)))
    R=pd.DataFrame(rows,columns=['day','bestk','best','k1'])
    print(f,'normal h22->23 score median',np.median(normal).round(2),'p90',np.percentile(normal,90).round(2))
    print('  best-pred score median',R.best.median().round(2),' k=1 score median',R.k1.median().round(2))
    print('  bestk dist',R.bestk.value_counts().to_dict())
    # sub_temp & sub_ec jumps along best predecessor vs k=1
    for c in ['sub_temp','sub_ec']:
        jb=[abs(P[c].loc[x-k,23]-P[c].loc[x,0]) for x,k in zip(R.day,R.bestk)]
        j1=[abs(P[c].loc[x-1,23]-P[c].loc[x,0]) for x in R.day if x-1 in days]
        print(f'  {c} |jump| best-pred {np.nanmean(jb):.3f}  k=1 {np.nanmean(j1):.3f}  within-day h22->23 {np.nanmean((P[c][23]-P[c][22]).abs()):.3f}')
    R.to_csv(f'local/eda_time_11_{f}.csv',index=False)
