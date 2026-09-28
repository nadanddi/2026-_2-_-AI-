"""Q5: is sub_temp an hourly mean or a point value?  lag structure of sub_temp vs in_temp, within day."""
import env, common, warnings
warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
tX, ty, sX = common.load_raw()
X = tX.merge(ty[['row_id','sub_temp','sub_ec']],on='row_id',how='left').sort_values(['farm','t'])
def run(farms,name):
    D=X[X.farm.isin(farms)].copy()
    g=D.groupby(['farm','day'])
    for k in range(-3,4):
        D[f'it{k:+d}']=g.in_temp.shift(-k)   # in_temp at t+k, same day only
    D['st_dm']=D.sub_temp-g.sub_temp.transform('mean')
    cols=[f'it{k:+d}' for k in range(-3,4)]
    for c in cols: D[c+'dm']=D[c]-g[c].transform('mean')
    E=D.dropna(subset=['st_dm']+[c+'dm' for c in cols])
    corr={c:round(np.corrcoef(E.st_dm,E[c+'dm'])[0,1],3) for c in cols}
    Xm=E[[c+'dm' for c in cols]].values; b=np.linalg.lstsq(Xm,E.st_dm.values,rcond=None)[0]
    # compare point vs 2-point averages
    for lab,v in [('t',E['it+0dm']),('mean(t-1,t)',(E['it-1dm']+E['it+0dm'])/2),('mean(t,t+1)',(E['it+0dm']+E['it+1dm'])/2),('mean(t-2..t)',(E['it-2dm']+E['it-1dm']+E['it+0dm'])/3)]:
        corr[lab]=round(np.corrcoef(E.st_dm,v)[0,1],4)
    print(name,'n',len(E)); print('  corr', corr); print('  joint coef', dict(zip(cols,b.round(3))))
run(['F13','F47'],'target')
run(['F13'],'F13'); run(['F47'],'F47')
others=sorted(set(X.farm)-{'F13','F47'})
run(others,'others')
