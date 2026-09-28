"""Q3b: cold-range coverage in other farms and sub_temp - in_temp behaviour by in_temp bin."""
import env, common, warnings
warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
tX, ty, sX = common.load_raw()
X = tX.merge(ty[['row_id','sub_temp']],on='row_id').dropna(subset=['sub_temp','in_temp'])
X['tgt']=X.farm.isin(['F13','F47'])
X['bin']=pd.cut(X.in_temp,[-10,4,6,8,10,12,15,20,50])
print(X.groupby(['bin','tgt']).agg(n=('sub_temp','size'),gap=('sub_temp',lambda s: (s-X.loc[s.index,'in_temp']).mean()),st=('sub_temp','mean')).round(2).unstack())
# per-farm cold rows count
c=X[X.in_temp<6].groupby('farm').size().sort_values(ascending=False)
print('other farms with most in_temp<6 labelled rows:', c.head(10).to_dict(), 'total', c.sum())
