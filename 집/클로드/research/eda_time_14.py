"""Q2: season from day length (out_rad>5 hours), first/last sunlit hour, peak radiation, out_temp."""
import env, common, warnings
warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
tX, ty, sX = common.load_raw()
X = pd.concat([tX, sX], ignore_index=True)
X['test']=X.row_id.isin(sX.row_id)
for f in ['F13','F47']:
    d=X[X.farm==f]
    def agg(s):
        lit=s[s.out_rad>5.0]
        return pd.Series({'nlit':len(lit),'first':lit.hour.min() if len(lit) else np.nan,'last':lit.hour.max() if len(lit) else np.nan,
                          'rmax':s.out_rad.max(),'rsum':s.out_rad.sum(),'ot':s.out_temp.mean(),'test':s.test.max()})
    A=d.groupby('day').apply(agg)
    A['blk']=(A.index//15)*15
    print(f); print(A.groupby('blk').agg(nlit=('nlit','mean'),first=('first','mean'),last=('last','mean'),rmax=('rmax','median'),rmax90=('rmax',lambda s:s.quantile(.9)),ot=('ot','mean'),test=('test','mean')).round(2).to_string())
    A.to_csv(f'local/eda_time_14_{f}.csv')
