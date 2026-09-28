"""Q1h: groups of days sharing identical outside-weather vectors (same calendar date?)."""
import env, common, warnings
warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
tX, ty, sX = common.load_raw()
X = pd.concat([tX, sX], ignore_index=True)
X = X[X.farm.isin(['F13','F47','F32'])]
key = X.pivot_table(index=['farm','day'], columns='hour', values='out_temp').round(2)
key2 = X.pivot_table(index=['farm','day'], columns='hour', values='out_rad').round(1)
sig = pd.Series(['|'.join(map(str,a))+'#'+'|'.join(map(str,b)) for a,b in zip(key.values.tolist(), key2.values.tolist())], index=key.index)
g = sig.reset_index(); g.columns=['farm','day','sig']
g['gid'] = g.groupby('sig').ngroup()
sz = g.groupby('gid').size()
g['gsize']=g.gid.map(sz)
print('group size distribution (rows):', g.gsize.value_counts().sort_index().to_dict())
for f in ['F13','F47']:
    gg = g[g.farm==f]
    print(f, 'days in multi-groups', (gg.gsize>1).mean().round(3))
# show members of groups for F13
mem = g[g.gsize>1].groupby('gid').apply(lambda s: ','.join(f"{a}:{b}" for a,b in zip(s.farm,s.day)))
print(mem.head(60).to_string())
# day gap within same-farm groups
gaps=[]
for gid,s in g[g.gsize>1].groupby('gid'):
    for f in ['F13','F47']:
        dd=sorted(s[s.farm==f].day)
        gaps += list(np.diff(dd))
print('within-farm gap distribution', pd.Series(gaps).value_counts().head(10).to_dict())
g.to_csv('local/eda_time_9_groups.csv',index=False)
