"""Q1b: which day actually follows day d?  and are F13/F47 days copies of other farms' days?"""
import env, common
import pandas as pd, numpy as np
tX, ty, sX = common.load_raw()
X = pd.concat([tX, sX], ignore_index=True)
print('farms with out_temp:', X[X.out_temp.notna()].farm.value_counts().to_dict())
print('farms with act_heating:', X[X.act_heating.notna()].farm.value_counts().to_dict())
sd = {c: X[X.farm.isin(['F13','F47'])][c].std() for c in ['in_temp','in_hum','in_co2']}
for f in ['F13','F47']:
    d = X[X.farm==f]
    P = {c: d.pivot_table(index='day', columns='hour', values=c) for c in sd}
    days = P['in_temp'].index.values
    end = np.stack([P[c][23].values/sd[c] for c in sd],1)
    endslope = np.stack([(P[c][23]-P[c][22]).values/sd[c] for c in sd],1)
    start = np.stack([P[c][0].values/sd[c] for c in sd],1)
    # extrapolated end = end + slope
    pred = end + endslope*0.5
    D = np.sqrt(((pred[:,None,:]-start[None,:,:])**2).sum(-1))
    np.fill_diagonal(D, np.nan)
    offs=[]; ranks=[]
    for i,dd in enumerate(days):
        row = D[i]
        if np.all(np.isnan(row)): continue
        j = np.nanargmin(row); offs.append(days[j]-dd)
        if dd+1 in days:
            k = np.where(days==dd+1)[0][0]
            ranks.append((np.sum(row < row[k])))
    offs = pd.Series(offs)
    print(f, 'best successor offset distribution (top):', offs.value_counts().head(10).to_dict())
    print(f, '  |offset|<=3 frac', (offs.abs()<=3).mean().round(3), ' rank of true d+1 among', len(days)-1, 'median', np.median(ranks), 'frac rank0', np.mean(np.array(ranks)==0).round(3))
    # rank of d+2
    r2=[]
    for i,dd in enumerate(days):
        if dd+2 in days:
            k=np.where(days==dd+2)[0][0]; r2.append(np.sum(D[i]<D[i][k]))
    print(f, '  rank of d+2 median', np.median(r2), 'frac rank0', np.mean(np.array(r2)==0).round(3))
