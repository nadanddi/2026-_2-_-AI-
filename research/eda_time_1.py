"""Q1: are consecutive days really consecutive?  midnight continuity of inside sensors."""
import env, common
import pandas as pd, numpy as np
tX, ty, sX = common.load_raw()
X = pd.concat([tX, sX], ignore_index=True).sort_values(['farm','t'])
X = X.merge(ty[['row_id','sub_temp','sub_ec']], on='row_id', how='left')
cols = ['in_temp','in_hum','in_co2','out_temp','out_hum','out_rad','sub_temp','sub_ec','act_heating','act_thermal']
rng = np.random.default_rng(0)
def jumps(df, c):
    s = df.set_index('t')[c]
    d = s.reindex(s.index + 1).values - s.values   # value at t+1 minus t
    h = (df.hour.values)
    return pd.Series(np.abs(d), index=h)
for farms, name in [(['F13','F47'],'target'), (sorted(set(X.farm)-{'F13','F47'}),'others')]:
    print('=====', name)
    for c in cols:
        rows=[]
        for f in farms:
            d = X[X.farm==f]
            if d[c].notna().sum()<100: continue
            rows.append(jumps(d,c))
        if not rows: continue
        j = pd.concat(rows)
        by = j.groupby(level=0).mean()
        # jump from hour h to h+1; h=23 is midnight
        other = by.drop(23)
        print(f"{c:12s} 23->0 {by[23]:.3f}  others mean {other.mean():.3f} max {other.max():.3f}(h={other.idxmax()}) ratio {by[23]/other.mean():.2f}  22->23 {by[22]:.3f} 0->1 {by[0]:.3f}")

# adjacent-day similarity: distance between hour-23 of day d and hour-0 of day d+k
print('===== successor test: |x(d,23) - x(d+k,0)| and daily-profile distance')
for f in ['F13','F47']:
    d = X[X.farm==f]
    for c in ['in_temp','in_hum','in_co2']:
        P = d.pivot_table(index='day', columns='hour', values=c)
        days = P.index.values
        res = {}
        for k in [1,2,3,5,10,30]:
            a = []
            for dd in days:
                if dd+k in P.index:
                    a.append(abs(P.loc[dd,23]-P.loc[dd+k,0]))
            res[k] = np.nanmean(a)
        # within-day h23->h22 comparison
        w = np.nanmean(np.abs(P[23]-P[22]))
        # daily mean corr at lags
        m = P.mean(axis=1)
        m = m.reindex(range(days.min(), days.max()+1))
        dm = m.diff()
        ac = {k: m.autocorr(k) for k in [1,2,3,7]}
        acd = {k: dm.autocorr(k) for k in [1,2]}
        print(f, c, 'h23->h0 by lag', {k: round(v,2) for k,v in res.items()}, 'h22->23', round(w,2), '| daily-mean AC', {k: round(v,3) for k,v in ac.items()}, 'diff AC', {k: round(v,3) for k,v in acd.items()})
