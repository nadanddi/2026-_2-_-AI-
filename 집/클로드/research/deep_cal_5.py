# -*- coding: utf-8 -*-
"""Q1: per-date season descriptors + best predecessor/successor for every date (analysis)."""
import env  # noqa
import numpy as np, pandas as pd
import common
tX, ty, sX = common.load_raw()
k = pd.read_csv(env.LOCAL + "/deep_cal_2_days.csv")
a = pd.concat([tX, sX], ignore_index=True)
a = a[a.farm.isin(["F13", "F47"])]
rep = k.sort_values("day").groupby("date").first().reset_index()
C = np.load(env.LOCAL + "/deep_cal_3_cost.npy")
rows = []
for _, r in rep.iterrows():
    g = a[(a.farm == r.farm) & (a.day == r.day)].sort_values("hour")
    rad = g.out_rad.to_numpy(float)
    floor = np.nanmin(rad)
    on = np.where(rad > floor + 15)[0]
    # interpolate sunrise/sunset crossing at floor+15
    def cross(i0, i1):
        y0, y1 = rad[i0] - floor - 15, rad[i1] - floor - 15
        return i0 + (-y0) / (y1 - y0) if y1 != y0 else i0
    sr = cross(on[0] - 1, on[0]) if len(on) and on[0] > 0 else np.nan
    ss = cross(on[-1], on[-1] + 1) if len(on) and on[-1] < 23 else np.nan
    rows.append(dict(date=r.date, farm=r.farm, day=r.day, tmean=g.out_temp.mean(), tmin=g.out_temp.min(),
                     radsum=np.nansum(rad - floor), sunrise=sr, sunset=ss, daylen=ss - sr,
                     peakhr=np.average(np.arange(24), weights=np.clip(rad - floor, 0, None)) if np.nansum(rad - floor) > 0 else np.nan))
S = pd.DataFrame(rows)
mem = k.groupby("date").apply(lambda x: ",".join("%s%d" % (f[1:], d) for f, d in zip(x.farm, x.day)))
S["members"] = S.date.map(mem)
bp = np.argsort(C, axis=0)[:3].T  # best predecessors of each date
bs = np.argsort(C, axis=1)[:, :3]
S["pred3"] = [",".join(map(str, x)) for x in bp]
S["succ3"] = [",".join(map(str, x)) for x in bs]
S["c_pred"] = C.min(axis=0).round(2); S["c_succ"] = C.min(axis=1).round(2)
S.to_csv(env.LOCAL + "/deep_cal_5_dates.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
print(S[["date", "tmean", "tmin", "radsum", "sunrise", "sunset", "daylen", "peakhr", "pred3", "succ3", "c_pred", "c_succ", "members"]].round(2).to_string())
