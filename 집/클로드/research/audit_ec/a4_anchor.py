# -*- coding: utf-8 -*-
"""EC audit 4: is 'previous same-dong date 23 h EC' an unbiased anchor for the next day's running mean (EA2 formula)?
Also: eval in_temp_h0 extrapolation and train-vs-eval input roughness (noise)."""
import sys, os
sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env  # noqa
PK = r"C:\Users\aozks\AppData\Local\Temp\claude\C--work-farmai\0038d811-b5c9-4a0d-913f-084fb31e14ab\scratchpad\build22\팜모니_정형데이터_재현패키지_22\EC"
import numpy as np, pandas as pd
R = pd.read_csv(os.path.join(PK, "st_dong_assign_v1.csv")).sort_values(["farm", "day"]).reset_index(drop=True)
R["date"] = R.groupby("farm").role.transform(lambda s: np.cumsum(s.values != "second") - 1)
y = pd.read_csv(os.path.join(PK, "data", "train_y.csv")); y = y[y.row_id.str[:3].isin(["F13", "F47"])]
y["farm"], y["day"], y["hour"] = y.row_id.str[:3], y.row_id.str[4:7].astype(int), y.row_id.str[8:10].astype(int)
Y = y.pivot_table(index=["farm", "day"], columns="hour", values="sub_ec")
key = R.set_index(["farm", "dong", "date"]).day
rows = []
for (f, d), v in Y.iterrows():
    r = R[(R.farm == f) & (R.day == d)]
    if not len(r): continue
    dong, date = r.dong.iloc[0], r.date.iloc[0]
    if (f, dong, date - 1) not in key.index: continue
    pd_ = key.loc[(f, dong, date - 1)]
    pd_ = pd_.iloc[0] if hasattr(pd_, "iloc") else pd_
    if (f, pd_) not in Y.index: continue
    A = Y.loc[(f, pd_), 23]; cm = v.expanding().mean().values
    rows.append(dict(farm=f, day=d, prev=pd_, p2=d >= 179, A=A, prevmean=Y.loc[(f, pd_)].mean(), mean=v.mean(), **{"rm%02d" % h: cm[h] - A for h in range(24)}, **{"e%02d" % h: v.values[h] - A for h in range(24)}))
T = pd.DataFrame(rows)
print("pairs", len(T), "pass-2 pairs", T.p2.sum())
for nm, g in [("all", T), ("pass2", T[T.p2])]:
    print("%s: day mean - prev23 : mean %+.4f median %+.4f sd %.4f | day mean - prev daymean: mean %+.4f sd %.4f | rel bias (mean/A-1) %+.3f" %
          (nm, (g["mean"] - g.A).mean(), (g["mean"] - g.A).median(), (g["mean"] - g.A).std(), (g["mean"] - g.prevmean).mean(), (g["mean"] - g.prevmean).std(), (g["mean"] / g.A - 1).median()))
    print("   running-mean minus A by hour (mean):", " ".join("%+.3f" % g["rm%02d" % h].mean() for h in range(0, 24, 3)))
    print("   hourly EC minus A by hour (mean):  ", " ".join("%+.3f" % g["e%02d" % h].mean() for h in range(0, 24, 3)))
    print("   |hour0 EC - A| median %.4f  (continuity at midnight)" % g.e00.abs().median())
# by level
T["lev"] = pd.qcut(T.A, 4)
print(T.groupby("lev", observed=True).apply(lambda g: pd.Series(dict(n=len(g), bias=(g["mean"] - g.A).mean(), rel=(g["mean"] / g.A - 1).mean()))).round(4).to_string())
# eval h0 temp extrapolation
x = pd.read_csv(os.path.join(PK, "data", "train_X.csv")); tx = pd.read_csv(os.path.join(PK, "data", "test_X.csv"))
for z in (x, tx): z["farm"], z["day"], z["hour"] = z.row_id.str[:3], z.row_id.str[4:7].astype(int), z.row_id.str[8:10].astype(int)
lo = x[x.hour == 0].in_temp.min()
e0 = tx[tx.hour == 0]
print("train min in_temp at h0 %.2f; eval records below it:" % lo, e0[e0.in_temp < lo][["row_id", "in_temp"]].values.tolist())
print("train in_temp overall min %.2f ; eval rows below train overall min: %d" % (x.in_temp.min(), int((tx.in_temp < x.in_temp.min()).sum())))
# roughness (noise): median |second difference| per record
def rough(z, c):
    z = z.sort_values(["farm", "day", "hour"]); s = z.groupby(["farm", "day"])[c].transform(lambda v: v.diff().diff().abs())
    return s
for c in ["in_temp", "in_hum", "in_co2"]:
    a = rough(x[x.day >= 179], c); b = rough(tx, c)
    print("%s |2nd diff| train p2 median %.3f p95 %.3f | eval median %.3f p95 %.3f" % (c, a.median(), a.quantile(.95), b.median(), b.quantile(.95)))
