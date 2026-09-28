# -*- coding: utf-8 -*-
"""Q4 (analysis): what explains the daily EC level -- day index, calendar, pass, source chain?"""
import env  # noqa
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold

S = pd.read_csv(env.LOCAL + "/deep_cal_11_days.csv")      # has cal, chain, ec_m
S["second"] = (S.day > 178).astype(int)
E = S.dropna(subset=["ec_m"]).copy()


def spline(x, knots):
    x = np.asarray(x, float)
    return np.column_stack([x] + [np.clip(x - k, 0, None) for k in knots])


def cv_r2(df, make):
    # 10 blocked folds on calendar to avoid neighbour leakage
    g = (df.cal // 13).values
    pred = np.zeros(len(df))
    for tr, te in GroupKFold(10).split(df, groups=g):
        Xtr, Xte = make(df.iloc[tr]), make(df.iloc[te])
        m = LinearRegression().fit(Xtr, df.ec_m.values[tr])
        pred[te] = m.predict(Xte)
    return 1 - np.mean((df.ec_m - pred) ** 2) / df.ec_m.var()


for f in ["F13", "F47"]:
    e = E[E.farm == f].reset_index(drop=True)
    kd = np.percentile(e.day, [20, 40, 60, 80]); kc = np.percentile(e.cal, [20, 40, 60, 80])
    res = {
        "day": lambda d: spline(d.day, kd),
        "cal": lambda d: spline(d.cal, kc),
        "cal+second": lambda d: np.column_stack([spline(d.cal, kc), d.second]),
        "day+cal": lambda d: np.column_stack([spline(d.day, kd), spline(d.cal, kc)]),
    }
    print(f, " ".join("%s R2cv=%.3f" % (k, cv_r2(e, fn)) for k, fn in res.items()))
    # pairs of copies of the same date inside the record
    g = e.groupby("cal")
    diffs, fs = [], []
    for c, x in g:
        if len(x) == 2:
            diffs.append(abs(x.ec_m.iloc[0] - x.ec_m.iloc[1]))
            if x.second.nunique() == 2:
                fs.append(x[x.second == 1].ec_m.iloc[0] - x[x.second == 0].ec_m.iloc[0])
    print("   same-date two copies |dEC| median %.3f (n=%d); second-pass copy minus first-pass copy: mean %+.3f median %+.3f (n=%d)"
          % (np.median(diffs), len(diffs), np.mean(fs), np.median(fs), len(fs)))
    # neighbour-in-calendar vs neighbour-in-day similarity for second-pass days
    sp = e[e.second == 1]
    fp_ = e[e.second == 0]
    near_cal = [fp_.iloc[(fp_.cal - c).abs().argsort()[:4]].ec_m.mean() for c in sp.cal]
    near_day = [e[(e.day != d)].iloc[(e[(e.day != d)].day - d).abs().argsort()[:4]].ec_m.mean() for d in sp.day]
    print("   second-pass labelled days: RMSE(EC, mean of 4 nearest first-pass days by CAL) %.3f ; by DAY index %.3f ; overall sd %.3f"
          % (np.sqrt(np.mean((sp.ec_m.values - near_cal) ** 2)), np.sqrt(np.mean((sp.ec_m.values - near_day) ** 2)), sp.ec_m.std()))
    # chain-level EC offsets: within same cal, do chains differ systematically?
    e["cal_mean"] = e.groupby("cal").ec_m.transform("mean")
    e["dev"] = e.ec_m - e.cal_mean
    ch = e[e.groupby("cal").ec_m.transform("size") == 2].groupby("chain").dev.agg(["mean", "size"])
    ch = ch[ch["size"] >= 8]
    print("   chain offsets vs same-date mean (chains with >=8 paired days):", [(int(i), round(m, 3), int(n)) for i, (m, n) in ch.iterrows()])
    # EC along chain position
    e["pos"] = e.groupby("chain").cal.rank()
    print("   corr(EC, chain position) %.2f ; corr(EC, cal) %.2f ; corr(EC, day) %.2f" % (e.ec_m.corr(e.pos), e.ec_m.corr(e.cal), e.ec_m.corr(e.day)))
