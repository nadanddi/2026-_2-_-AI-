# -*- coding: utf-8 -*-
"""Inspector 4-3, step 2: EC physics and irrigation signatures (labels used
for DIAGNOSIS only).

  a. night (1-5 h) dlogEC ~ dT_sub + intercept: temperature effect vs drift
     (drying / redistribution).  Nights where the substrate warms.
  b. across days in the same source chain / same calendar date: does the day
     mean log EC track the day mean substrate temperature at ~2%/C?
  c. diurnal EC shape, morning step timing vs sunrise / accumulated radiation
     (irrigation start), per hour dEC profile.
  d. noise detector refinement: night-only CO2 diff ac1 (immune to dosing).
Run:  cd research && PYTHONPATH="" <python> -u audit4_3_02_ec_physics.py
"""
import env  # noqa: F401
import numpy as np
import pandas as pd

import common

pd.set_option("display.width", 200)


def ols(X, y):
    X = np.column_stack([np.ones(len(y))] + list(X))
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ b
    s2 = r @ r / (len(y) - X.shape[1])
    se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
    return b, se


def main():
    tX, ty, _ = common.load_raw()
    a = tX.merge(ty[["row_id", "sub_temp", "sub_ec"]], on="row_id")
    a = a[a.farm.isin(["F13", "F47"])].sort_values(["farm", "t"]).reset_index(drop=True)
    a["lec"] = np.log(a.sub_ec.clip(lower=0.03))
    g = a.groupby(["farm", "day"])
    a["dlec"] = g.lec.diff()
    a["dts"] = g.sub_temp.diff()
    a["dair"] = g.in_temp.diff()

    print("== a. night 1-5 h: dlogEC = b0 + b1*dTsub ==")
    n = a[a.hour.between(1, 5)].dropna(subset=["dlec", "dts"])
    for f, s in [("all", n)] + list(n.groupby("farm")):
        b, se = ols([s.dts.values], s.dlec.values)
        print("  %-4s n=%d  b0 %+.4f (se %.4f)/h  b1 %+.4f (se %.4f)/C" % (f, len(s), b[0], se[0], b[1], se[1]))
    for lo, hi in [(-9, -0.3), (-0.3, -0.1), (-0.1, 0.1), (0.1, 9)]:
        s = n[(n.dts > lo) & (n.dts <= hi)]
        print("  dTsub in (%+.1f,%+.1f]: n=%4d mean dlogEC %+.4f  mean dTsub %+.3f" % (lo, hi, len(s), s.dlec.mean(), s.dts.mean()))
    # evening 19-23 h too
    e = a[a.hour.between(19, 23)].dropna(subset=["dlec", "dts"])
    b, se = ols([e.dts.values], e.dlec.values)
    print("  evening 19-23 h: b0 %+.4f/h  b1 %+.4f/C" % (b[0], b[1]))

    print("\n== b. between days: day-mean log EC vs day-mean substrate T ==")
    dm = a.groupby(["farm", "day"]).agg(lec=("lec", "mean"), ts=("sub_temp", "mean"), ot=("out_temp", "mean"),
                                         rad=("out_rad", "sum"), ok=("out_temp", lambda s: tuple(np.round(s.values, 1)))).reset_index()
    # same calendar date pairs (identical outdoor vector) -> source differences at fixed date
    diffs = []
    for k, gg in dm.groupby("ok"):
        if len(gg) < 2:
            continue
        gg = gg.sort_values("day")
        for i in range(len(gg)):
            for j in range(i + 1, len(gg)):
                diffs.append((gg.ts.iloc[j] - gg.ts.iloc[i], gg.lec.iloc[j] - gg.lec.iloc[i]))
    dd = np.array(diffs)
    b, se = ols([dd[:, 0]], dd[:, 1])
    print("  same-date pairs n=%d: dlogEC = %+.3f + %+.4f*dTsub (se %.4f) corr %.3f"
          % (len(dd), b[0], b[1], se[1], np.corrcoef(dd[:, 0], dd[:, 1])[0, 1]))
    # d vs d-2 (usual same-source previous day)
    for lag in (1, 2):
        x = dm.set_index(["farm", "day"])
        rows = []
        for (f, d) in x.index:
            if (f, d - lag) in x.index:
                rows.append((x.loc[(f, d)].ts - x.loc[(f, d - lag)].ts, x.loc[(f, d)].lec - x.loc[(f, d - lag)].lec))
        r = np.array(rows)
        b, se = ols([r[:, 0]], r[:, 1])
        print("  d vs d-%d: n=%d slope %+.4f (se %.4f) corr %.3f" % (lag, len(r), b[1], se[1], np.corrcoef(r[:, 0], r[:, 1])[0, 1]))

    print("\n== c. diurnal shape ==")
    a["lec_dev"] = a.lec - g.lec.transform("mean")
    a["ts_dev"] = a.sub_temp - g.sub_temp.transform("mean")
    prof = a.groupby("hour").agg(lec_dev=("lec_dev", "mean"), ts_dev=("ts_dev", "mean"), dlec=("dlec", "mean"),
                                 dts=("dts", "mean"), rad=("out_rad", "mean"))
    prof["dlec_minus_T"] = prof.dlec - 0.02 * prof.dts
    print(prof.round(4).to_string())
    # morning: temperature-corrected EC step vs sunrise
    a["dlec_c"] = a.dlec - 0.02 * a.dts
    rows = []
    for (f, d), gg in a.groupby(["farm", "day"]):
        gg = gg.set_index("hour")
        if len(gg) < 24:
            continue
        sr = gg.index[gg.out_rad > 20]
        if not len(sr):
            continue
        sr = int(sr.min())
        m = gg.loc[sr:sr + 8, "dlec_c"]
        rows.append(dict(farm=f, day=d, sunrise=sr, h_min=int(m.idxmin()), v_min=m.min(), h_max=int(m.idxmax()), v_max=m.max(),
                         rad_to_min=gg.loc[sr:int(m.idxmin()), "out_rad"].sum()))
    M = pd.DataFrame(rows)
    M["lag_min"] = M.h_min - M.sunrise
    print("\n  biggest T-corrected EC drop after sunrise: lag (h) distribution", M.lag_min.value_counts().sort_index().to_dict())
    print("  median drop %.3f, median rise %.3f, lag_max dist %s" % (M.v_min.median(), M.v_max.median(),
          (M.h_max - M.sunrise).value_counts().sort_index().to_dict()))

    print("\n== d. night-only CO2 diff ac1 as noise detector ==")
    D = pd.read_csv(env.LOCAL + "/audit4_3_01_days.csv")
    b = pd.concat([tX, common.load_raw()[2]]).sort_values(["farm", "t"])
    b = b[b.farm.isin(["F13", "F47"])]
    rows = []
    for (f, d), gg in b.groupby(["farm", "day"]):
        nn = gg[(gg.hour <= 6) | (gg.hour >= 19)].sort_values("t")
        dc = nn.in_co2.diff()[nn.t.diff() == 1]
        rows.append(dict(farm=f, day=d, ac1_night=dc.autocorr(1) if dc.notna().sum() > 8 and dc.std() > 0 else np.nan))
    R = pd.DataFrame(rows)
    D = D.merge(R, on=["farm", "day"])
    t = D[D.farm.isin(["F13", "F47"])]
    print(t.groupby("grp").ac1_night.describe().round(2).to_string())
    tn = t[t.grp == "train_noisy"]
    print("  train_noisy days with night ac1 > -0.2 (night looks clean): %d / %d; among these CO2-dosing days (>2): %d"
          % ((tn.ac1_night > -0.2).sum(), len(tn), ((tn.ac1_night > -0.2) & (tn.act_co2 > 2)).sum()))
    tc = t[t.grp == "train_clean"]
    print("  train_clean days with night ac1 < -0.35 (night looks noisy): %d / %d; test: %d / %d"
          % ((tc.ac1_night < -0.35).sum(), len(tc), (t[t.grp == 'test'].ac1_night < -0.35).sum(), (t.grp == 'test').sum()))
    D[["farm", "day", "ac1_night"]].to_csv(env.LOCAL + "/audit4_3_02_night_ac1.csv", index=False)


if __name__ == "__main__":
    main()
