# -*- coding: utf-8 -*-
"""EH1: do some high-EC days share a common input change while others show none?  (2026-10-09 집 클로드, user:
"고EC날들 중에 공통적인 데이터의 변화가 보이는 애들이 있고, 그런게 전혀 보이지 않은 애들이 있는지 확인해봐").
Exploratory, inputs only (train_X), labels only to define the day sets.
For each of the 26 high days (hx2_day_sets HIGH; EXPL = 9 well-predicted, UNEX2 = 17 unexplained) the day's input
summaries are compared with the SAME farm's NORMAL days (day mean EC < .8, not high) within +-10 record days:
robust z = (value - median of neighbours) / (1.4826 MAD of all normal days of that farm).
Summaries (24 h): means of in_temp, in_hum, in_co2, out_temp, out_rad (sum), act_vent, act_circfan, act_heating,
act_thermal, act_shade, act_co2, act_fog; closed hours (vent==0 & circfan==0); in-out temperature difference; VPD;
CO2 roughness (mean |hourly change|); night (0-5 h) in_temp; day-to-day change of in_temp mean vs d-1.
A day 'shows a change' if any |z| >= 2.5.  Days are clustered (Ward, standardized z) for a descriptive grouping.
Run:  cd 집/클로드/research && PYTHONPATH="" py -3.12 -u eh1_highec_input_signatures_v1.py
"""
import env  # noqa: F401
import os, json
import numpy as np, pandas as pd
import common
from scipy.cluster.hierarchy import linkage, fcluster


def day_summaries(x):
    x = x.sort_values(["farm", "day", "hour"]).copy()
    es = 0.6108 * np.exp(17.27 * x.in_temp / (x.in_temp + 237.3)); x["vpd"] = es * (1 - x.in_hum / 100)
    x["closed"] = ((x.act_vent.fillna(0) == 0) & (x.act_circfan.fillna(0) == 0)).astype(float)
    x["dT"] = x.in_temp - x.out_temp
    x["co2_absd"] = x.groupby(["farm", "day"]).in_co2.diff().abs()
    g = x.groupby(["farm", "day"])
    S = g[["in_temp", "in_hum", "in_co2", "out_temp", "act_vent", "act_circfan", "act_heating", "act_thermal",
           "act_shade", "act_co2", "act_fog", "dT", "vpd", "co2_absd"]].mean()
    S["out_rad_sum"] = g.out_rad.sum(); S["closed_h"] = g.closed.sum()
    S["night_temp"] = x[x.hour <= 5].groupby(["farm", "day"]).in_temp.mean()
    prev = S.in_temp.copy(); prev.index = pd.MultiIndex.from_arrays([prev.index.get_level_values(0), prev.index.get_level_values(1) + 1])
    S["d_temp_vs_prev"] = S.in_temp - prev.reindex(S.index)
    return S


def main():
    tX, ty, sX = common.load_raw()
    x = tX[tX.row_id.str[:3].isin(["F13", "F47"])].copy()
    x[["farm", "day", "hour"]] = x.row_id.str.split("_", expand=True); x.day = x.day.astype(int); x.hour = x.hour.astype(int)
    y = ty[ty.row_id.str[:3].isin(["F13", "F47"])].copy()
    y[["farm", "day", "hour"]] = y.row_id.str.split("_", expand=True); y.day = y.day.astype(int)
    dm = y.groupby(["farm", "day"]).sub_ec.mean()
    sets = json.load(open(os.path.join(env.LOCAL, "hx2_day_sets.json"), encoding="utf-8"))
    high = {(f, int(d)) for f, d in sets["HIGH"]}; unex = {(f, int(d)) for f, d in sets["UNEX2"]}
    S = day_summaries(x); feats = list(S.columns)
    normal = [k for k, v in dm.items() if v < .8 and k not in high and k in S.index]
    Z = {}
    for (f, d) in sorted(high):
        nb = [k for k in normal if k[0] == f and abs(k[1] - d) <= 10]
        allnorm = S.loc[[k for k in normal if k[0] == f]]
        mad = 1.4826 * (allnorm - allnorm.median()).abs().median()
        z = (S.loc[(f, d)] - S.loc[nb].median()) / mad.replace(0, np.nan)
        Z[(f, d)] = z
    Z = pd.DataFrame(Z).T
    Z.index = pd.MultiIndex.from_tuples(Z.index, names=["farm", "day"])
    grp = ["UNEX2" if k in unex else "EXPL" for k in Z.index]
    print("neighbour normal days per high day: min %d" % min(len([k for k in normal if k[0] == f and abs(k[1] - d) <= 10]) for f, d in Z.index))
    print("\n== per high day: EC day mean, group, number of |z|>=2.5, strongest deviations")
    rows = []
    for (k, z), g in zip(Z.iterrows(), grp):
        big = z[z.abs() >= 2.5].sort_values(key=np.abs, ascending=False)
        rows.append((k, g, len(big)))
        print("%s %3d %-5s EC %.2f  n|z|>=2.5: %2d  %s" % (k[0], k[1], g, dm[k], len(big),
              ", ".join("%s %+.1f" % (c, v) for c, v in big.head(5).items())))
    R = pd.DataFrame(rows, columns=["k", "grp", "n"])
    print("\n== days showing a change (any |z|>=2.5): EXPL %d/9, UNEX2 %d/17" % (int((R[R.grp == "EXPL"].n > 0).sum()), int((R[R.grp == "UNEX2"].n > 0).sum())))
    print("\n== features: share of high days with z >= +2 / <= -2 (EXPL | UNEX2)")
    for c in feats:
        e, u = Z[c][[g == "EXPL" for g in grp]], Z[c][[g == "UNEX2" for g in grp]]
        print("  %-15s up %d/9 down %d/9 | up %2d/17 down %2d/17 | median z %+.1f | %+.1f" % (
            c, (e >= 2).sum(), (e <= -2).sum(), (u >= 2).sum(), (u <= -2).sum(), e.median(), u.median()))
    Zc = Z.fillna(0).clip(-6, 6)
    cl = fcluster(linkage(Zc.values, "ward"), 3, "maxclust")
    print("\n== Ward clusters (3)")
    for c in sorted(set(cl)):
        m = cl == c
        mem = [("%s%d%s" % (k[0][1:], k[1], "*" if g == "UNEX2" else "")) for k, g, mm in zip(Z.index, grp, m) if mm]
        top = Zc[m].mean().sort_values(key=np.abs, ascending=False).head(5)
        print("  cluster %d (%d days; * = unexplained): %s\n     mean z: %s" % (c, m.sum(), " ".join(mem),
              ", ".join("%s %+.1f" % (k, v) for k, v in top.items())))
    Z.assign(grp=grp, ec=[dm[k] for k in Z.index], cluster=cl).to_csv(os.path.join(env.LOCAL, "eh1_highec_z.csv"))


if __name__ == "__main__":
    main()
