# -*- coding: utf-8 -*-
"""RD5: are F13/F47 sub_ec LABELS partly filled-in straight lines?
rd4 found 3,237 of 9,600 sub_ec rows inside >= 6-h stretches whose 2nd
difference is within 0.001 (sub_temp: 71 rows).  2026-10-02 집 클로드.

Checks
  1. stretch anatomy: constant vs sloped, length distribution, hours, farms,
     early/late segment, per-day coverage (whole days straight?)
  2. physics test: a real raw-EC sensor tracks substrate temperature (~2 %/C,
     6.27).  Inside straight stretches the EC deviation from its line cannot
     follow sub_temp; compare the within-day corr(log EC, sub_temp) on
     straight-stretch rows vs other rows, and the share of days with a
     positive slope.
  3. random-walk control: how often would a real series of this volatility
     produce >= 6-h straight stretches at 0.001 resolution?  Estimated from
     the test of the same rule on sub_ec rows outside the flagged days
     (hour-to-hour |diff| distribution) - simulation of the run statistic.
  4. EC v2 DIAG10 error on straight rows vs others; level vs shape part.
  5. examples.

Run:  cd 집/클로드/research && PYTHONPATH="" py -3.12 -u rd5_ec_label_fill_v1.py
"""
import env  # noqa: F401
import os

import numpy as np
import pandas as pd

D = env.DATA
RNG = np.random.default_rng(5)

y = pd.read_csv(os.path.join(D, "train_y.csv"))
p = y.row_id.str.split("_", expand=True)
y["farm"], y["day"], y["hour"] = p[0], p[1].astype(int), p[2].astype(int)
y = y[y.farm.isin(["F13", "F47"])].sort_values(["farm", "day", "hour"]).reset_index(drop=True)
y["t"] = y.day * 24 + y.hour


def runs(v, t, tol):
    n = len(v)
    ok = np.zeros(n, bool)
    for i in range(1, n - 1):
        if t[i + 1] - t[i] == 1 and t[i] - t[i - 1] == 1 and not np.isnan(v[i - 1:i + 2]).any():
            ok[i] = abs(v[i + 1] - 2 * v[i] + v[i - 1]) <= tol + 1e-9
    out = []
    i = 0
    while i < n:
        if not ok[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and ok[j + 1]:
            j += 1
        out.append((i - 1, j + 1))
        i = j + 1
    return out


rows = []
y["L"] = 0
y["const"] = False
y["slope"] = np.nan
for f, g in y.groupby("farm"):
    v, t, ix = g.sub_ec.values, g.t.values, g.index.values
    for a, b in runs(v, t, 0.001):
        L = b - a + 1
        sl = (v[b] - v[a]) / (L - 1)
        const = bool(np.all(np.diff(v[a:b + 1]) == 0))
        rows.append(dict(farm=f, t0=int(t[a]), L=L, const=const, slope=sl,
                         cross_mid=(t[a] // 24) != (t[b] // 24), lvl=float(v[a:b + 1].mean())))
        y.loc[ix[a:b + 1], "L"] = np.maximum(y.loc[ix[a:b + 1], "L"], L)
        y.loc[ix[a:b + 1], "const"] = const
        y.loc[ix[a:b + 1], "slope"] = sl
R = pd.DataFrame(rows)
S = y.L >= 6
print("1. stretches >= 6 h: %d (rows %d = %.1f%%)" % ((R.L >= 6).sum(), S.sum(), 100 * S.mean()))
R6 = R[R.L >= 6]
print("   constant %d, sloped %d; length quantiles %s; crossing midnight %d"
      % (R6.const.sum(), (~R6.const).sum(), np.quantile(R6.L, [.25, .5, .75, .95, 1]).tolist(), R6.cross_mid.sum()))
print("   |slope| per h of sloped stretches: median %.4f, quantiles %s"
      % (R6.slope[~R6.const].abs().median(), np.round(np.quantile(R6.slope[~R6.const].abs(), [.1, .5, .9]), 4).tolist()))
print("   by farm rows:", y[S].farm.value_counts().to_dict(), " late(>=179) share %.1f%% (all rows %.1f%%)"
      % (100 * (y.day[S] >= 179).mean(), 100 * (y.day >= 179).mean()))
print("   hour share of straight rows:", (y[S].hour.value_counts(normalize=True).sort_index() * 100).round(1).to_dict())
cov = y.groupby(["farm", "day"]).apply(lambda g: (g.L >= 6).mean(), include_groups=False)
print("   per-day coverage: days with 0%% %d, 1-49%% %d, 50-99%% %d, 100%% %d"
      % ((cov == 0).sum(), ((cov > 0) & (cov < .5)).sum(), ((cov >= .5) & (cov < 1)).sum(), (cov == 1).sum()))
print("   level of straight rows mean EC %.3f vs others %.3f" % (y.sub_ec[S].mean(), y.sub_ec[~S].mean()))

# 2. physics: within-day partial relation with sub_temp
y["lec"] = np.log(y.sub_ec.clip(lower=1e-3))
y["dl"] = y.lec - y.groupby(["farm", "day"]).lec.transform("mean")
y["dt"] = y.sub_temp - y.groupby(["farm", "day"]).sub_temp.transform("mean")
for nm, m in (("straight-stretch rows", S), ("other rows", ~S)):
    h = y[m]
    b = (h.dl * h.dt).sum() / (h.dt ** 2).sum()
    print("2. %-22s pooled within-day slope dlogEC/dT %.4f  corr %.3f  (n=%d)" % (nm, b, np.corrcoef(h.dl, h.dt)[0, 1], len(h)))
# day-level: fully straight days vs no-straight days
fd = cov[cov >= .99].index
nd = cov[cov == 0].index
for nm, idx in (("days >=99% straight", fd), ("days with no straight", nd)):
    sl = []
    for k in idx:
        g = y[(y.farm == k[0]) & (y.day == k[1])]
        if g.dt.std() > 0.3:
            sl.append(np.polyfit(g.dt, g.dl, 1)[0])
    sl = np.array(sl)
    print("   %-22s n=%d  median slope %.4f  positive %.0f%%" % (nm, len(sl), np.median(sl) if len(sl) else np.nan,
                                                              100 * (sl > 0).mean() if len(sl) else np.nan))
# residual of EC around its straight line vs temperature: inside sloped stretches the residual is 0 by
# construction, so instead check the 2nd-difference relation: real data -> d2(logEC) tracks d2(T)
y["d2e"] = y.groupby("farm").sub_ec.diff().groupby(y.farm).diff()
y["d2t"] = y.groupby("farm").sub_temp.diff().groupby(y.farm).diff()
for nm, m in (("straight", S), ("other", ~S)):
    h = y[m & y.d2e.notna() & y.d2t.notna()]
    print("   corr(d2 EC, d2 sub_temp) %-8s %.3f (n=%d)" % (nm, np.corrcoef(h.d2e, h.d2t)[0, 1], len(h)))

# 3. control simulation: shuffle hourly increments within each non-straight day and count >= 6-h straight runs
inc = []
for (f, d), g in y[~S].groupby(["farm", "day"]):
    if len(g) == 24:
        inc.append(np.diff(g.sub_ec.values))
inc = np.array(inc)
sim_rows = 0
sim_tot = 0
for _ in range(200):
    k = RNG.integers(0, len(inc))
    dd = RNG.permutation(inc[k])
    v = np.r_[0.5, 0.5 + np.cumsum(dd)]
    v = np.round(v, 3)
    rr = runs(v, np.arange(24), 0.001)
    sim_rows += sum(b - a + 1 for a, b in rr if b - a + 1 >= 6)
    sim_tot += 24
print("\n3. control: shuffled real increments of non-straight days -> rows in >=6h straight runs %.1f%%"
      % (100 * sim_rows / sim_tot))

# 4. EC v2 error
o = pd.read_csv(os.path.join(env.ROOT, u"집", u"코덱스", "local", "ec_restart_phase3_20261001_v1",
                             "oof_predictions.csv"), encoding="utf-8-sig")
o = o[o.validator == "DIAG10"].merge(y[["row_id", "L", "const"]], on="row_id", how="left")
o["e"] = o.v2 - o.sub_ec
o["e_lvl"] = o.groupby(["farm", "day"]).e.transform("mean")
o["e_shp"] = o.e - o.e_lvl
r = lambda e: float(np.sqrt(np.mean(np.square(e))))
for nm, m in (("straight", o.L >= 6), ("other", o.L < 6)):
    h = o[m]
    print("4. EC v2 DIAG10 %-8s rows %d RMSE %.3f  level part %.3f  shape part %.3f"
          % (nm, len(h), r(h.e), r(h.e_lvl), r(h.e_shp)))

# 5. examples
print("\n5. examples (longest stretches)")
for _, rr in R6.sort_values("L", ascending=False).head(4).iterrows():
    g = y[(y.farm == rr.farm) & (y.t >= rr.t0 - 2) & (y.t <= rr.t0 + rr.L + 1)]
    print("  %s day %d hour %d len %d const %s" % (rr.farm, rr.t0 // 24, rr.t0 % 24, rr.L, rr.const))
    print("    sub_ec  ", g.sub_ec.round(3).tolist())
    print("    sub_temp", g.sub_temp.round(2).tolist())
y[["row_id", "L", "const", "slope"]].to_csv(os.path.join(env.LOCAL, "rd5_ec_straight_rows.csv"), index=False)
