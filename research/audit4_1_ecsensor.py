# -*- coding: utf-8 -*-
"""Inspector 4-1: is sub_ec really uncompensated raw EC (catalog 6.27)?  Sharper tests.
Read-only; stdout only."""
import env  # noqa: F401
import numpy as np
import pandas as pd
from harness import load

panel, _, lab = load()
d = lab[["farm", "day", "hour", "t", "sub_ec", "sub_temp", "in_temp", "out_rad", "in_hum", "act_vent", "act_heating"]].copy()
d = d[d.sub_ec > 0].sort_values(["farm", "t"]).reset_index(drop=True)
d["lec"] = np.log(d.sub_ec)
key = [d.farm, d.day]


def dm(s, keys):
    return s - s.groupby(keys).transform("mean")


def slope(x, y):
    m = ~(np.isnan(x) | np.isnan(y))
    x, y = x[m], y[m]
    return float(np.sum(x * y) / np.sum(x * x))


def ols(Y, X):
    m = ~np.isnan(Y) & ~np.isnan(X).any(1)
    b, *_ = np.linalg.lstsq(X[m], Y[m], rcond=None)
    return b


# A. replicate within-day slope
le_w, T_w = dm(d.lec, key), dm(d.sub_temp, key)
print("A. within-day slope dlogEC/dT = %.4f" % slope(T_w.values, le_w.values))
# per-day corr
cc = d.assign(lw=le_w, tw=T_w).groupby(["farm", "day"]).apply(lambda g: g.lw.corr(g.tw) if g.tw.std() > 0 else np.nan)
print("   per-day corr median %.2f, share>0 %.0f%%" % (cc.median(), 100 * (cc > 0).mean()))

# B. day FE + hour FE (two-way demeaning by alternating projections)
def twoway(s):
    x = s.copy()
    for _ in range(30):
        x = x - x.groupby(key).transform("mean")
        x = x - x.groupby(d.hour).transform("mean")
    return x
print("B. day FE + hour-of-day FE slope = %.4f  (removes shared diurnal co-movement)" % slope(twoway(d.sub_temp).values, twoway(d.lec).values))
# B2 horse race: sub_temp vs in_temp vs out_rad within day+hour FE
X = np.column_stack([twoway(d.sub_temp).values, twoway(d.in_temp).values, twoway(d.out_rad / 100).values])
print("B2. two-way FE multivariate: sub_temp %.4f | in_temp %.4f | out_rad/100 %.4f" % tuple(ols(twoway(d.lec).values, X)))

# C. night first differences (contiguous hours, same day, 01-05h)
g = d.groupby(["farm", "day"])
d["dle"] = g.lec.diff()
d["dT"] = g.sub_temp.diff()
d["cont"] = g.t.diff() == 1
night = d.hour.between(1, 5) & d.cont
n = d[night]
print("\nC. night 01-05h first differences: n=%d" % len(n))
print("   slope dlogEC on dT = %.4f" % slope(n.dT.values, n.dle.values))
for nm, m in (("dT>+0.1", n.dT > 0.1), ("|dT|<=0.1", n.dT.abs() <= 0.1), ("dT<-0.1", n.dT < -0.1), ("dT<-0.5", n.dT < -0.5)):
    print("   %-10s n=%4d mean dlogEC %+.4f  mean dT %+.3f  ratio %.4f" % (nm, m.sum(), n.dle[m].mean(), n.dT[m].mean(),
          n.dle[m].mean() / n.dT[m].mean() if abs(n.dT[m].mean()) > 0.05 else np.nan))
# C2 all hours, first differences, hour FE on the diffs (removes the systematic hourly trend of both)
a = d[d.cont & (d.hour != 0)].copy()
a["dle_h"] = a.dle - a.groupby("hour").dle.transform("mean")
a["dT_h"] = a.dT - a.groupby("hour").dT.transform("mean")
print("C2. all-hour first diffs, hour-demeaned: slope %.4f (n=%d)" % (slope(a.dT_h.values, a.dle_h.values), len(a)))
for hb in ((1, 5), (6, 9), (10, 15), (16, 23)):
    m = a.hour.between(*hb)
    print("    hours %02d-%02d slope %.4f  (raw slope %.4f)" % (hb + (slope(a.dT_h[m].values, a.dle_h[m].values),
                                                             slope(a.dT[m].values, a.dle[m].values))))
# C3 heating events at night: hours where heating rises by >20 and sub_temp rises
a["dheat"] = a.groupby(["farm", "day"]).act_heating.diff()
m = a.hour.between(19, 23) | a.hour.between(1, 6)
up = m & (a.dT > 0.2)
print("C3. night hours with sub_temp rising >0.2: n=%d, mean dlogEC %+.4f (vs night falling hours %+.4f)"
      % (up.sum(), a.dle[up].mean(), a.dle[m & (a.dT < -0.2)].mean()))

# D. lead/lag profile inside day (sensor effect should peak at lag 0 on sub_temp)
print("\nD. within-day slope of logEC(t) on sub_temp(t+k) (day-demeaned, same day only):")
for k in (-3, -2, -1, 0, 1, 2, 3):
    Tk = d.groupby(["farm", "day"]).sub_temp.shift(-k)
    ok = Tk.notna()
    x = dm(Tk[ok], [d.farm[ok], d.day[ok]]); y = dm(d.lec[ok], [d.farm[ok], d.day[ok]])
    print("   k=%+d  slope %.4f  corr %.3f" % (k, slope(x.values, y.values), np.corrcoef(x, y)[0, 1]))
print("   same with in_temp(t+k):")
for k in (-3, -1, 0, 1):
    Tk = d.groupby(["farm", "day"]).in_temp.shift(-k)
    ok = Tk.notna()
    x = dm(Tk[ok], [d.farm[ok], d.day[ok]]); y = dm(d.lec[ok], [d.farm[ok], d.day[ok]])
    print("   k=%+d  corr %.3f" % (k, np.corrcoef(x, y)[0, 1]))

# E. across-day: daily mean log EC vs daily mean sub_temp (compensation prediction ~ +0.02 if level were thermal)
dd = d.groupby(["farm", "day"]).agg(lec=("lec", "mean"), Tm=("sub_temp", "mean")).reset_index()
print("\nE. across-day slope (daily means) %.4f ; with 15-day rolling demeaning per farm:" % slope(dm(dd.Tm, dd.farm).values, dm(dd.lec, dd.farm).values))
dd = dd.sort_values(["farm", "day"])
for w in (7, 15):
    rT = dd.Tm - dd.groupby("farm").Tm.transform(lambda s: s.rolling(w, center=True, min_periods=3).mean())
    rE = dd.lec - dd.groupby("farm").lec.transform(lambda s: s.rolling(w, center=True, min_periods=3).mean())
    print("   window %d: slope %.4f corr %.3f" % (w, slope(rT.values, rE.values), np.corrcoef(rT.fillna(0), rE.fillna(0))[0, 1]))

# F. per-farm and per-sealed split of within-day slope
for f in ("F13", "F47"):
    m = (d.farm == f).values
    print("F. %s within-day slope %.4f | two-way %.4f" % (f, slope(T_w.values[m], le_w.values[m]),
          slope(twoway(d.sub_temp).values[m], twoway(d.lec).values[m])))
# G. does the round-3 OOF within-day error correlate with T deviation? (room for a shape fix)
z = np.load(env.LOCAL + "/oof_ec_diag.npz", allow_pickle=True)
p = pd.Series(z["oof"], index=z["row_id"]).loc[lab.row_id].values
lab2 = lab.assign(p=p)
lab2 = lab2[lab2.sub_ec > 0]
k2 = [lab2.farm, lab2.day]
r = np.log(lab2.sub_ec) - np.log(np.clip(lab2.p, 0.05, None))
print("\nG. OOF log-residual (y/p) within-day vs T deviation: slope %.4f corr %.3f"
      % (slope(dm(lab2.sub_temp, k2).values, dm(r, k2).values), np.corrcoef(dm(lab2.sub_temp, k2), dm(r, k2))[0, 1]))
pw = np.log(np.clip(lab2.p, 0.05, None))
print("   OOF prediction itself within-day slope on T %.4f (label %.4f)" % (slope(dm(lab2.sub_temp, k2).values, dm(pw, k2).values),
      slope(dm(lab2.sub_temp, k2).values, dm(np.log(lab2.sub_ec), k2).values)))
