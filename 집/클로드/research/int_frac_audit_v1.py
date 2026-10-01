# -*- coding: utf-8 -*-
"""IF1: in the 48 integer-resolution farms, ~1.5-7% of in_temp rows carry a
decimal.  Are those rows filled-in (interpolated / restored) values, or real
measurements at a finer resolution?  Descriptive audit, 2026-10-02 집 클로드
(user question).

Checks
  1. decimal digits and fractional-part distribution of the fractional rows
  2. run structure: length of consecutive fractional-row runs
  3. linear-interpolation test: for each run, rebuild the values by a straight
     line between the integer rows just before and after; share of runs whose
     values match within 0.051 (one decimal of rounding)
  4. same rows in in_hum / in_co2 / in_rad: missing? constant? (were the other
     channels touched at the same time?)
  5. labels: sub_temp missing / fractional on those rows
  6. time of day and farm-day clustering

Run:  cd 집/클로드/research && PYTHONPATH="" py -3.12 -u int_frac_audit_v1.py
"""
import env  # noqa: F401
import os

import numpy as np
import pandas as pd

X = pd.read_csv(os.path.join(env.DATA, "train_X.csv"))
Y = pd.read_csv(os.path.join(env.DATA, "train_y.csv"))
X = X.merge(Y, on="row_id", how="left")
p = X.row_id.str.split("_", expand=True)
X["farm"], X["day"], X["hour"] = p[0], p[1].astype(int), p[2].astype(int)
X = X[~X.farm.isin(["F13", "F47", "F32"])].sort_values(["farm", "day", "hour"]).reset_index(drop=True)
X["t"] = X.day * 24 + X.hour


def isfrac(s):
    return (s.notna()) & (np.abs(s - np.round(s)) > 1e-9)


X["fr"] = isfrac(X.in_temp)
F = X[X.fr]
print("farms", X.farm.nunique(), "rows", len(X), "fractional in_temp rows", len(F), "(%.2f%%)" % (100 * len(F) / X.in_temp.notna().sum()))

# 1. decimals
dec = F.in_temp.astype(str).str.split(".").str[1].str.rstrip("0").str.len()
print("\n1. decimal digits:", dec.value_counts().sort_index().to_dict())
fp = np.round(F.in_temp - np.floor(F.in_temp), 4)
print("   fractional part top 12:", fp.value_counts().head(12).round(4).to_dict())

# 2/3. runs within a farm, contiguous hours
runs = []
for f, g in X.groupby("farm"):
    t = g.t.values
    v = g.in_temp.values
    fr = g.fr.values
    i = 0
    n = len(g)
    while i < n:
        if not fr[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and fr[j + 1] and t[j + 1] == t[j] + 1:
            j += 1
        # anchors: immediately adjacent integer rows (contiguous in time)
        L = i - 1 if i - 1 >= 0 and t[i - 1] == t[i] - 1 and not np.isnan(v[i - 1]) else None
        R = j + 1 if j + 1 < n and t[j + 1] == t[j] + 1 and not np.isnan(v[j + 1]) else None
        match = np.nan
        if L is not None and R is not None:
            line = v[L] + (v[R] - v[L]) * (t[i:j + 1] - t[L]) / (t[R] - t[L])
            match = float(np.max(np.abs(line - v[i:j + 1])))
        runs.append(dict(farm=f, start=t[i], len=j - i + 1, maxdev=match,
                         gapL=(L is None), gapR=(R is None),
                         nanL=(i - 1 >= 0 and t[i - 1] == t[i] - 1 and np.isnan(v[i - 1]))))
        i = j + 1
R = pd.DataFrame(runs)
print("\n2. runs:", len(R), " length distribution:", R.len.value_counts().sort_index().head(15).to_dict())
print("   runs at a time gap / missing neighbour (left gap %d, right gap %d, left neighbour NaN %d)"
      % (R.gapL.sum(), R.gapR.sum(), R.nanL.sum()))
ok = R.maxdev.notna()
print("\n3. linear-interpolation test on %d runs with both integer neighbours:" % ok.sum())
for thr in (0.051, 0.11, 0.5):
    print("   max |value - straight line| <= %.3f : %.1f%%" % (thr, 100 * (R.maxdev[ok] <= thr).mean()))
print("   by run length (share matching <= 0.051):",
      R[ok].groupby("len").maxdev.apply(lambda s: "%.0f%% (n=%d)" % (100 * (s <= .051).mean(), len(s))).head(10).to_dict())

# 4. other channels on the same rows
for c in ("in_hum", "in_co2", "in_rad"):
    a = X.loc[X.fr, c]
    b = X.loc[~X.fr & X.in_temp.notna(), c]
    print("\n4. %-6s missing on frac rows %.1f%% vs others %.1f%%; fractional %.2f%% vs %.2f%%"
          % (c, 100 * a.isna().mean(), 100 * b.isna().mean(), 100 * isfrac(a).mean(), 100 * isfrac(b).mean()))
# does in_hum/in_co2 also follow a straight line on those runs? (|2nd difference| == 0)
X["d2h"] = X.groupby("farm").in_hum.diff().groupby(X.farm).diff().abs()
X["d2c"] = X.groupby("farm").in_co2.diff().groupby(X.farm).diff().abs()
print("   share of rows with 2nd difference 0:  in_hum frac %.1f%% vs others %.1f%% | in_co2 frac %.1f%% vs others %.1f%%"
      % (100 * (X.d2h[X.fr] == 0).mean(), 100 * (X.d2h[~X.fr] == 0).mean(),
         100 * (X.d2c[X.fr] == 0).mean(), 100 * (X.d2c[~X.fr] == 0).mean()))

# 5. labels
print("\n5. sub_temp missing on frac rows %.1f%% vs others %.1f%%; sub_temp fractional %.2f%% vs %.2f%%"
      % (100 * X.sub_temp[X.fr].isna().mean(), 100 * X.sub_temp[~X.fr].isna().mean(),
         100 * isfrac(X.sub_temp[X.fr]).mean(), 100 * isfrac(X.sub_temp[~X.fr]).mean()))
if X.fr.any():
    lab = X[X.fr & X.sub_temp.notna()]
    print("   |sub_temp - in_temp| frac rows %.2f vs others %.2f"
          % ((lab.sub_temp - lab.in_temp).abs().mean(),
             (X[~X.fr & X.sub_temp.notna()].sub_temp - X[~X.fr & X.sub_temp.notna()].in_temp).abs().mean()))

# 6. hour / day clustering
print("\n6. hour share of frac rows:", (F.hour.value_counts(normalize=True).sort_index() * 100).round(1).to_dict())
fd = X.groupby(["farm", "day"]).fr.sum()
print("   farm-days with any frac row: %d of %d; frac rows on days with >=6 frac rows: %.1f%%"
      % ((fd > 0).sum(), len(fd), 100 * fd[fd >= 6].sum() / fd.sum()))
print("   per-farm-day frac count distribution:", fd[fd > 0].value_counts().sort_index().head(25).to_dict())

# examples
print("\nexamples (3 longest runs with both neighbours):")
for _, r in R[ok].sort_values("len", ascending=False).head(3).iterrows():
    g = X[(X.farm == r.farm) & (X.t >= r.start - 2) & (X.t <= r.start + r.len + 1)]
    print(r.farm, "day", int(r.start // 24), "hour", int(r.start % 24), "len", int(r.len), "maxdev %.3f" % r.maxdev)
    print("   in_temp", g.in_temp.round(2).tolist())
    print("   in_hum ", g.in_hum.tolist())
    print("   in_co2 ", g.in_co2.tolist())
    print("   sub_temp", g.sub_temp.tolist())
