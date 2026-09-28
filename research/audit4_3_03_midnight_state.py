# -*- coding: utf-8 -*-
"""Inspector 4-3, step 3: substrate initial state at the midnight source
switch.

Physics: the slab lags air by ~3-4 h, so at 00-03 h it still carries the
TRUE source's previous evening.  In the spliced record the previous evening
(d-1) usually belongs to another source; the same source is often d-2.
  1. which earlier evening (d-1 or d-2, 20-23 h) best matches in_temp at 00 h
     (inputs only; train vs test)
  2. on the 49 unspliced greenhouses: sub(0)-air(0) vs air(prev eve)-air(0)
  3. OOF temperature residual at 00-05 h (F60ND__DIAG10) vs the same gap
     computed from d-1 and from d-2
Run:  cd research && PYTHONPATH="" <python> -u audit4_3_03_midnight_state.py
"""
import env  # noqa: F401
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import common

pd.set_option("display.width", 200)


def eve(a, lag):
    """mean in_temp 20-23 h of day d-lag, same farm, mapped to (farm, day)."""
    e = a[a.hour.between(20, 23)].groupby(["farm", "day"]).in_temp.mean().rename("e").reset_index()
    e["day"] = e.day + lag
    return e.rename(columns={"e": "eve%d" % lag})


def main():
    tX, ty, sX = common.load_raw()
    b = pd.concat([tX.assign(te=False), sX.assign(te=True)], ignore_index=True)
    b = b.merge(ty[["row_id", "sub_temp"]], on="row_id", how="left")
    h0 = b[b.hour == 0][["farm", "day", "te", "in_temp", "sub_temp", "out_temp"]].rename(
        columns={"in_temp": "a0", "sub_temp": "s0", "out_temp": "o0"})
    for lag in (1, 2, 3):
        h0 = h0.merge(eve(b, lag), on=["farm", "day"], how="left")
    last23 = b[b.hour == 23][["farm", "day", "in_temp"]].copy()
    for lag in (1, 2):
        x = last23.copy(); x["day"] += lag
        h0 = h0.merge(x.rename(columns={"in_temp": "a23_%d" % lag}), on=["farm", "day"], how="left")

    print("== 1. |a0 - a23(d-lag)| and |a0 - eve(d-lag)| medians ==")
    tg = h0[h0.farm.isin(["F13", "F47"])]
    for nm, s in [("train", tg[~tg.te]), ("test", tg[tg.te]), ("other49", h0[~h0.farm.isin(["F13", "F47"])])]:
        print("  %-8s n=%4d | 23h d-1 %.2f d-2 %.2f | eve d-1 %.2f d-2 %.2f d-3 %.2f | d-2 closer than d-1: %.2f"
              % (nm, len(s), (s.a0 - s.a23_1).abs().median(), (s.a0 - s.a23_2).abs().median(),
                 (s.a0 - s.eve1).abs().median(), (s.a0 - s.eve2).abs().median(), (s.a0 - s.eve3).abs().median(),
                 ((s.a0 - s.a23_2).abs() < (s.a0 - s.a23_1).abs()).mean()))
    print("  test by farm/day:")
    tt = tg[tg.te].sort_values(["farm", "day"])
    print("   ", "  ".join("%s%d:%+.1f/%+.1f" % (r.farm[1:], r.day, r.a0 - r.a23_1, r.a0 - r.a23_2) for r in tt.itertuples()))

    print("\n== 2. unspliced greenhouses: s0-a0 vs eve1-a0 ==")
    o = h0[~h0.farm.isin(["F13", "F47"])].dropna(subset=["s0", "eve1", "a0"])
    x, y = (o.eve1 - o.a0).values, (o.s0 - o.a0).values
    X = np.column_stack([np.ones(len(x)), x])
    print("  other49 pooled: slope %.3f  corr %.3f n=%d" % (np.linalg.lstsq(X, y, rcond=None)[0][1], np.corrcoef(x, y)[0, 1], len(x)))
    for f, s in [("F13/F47 train", tg[~tg.te & tg.s0.notna()])]:
        for lag in (1, 2):
            s2 = s.dropna(subset=["eve%d" % lag, "a0", "s0"])
            x, y = (s2["eve%d" % lag] - s2.a0).values, (s2.s0 - s2.a0).values
            print("  %s eve%d: slope %.3f corr %.3f n=%d" % (f, lag, np.linalg.lstsq(np.column_stack([np.ones(len(x)), x]), y, rcond=None)[0][1],
                                                          np.corrcoef(x, y)[0, 1], len(x)))

    print("\n== 3. OOF residual (F60ND__DIAG10) at 00-05 h vs eve gaps ==")
    z = np.load(env.LOCAL + "/eval_v6_oof.npz", allow_pickle=True)
    oof = pd.Series(z["F60ND__DIAG10"], index=z["row_id"])
    L = b[b.farm.isin(["F13", "F47"]) & ~b.te & b.sub_temp.notna()].copy()
    L["p"] = oof.reindex(L.row_id).values
    L["r"] = L.sub_temp - L.p
    L = L.merge(h0[["farm", "day", "a0", "eve1", "eve2", "eve3"]], on=["farm", "day"], how="left")
    for hh in range(0, 7):
        s = L[(L.hour == hh)].dropna(subset=["eve1", "eve2", "eve3", "a0", "r"])
        print("  h%02d n=%d rmse %.3f | rho(r, eve1-a0) %+.3f | rho(r, eve2-a0) %+.3f | rho(r, eve3-a0) %+.3f"
              % (hh, len(s), np.sqrt((s.r ** 2).mean()), spearmanr(s.r, s.eve1 - s.a0).correlation,
                 spearmanr(s.r, s.eve2 - s.a0).correlation, spearmanr(s.r, s.eve3 - s.a0).correlation))
    # day-level residual too
    dr = L.groupby(["farm", "day"]).agg(r=("r", "mean"), a0=("a0", "first"), e1=("eve1", "first"), e2=("eve2", "first")).dropna()
    print("  day-mean residual: rho eve1-a0 %+.3f  eve2-a0 %+.3f"
          % (spearmanr(dr.r, dr.e1 - dr.a0).correlation, spearmanr(dr.r, dr.e2 - dr.a0).correlation))


if __name__ == "__main__":
    main()
