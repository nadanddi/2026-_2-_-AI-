# -*- coding: utf-8 -*-
"""EC gap analysis, step 4: on sealed / fan-off days, what sets the EC level?

Sealed day (catalog 6.11): daily mean circulation fan < 10 and share of
hours with the vent at 0 > 0.85.  Training: 59 of 400 days, EC 0.39-1.8.
Test: 21 of 60 days.

Candidate day-level quantities, all computed from INPUTS of the same record
up to and including the day itself (test days' inputs count as history; no
label is used):
  streak2   sealed days in a row along the every-other-day chain (d, d-2, ...)
  streak1   sealed days in a row along consecutive days (d, d-1, ...)
  prev2_sealed / prev1_sealed
  rad2_k    sum of daily mean out_rad over d, d-2, ..., (k days of the chain)
  plus same-day summaries (in_temp, out_rad, in_hum, VPD proxy, heating,
  shade, CO2, thermal screen)

Reported on training sealed days: Spearman correlation with the daily mean
EC and with the round-3 OOF residual (what the model misses), with a
permutation p-value; the same on all days for the chain quantities.

Run:  cd research && PYTHONPATH="" <python> -u anal_e4_sealed_level.py
"""
import env  # noqa: F401
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from common import TARGET_FARMS
from harness import load
from anal_e3_spikes import day_table


def sealed(d):
    return (d.act_circfan_mean < 10) & (d.act_vent_zero > 0.85)


def chain_feats(D):
    out = []
    for f in TARGET_FARMS:
        g = D[D.farm == f].set_index("day").sort_index()
        s = sealed(g)
        rad = g.out_rad_mean
        for d in g.index:
            r = dict(farm=f, day=d)
            for step, nm in ((2, "streak2"), (1, "streak1")):
                k, x = 0, d
                while x in s.index and bool(s[x]):
                    k += 1
                    x -= step
                r[nm] = k
            r["prev2_sealed"] = float(s.get(d - 2, np.nan)) if (d - 2) in s.index else np.nan
            r["prev1_sealed"] = float(s.get(d - 1, np.nan)) if (d - 1) in s.index else np.nan
            for k in (2, 3, 5):
                v = [rad.get(d - 2 * i, np.nan) for i in range(k)]
                r["rad2_%d" % k] = np.nansum(v) if not np.all(np.isnan(v)) else np.nan
            v = [g.in_temp_mean.get(d - 2 * i, np.nan) for i in range(3)]
            r["temp2_3"] = np.nanmean(v)
            out.append(r)
    return pd.DataFrame(out)


def perm_p(x, y, n=2000, seed=0):
    m = ~(np.isnan(x) | np.isnan(y))
    x, y = x[m], y[m]
    r0 = spearmanr(x, y).correlation
    rng = np.random.RandomState(seed)
    null = np.array([spearmanr(x, rng.permutation(y)).correlation for _ in range(n)])
    return r0, float((np.abs(null) >= abs(r0)).mean()), int(m.sum())


def main():
    panel, _, lab = load()
    z = np.load(env.LOCAL + "/oof_ec_diag.npz", allow_pickle=True)
    oof = pd.Series(z["oof"], index=z["row_id"])
    D = day_table(panel)                         # train + test inputs
    D["sealed"] = sealed(D)
    D = D.merge(chain_feats(D), on=["farm", "day"])
    lab = lab.copy()
    lab["p"] = oof.loc[lab.row_id].values
    lv = lab.groupby(["farm", "day"]).agg(y=("sub_ec", "mean"), p=("p", "mean")).reset_index()
    D = D.merge(lv, on=["farm", "day"], how="left")
    D["res"] = D.y - D.p
    D["vpd_proxy"] = D.in_temp_mean * (1 - D.in_hum_mean / 100.0)

    cand = ["streak2", "streak1", "prev2_sealed", "prev1_sealed", "rad2_2", "rad2_3", "rad2_5", "temp2_3",
            "day", "in_temp_mean", "in_temp_max", "out_rad_mean", "in_hum_mean", "vpd_proxy",
            "act_heating_mean", "act_shade_mean", "act_shade_zero", "in_co2_mean", "act_co2_mean",
            "act_thermal_mean", "act_fog_mean", "out_temp_mean", "dT_in_out", "day_night_T"]
    tr = D[D.y.notna()]
    S = tr[tr.sealed]
    print("training sealed days %d (F13 %d, F47 %d)" % (len(S), (S.farm == "F13").sum(), (S.farm == "F47").sum()))
    rows = []
    for c in cand:
        r1, p1, n1 = perm_p(S[c].values.astype(float), S.y.values)
        r2, p2, _ = perm_p(S[c].values.astype(float), S.res.values)
        rows.append(dict(feat=c, n=n1, rho_EC=r1, p_EC=p1, rho_res=r2, p_res=p2))
    R = pd.DataFrame(rows).sort_values("p_EC")
    print("\n-- sealed days: Spearman with daily EC and with OOF residual (perm p) --")
    print(R.round(3).to_string(index=False))
    print("(24 candidates: expect ~1 with p<0.05 by chance)")

    print("\n-- streak2 on sealed days --")
    print(S.groupby(S.streak2.clip(upper=4)).agg(n=("y", "size"), EC=("y", "mean"), pred=("p", "mean"),
                                                   res=("res", "mean")).round(3).to_string())
    print("\n-- all training days: EC by (sealed, prev2_sealed) --")
    print(tr.groupby(["sealed", "prev2_sealed"]).agg(n=("y", "size"), EC=("y", "mean"), pred=("p", "mean"),
                                                        res=("res", "mean")).round(3).to_string())

    te = D[D.y.isna() & D.day.ge(180)]
    print("\n-- test sealed days: chain context --")
    print(te[te.sealed][["farm", "day", "streak2", "streak1", "prev2_sealed", "in_temp_mean", "out_rad_mean",
                         "act_heating_mean"]].round(2).to_string(index=False))
    print("\nsealed-day streak2 distribution: train %s | test %s"
          % (S.streak2.value_counts().sort_index().to_dict(), te[te.sealed].streak2.value_counts().sort_index().to_dict()))


if __name__ == "__main__":
    main()
