# -*- coding: utf-8 -*-
"""Inspector 4-3, step 1: actuator semantics, "injected noise" vs real sensor
behaviour, and what "sealed / fan-off" days really are.

Discriminating tests
  noise   injected i.i.d. noise on in_temp / in_hum breaks the physical coupling
          of hourly changes (dT up -> dRH down at fixed vapour content) and the
          substrate's response to air changes.  Real poor-mixing / dosing noise
          keeps both.  CO2 roughness vs act_co2 / circfan / vent.
  sealed  weather (cold, dark) vs source (actuator fingerprint) vs day index.
Run:  cd research && PYTHONPATH="" <python> -u audit4_3_01_noise_sealed.py
"""
import env  # noqa: F401
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import common
import train_flags_v6 as TF

pd.set_option("display.width", 200)


def main():
    tX, ty, sX = common.load_raw()
    tX = tX.merge(ty[["row_id", "sub_temp", "sub_ec"]], on="row_id", how="left")
    tX["is_test"] = False
    sX = sX.copy(); sX["is_test"] = True
    a = pd.concat([tX, sX], ignore_index=True).sort_values(["farm", "t"]).reset_index(drop=True)
    tg = a[a.farm.isin(["F13", "F47"])].copy()

    print("== actuator semantics by hour (F13/F47, all rows) ==")
    tg["dT"] = tg.in_temp - tg.out_temp
    print(tg.groupby("hour")[["act_thermal", "act_shade", "act_vent", "act_circfan", "act_heating", "act_fog", "act_co2", "out_rad", "dT"]]
          .mean().round(1).iloc[::2].to_string())
    n = tg[(tg.hour <= 4) | (tg.hour >= 21)]
    n = n.assign(thb=pd.cut(n.act_thermal, [-1, 1, 50, 99, 100]))
    print("\nnight in-out dT by thermal opening:\n", n.groupby("thb", observed=True).dT.agg(["size", "mean"]).round(2).to_string())
    d = tg[tg.hour.between(10, 14)]
    d = d.assign(shb=pd.cut(d.act_shade, [-1, 1, 50, 99, 100]))
    print("\nmidday by shade opening (0=?):\n", d.groupby("shb", observed=True)[["out_rad", "in_temp", "dT", "act_vent"]]
          .agg(["size", "mean"]).round(1).to_string())
    print("farms with shade/thermal columns:", a.groupby("farm").act_shade.apply(lambda s: s.notna().mean()).gt(0).sum())

    # ---------- day table ----------
    rows = []
    for (f, dd), g in a.groupby(["farm", "day"]):
        g = g.sort_values("t")
        cont = g.t.diff() == 1
        dT, dH, dC = g.in_temp.diff()[cont], g.in_hum.diff()[cont], g.in_co2.diff()[cont]
        r = dict(farm=f, day=int(dd), is_test=bool(g.is_test.iloc[0]))
        ok = dT.notna() & dH.notna()
        r["c_dTdH"] = np.corrcoef(dT[ok], dH[ok])[0, 1] if ok.sum() > 8 and dT[ok].std() > 0 and dH[ok].std() > 0 else np.nan
        r["co2_ac1"] = dC.autocorr(1) if dC.notna().sum() >= 8 else np.nan
        r["T_ac1"] = dT.autocorr(1) if dT.notna().sum() >= 8 else np.nan
        r["H_ac1"] = dH.autocorr(1) if dH.notna().sum() >= 8 else np.nan
        if "act_co2" in g and g.act_co2.notna().any():
            ac = g.act_co2.diff()[cont]
            okc = ac.notna() & dC.notna()
            r["c_dC_dose"] = np.corrcoef(ac[okc], dC[okc])[0, 1] if okc.sum() > 8 and ac[okc].std() > 0 else np.nan
        for c in ["out_temp", "out_rad", "in_temp", "act_circfan", "act_vent", "act_heating", "act_co2", "act_shade", "act_fog", "act_thermal"]:
            r[c] = g[c].mean()
        r["vent0"] = (g.act_vent.fillna(0) == 0).mean()
        r["fan0"] = (g.act_circfan.fillna(0) == 0).mean()
        r["ec"] = g.sub_ec.mean(); r["st"] = g.sub_temp.mean()
        # substrate response to high-frequency air change (diagnostic, labels)
        hf = (g.in_temp - g.in_temp.rolling(5, center=True, min_periods=3).mean())
        sr = g.sub_temp.shift(-2) - g.sub_temp.rolling(5, center=True, min_periods=3).mean().shift(-2)
        oks = hf.notna() & sr.notna()
        r["c_hf_sub"] = np.corrcoef(hf[oks], sr[oks])[0, 1] if oks.sum() > 8 and hf[oks].std() > 0 and sr[oks].std() > 0 else np.nan
        rows.append(r)
    D = pd.DataFrame(rows)
    nd = TF.noisy_days()[["farm", "day", "noisy", "ac1", "d2"]]
    D = D.merge(nd, on=["farm", "day"], how="left")
    D["grp"] = np.where(D.farm.isin(["F13", "F47"]),
                        np.where(D.is_test, "test", np.where(D.noisy.fillna(False), "train_noisy", "train_clean")), "other49")
    D["sealed"] = (D.act_circfan < 10) & (D.vent0 > 0.85)
    D.to_csv(env.LOCAL + "/audit4_3_01_days.csv", index=False)

    print("\n== noise discrimination (medians) ==")
    print(D.groupby("grp")[["co2_ac1", "T_ac1", "H_ac1", "c_dTdH", "c_hf_sub", "c_dC_dose"]].median().round(3).to_string())
    print("share of days with c_dTdH > -0.2:\n", D.groupby("grp").c_dTdH.apply(lambda s: (s > -0.2).mean()).round(3).to_string())
    tgD = D[D.farm.isin(["F13", "F47"])]
    print("\nwithin F13/F47 days: spearman(co2_ac1, c_dTdH) %.3f  (co2_ac1, T_ac1) %.3f  (co2_ac1, c_hf_sub) %.3f"
          % (spearmanr(tgD.co2_ac1, tgD.c_dTdH, nan_policy="omit").correlation,
             spearmanr(tgD.co2_ac1, tgD.T_ac1, nan_policy="omit").correlation,
             spearmanr(tgD.co2_ac1, tgD.c_hf_sub, nan_policy="omit").correlation))
    print("\nnoisy vs clean training days, conditions (means):")
    print(tgD.groupby("grp")[["out_temp", "out_rad", "act_circfan", "fan0", "vent0", "act_co2", "act_heating", "sealed"]].mean().round(2).to_string())

    print("\n== sealed days ==")
    print(tgD.groupby(["grp", "sealed"])[["out_temp", "out_rad", "in_temp", "act_heating", "act_co2", "act_shade", "act_fog", "fan0", "ec", "day"]]
          .mean().round(2).to_string())
    # sealed vs weather within F13/F47: logistic-like AUC of weather alone
    from sklearn.metrics import roc_auc_score
    tr = tgD.dropna(subset=["out_temp", "out_rad"])
    for c in ["out_temp", "out_rad", "day", "act_heating", "act_shade", "act_co2"]:
        s = tr[[c, "sealed"]].dropna()
        print("  AUC sealed ~ %-12s %.3f" % (c, roc_auc_score(s.sealed, s[c])))
    # fan: is circfan==0 all day (a source that simply never runs fans)?
    print("  sealed days with circfan exactly 0 all day: %.2f" % (tgD[tgD.sealed].fan0 == 1).mean())
    print("  circfan daily-mean histogram (F13/F47):", np.histogram(tgD.act_circfan.dropna(), [0, 1, 5, 10, 20, 40, 60, 101])[0])
    # sealed at matched weather: pairs of days that share the calendar date (same out_temp 24h vector)
    key = a[a.farm.isin(["F13", "F47"])].groupby(["farm", "day"]).out_temp.apply(lambda s: tuple(np.round(s.values, 1)))
    K = key.reset_index().rename(columns={"out_temp": "k"}).merge(tgD[["farm", "day", "sealed", "ec", "is_test", "act_circfan"]], on=["farm", "day"])
    grp = K.groupby("k")
    same, mixed, nn = 0, 0, 0
    ecdiff = []
    for k, g in grp:
        if len(g) < 2 or g.sealed.isna().any():
            continue
        nn += 1
        if g.sealed.nunique() == 1:
            same += 1
        else:
            mixed += 1
            s1, s0 = g[g.sealed].ec.mean(), g[~g.sealed].ec.mean()
            if np.isfinite(s1) and np.isfinite(s0):
                ecdiff.append(s1 - s0)
    print("  same-weather groups: %d, sealed status identical %d, mixed %d; EC(sealed)-EC(not) within mixed groups: mean %.3f median %.3f n=%d"
          % (nn, same, mixed, np.mean(ecdiff), np.median(ecdiff), len(ecdiff)))


if __name__ == "__main__":
    main()
