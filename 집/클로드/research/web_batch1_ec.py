# -*- coding: utf-8 -*-
"""Web-research batch 1, EC (research 1: control set-point inference from the
Korean top-farm study; transpiration accumulation from the strawberry ET papers).
All features are WITHIN-DAY, expanding up to the current hour (same record day
only), so no cross-day or cross-test path exists.

  SP  set-point fingerprints of the source's control rules, so far today:
        sp_heat_T   min in_temp over hours with heating on
        sp_vent_T   in_temp at the first hour the roof vent opened
        sp_co2_h    hour CO2 dosing first switched on
        sp_curt_h   hour the thermal curtain first closed after 12 h
        sp_night_T  mean in_temp over hours 0..min(h,6)
  TR  water-loss proxies, cumulative so far today:
        tr_rad      sum rad_eff, tr_vpd sum vpd_in, tr_ratio = tr_vpd / (tr_rad + 1)
Added to the ExtraTrees member (the one that already carries the fingerprint).

Pre-set rule: adopt only if the variant beats round-3 EC on geometry A (mean of
per-fold RMSE) AND DIAG10 (pooled) for both seeds (7, 8), and the DIAG10
block-bootstrap CI excludes 0.

Run:  cd research && PYTHONPATH="" <python> -u web_batch1_ec.py
"""
import env  # noqa: F401
import numpy as np
import pandas as pd

import ec_v6
from common import split_mask, rmse, USABLE, OUT_COLS
from harness import load, folds
import features_v4 as F4
from anal_q1_errors import diag_folds
from make_submission_v3 import causal_shrink
from screen_v6 import boot

SP = ["sp_heat_T", "sp_vent_T", "sp_co2_h", "sp_curt_h", "sp_night_T"]
TR = ["tr_rad", "tr_vpd", "tr_ratio"]


def within_day(df):
    df = df.sort_values(["farm", "t"]).copy()
    key = [df.farm, df.day]
    heat_on = df.act_heating.fillna(0) > 0
    df["sp_heat_T"] = df.in_temp.where(heat_on).groupby(key).cummin()
    df["sp_heat_T"] = df.groupby(["farm", "day"]).sp_heat_T.ffill()
    first = lambda m, v: v.where(m & (m.groupby(key).cumsum() == 1)).groupby(key).ffill()
    df["sp_vent_T"] = first(df.act_vent.fillna(0) > 0, df.in_temp)
    df["sp_co2_h"] = first(df.act_co2.fillna(0) > 0, df.hour.astype(float))
    curt = (df.hour >= 12) & (df.act_thermal.fillna(100) < 50)
    df["sp_curt_h"] = first(curt, df.hour.astype(float))
    df["sp_night_T"] = df.in_temp.where(df.hour <= 6).groupby(key).transform(lambda s: s.expanding().mean())
    df["sp_night_T"] = df.groupby(["farm", "day"]).sp_night_T.ffill()
    df["tr_rad"] = df.rad_eff.fillna(0).groupby(key).cumsum()
    df["tr_vpd"] = df.vpd_in.fillna(0).groupby(key).cumsum()
    df["tr_ratio"] = df.tr_vpd / (df.tr_rad + 1.0)
    return df


def run(tr, va, c_et, c_rest, seed):
    ec_v6.SEED = seed
    y = tr.sub_ec.values
    et = ec_v6.et().fit(tr[c_et], y).predict(va[c_et])
    ltw = ec_v6.ltw().fit(tr[c_rest], y).predict(va[c_rest])
    mlp = ec_v6.mlp().fit(tr[c_rest], y).predict(va[c_rest])
    return np.clip(causal_shrink(0.6 * et + 0.3 * ltw + 0.1 * mlp, va, 0.5), 0.062, 3.46)


def main():
    panel, _, lab0 = load()
    fp = F4.fp_features()
    lab = within_day(lab0.merge(fp, on="row_id", how="left")).reset_index(drop=True)
    f14 = [c for c in (list(USABLE) + ["day", "hr_sin", "hr_cos", "midnight"]) if c not in OUT_COLS]
    base_et = f14 + F4.names(fp)
    V = {"base": base_et, "SP": base_et + SP, "TR": base_et + TR}
    y = lab.sub_ec.values
    print("coverage:", {c: round(float(lab[c].notna().mean()), 2) for c in SP + TR}, flush=True)
    verdict = {"SP": True, "TR": True}
    for vset, fds in (("A", folds("A")), ("DIAG10", diag_folds(lab))):
        for sd in (7, 8):
            pooled = {k: np.full(len(lab), np.nan) for k in V}
            per = {k: [] for k in V}
            for fd in fds:
                trm, vam = split_mask(lab, fd)
                tr, va = lab[trm], lab[vam].reset_index(drop=True)
                idx = np.where(vam)[0]
                for k, cols in V.items():
                    p = run(tr, va, cols, f14, sd)
                    pooled[k][idx] = p
                    per[k].append(rmse(p, y[idx]))
            line = "%-6s seed %d" % (vset, sd)
            g = ~np.isnan(pooled["base"])
            for k in ("SP", "TR"):
                if vset == "A":
                    b, c = np.array(per["base"]), np.array(per[k])
                    d = c.mean() / b.mean() - 1
                    line += " | %s %.4f vs %.4f (%+.2f%%, better %d/%d)" % (k, c.mean(), b.mean(), 100 * d, int((c < b).sum()), len(b))
                    ok = d < 0
                else:
                    pr, lo, hi, pw = boot(lab[g].reset_index(drop=True), "sub_ec", pooled["base"][g], pooled[k][g])
                    d = rmse(pooled[k][g], y[g]) / rmse(pooled["base"][g], y[g]) - 1
                    line += " | %s %.4f vs %.4f (%+.2f%%) [%+.4f, %+.4f]" % (k, rmse(pooled[k][g], y[g]), rmse(pooled["base"][g], y[g]), 100 * d, lo, hi)
                    ok = d < 0 and hi < 0
                verdict[k] = verdict[k] and ok
            print(line, flush=True)
    print("\nPRE-SET RULE VERDICT:", {k: ("ADOPT" if v else "REJECT") for k, v in verdict.items()})


if __name__ == "__main__":
    main()
