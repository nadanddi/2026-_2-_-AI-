# -*- coding: utf-8 -*-
"""Inspector 4-3, step 5:
  A. the midnight carry-over correction (step 4) on the calibrated temperature
     validators EXT8/10/12 (cold days held out at once).  Coefficients fitted
     on DIAG10 OOF residuals of days OUTSIDE the EXT hold-out (+-1 day buffer).
  B. EC: does the same carry-over show up in the EC OOF (oof_ec_diag)?  And a
     screen of causal day-level candidates (running mean up to the current
     hour, earlier-day inputs of the same greenhouse) against the EC OOF
     residual, with a permutation null for the maximum |rho|.
Run:  cd research && PYTHONPATH="" <python> -u audit4_3_05_ext_and_ec.py
"""
import env  # noqa: F401
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import common
from common import split_mask, rmse, TARGET_FARMS
from harness import load
from screen_v6 import boot
from audit4_3_04_midnight_fix import gaps, design, TAUS


def main():
    _, lab0, labe0 = load()
    z = np.load(env.LOCAL + "/eval_v6_oof.npz", allow_pickle=True)
    rid = z["row_id"]
    lab = lab0.set_index("row_id").loc[rid].reset_index()
    lab = lab.merge(gaps(), on=["farm", "day"], how="left")
    lab[["g1", "g2"]] = lab[["g1", "g2"]].fillna(0.0)
    y = lab.sub_temp.values
    rd = y - z["F60ND__DIAG10"]
    print("== A. EXT validators ==")
    for key in ("F60ND__EXT8", "F60ND__EXT10", "F60ND__EXT12"):
        base = z[key]
        ho = ~np.isnan(base)
        days = lab[ho].groupby("farm").day.apply(lambda s: set(int(d) for d in s.unique())).to_dict()
        fd = {f: days.get(f, set()) for f in TARGET_FARMS}
        trm, vam = split_mask(lab, fd)
        tr = lab[trm].assign(r=rd[trm])
        best = None
        for tau in TAUS:
            X = design(tr, tau, True)
            c, *_ = np.linalg.lstsq(X, tr.r.values, rcond=None)
            s = float(((tr.r.values - X @ c) ** 2).sum())
            if best is None or s < best[0]:
                best = (s, tau, c)
        _, tau, c = best
        p = base.copy()
        p[vam] = base[vam] + design(lab[vam], tau, True) @ c
        g = vam & ho
        pt, lo, hi, pw = boot(lab[g].reset_index(drop=True), "sub_temp", base[g], p[g])
        print("  %s n=%d tau %.1f c %s | %.4f -> %.4f (%+.2f%%) CI [%+.4f,%+.4f] P(worse) %.3f"
              % (key, g.sum(), tau, np.round(c, 3), rmse(base[g], y[g]), rmse(p[g], y[g]),
                 100 * (rmse(p[g], y[g]) / rmse(base[g], y[g]) - 1), lo, hi, pw))

    print("\n== B. EC OOF residual ==")
    ze = np.load(env.LOCAL + "/oof_ec_diag.npz", allow_pickle=True)
    L = labe0.set_index("row_id").loc[ze["row_id"]].reset_index()
    L["r"] = L.sub_ec.values - ze["oof"]
    L = L.merge(gaps(), on=["farm", "day"], how="left")
    for hh in (0, 2, 5, 12):
        s = L[L.hour == hh].dropna(subset=["g1", "g2"])
        print("  h%02d rho(r, g1) %+.3f rho(r, g2) %+.3f" % (hh, spearmanr(s.r, s.g1).correlation, spearmanr(s.r, s.g2).correlation))

    # causal candidates, evaluated as ROW features against row residual, and day means at 23 h
    tX, _, sX = common.load_raw()
    b = pd.concat([tX, sX], ignore_index=True)
    b = b[b.farm.isin(["F13", "F47"])].sort_values(["farm", "t"]).reset_index(drop=True)
    es = 610.78 * np.exp(17.27 * b.in_temp / (b.in_temp + 237.3)) / 1000
    b["vpd"] = es * (1 - b.in_hum / 100)
    b["rad"] = b.out_rad.clip(lower=0) - 5
    gd = b.groupby(["farm", "day"])
    b["cum_rad"] = gd.rad.cumsum()
    b["cum_vpd"] = gd.vpd.cumsum()
    b["run_tin"] = gd.in_temp.transform(lambda s: s.expanding().mean())
    b["run_heat"] = gd.act_heating.transform(lambda s: s.fillna(0).expanding().mean())
    b["run_fan"] = gd.act_circfan.transform(lambda s: s.fillna(0).expanding().mean())
    b["run_co2dose"] = gd.act_co2.transform(lambda s: s.fillna(0).expanding().mean())
    b["run_ventzero"] = gd.act_vent.transform(lambda s: (s.fillna(0) == 0).expanding().mean())
    dsum = b.groupby(["farm", "day"]).agg(rad_d=("rad", "sum"), vpd_d=("vpd", "sum"), tin_d=("in_temp", "mean"),
                                          heat_d=("act_heating", "mean"), dose_d=("act_co2", "mean")).reset_index()
    for lag in (1, 2, 4):
        x = dsum.copy(); x["day"] += lag
        b = b.merge(x.rename(columns={c: c + "_l%d" % lag for c in ["rad_d", "vpd_d", "tin_d", "heat_d", "dose_d"]}),
                    on=["farm", "day"], how="left")
    b["rad_2src"] = b.rad_d_l2 + b.rad_d_l4          # same-source proxy, two earlier days
    b["vpd_2src"] = b.vpd_d_l2 + b.vpd_d_l4
    b["dtin_l2"] = b.run_tin - b.tin_d_l2
    cands = [c for c in b.columns if c.startswith(("cum_", "run_")) or "_l" in c or c.endswith("2src")]
    M = L[["row_id", "farm", "day", "hour", "r"]].merge(b[["row_id"] + cands], on="row_id", how="left")
    late = M[M.hour == 23]
    rows = []
    for c in cands:
        s = late[["r", c]].dropna()
        rows.append((c, spearmanr(s.r, s[c]).correlation, len(s)))
    R = pd.DataFrame(rows, columns=["feat", "rho_day_resid", "n"]).sort_values("rho_day_resid", key=abs, ascending=False)
    rng = np.random.RandomState(0)
    mx = []
    for _ in range(200):
        perm = late.r.sample(frac=1, random_state=rng).values
        mx.append(max(abs(spearmanr(perm, late[c].values, nan_policy="omit").correlation) for c in cands))
    print("  day residual (23 h row) vs causal candidates; permutation 95%% of max|rho| = %.3f" % np.percentile(mx, 95))
    print(R.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
