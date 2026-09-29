# -*- coding: utf-8 -*-
"""Why are the other 49 greenhouses in the training data?  Analysis only
(test_X inputs are LOOKED AT, never fed to a model here).

Part A  profile per greenhouse: rows, labelled rows, missing inputs,
        substrate - air temperature, EC level, integer-rounded labels,
        share of cold hours, in_temp range.
Part B  near-duplicate search.  Earlier checks (catalog 1.11) looked for EXACT
        copies only; training inputs carry injected noise, so a copied day
        would not match exactly.  For every F13/F47 day (training and test)
        take the 24-h indoor vectors (in_temp, in_hum) and find the closest
        day in any OTHER greenhouse (RMSE of in_temp, and of the centred
        shape).  Compare with a null: the closest-day distance between two
        greenhouses that cannot share days (random pairs of other farms).
        If test days have partners far closer than the null, the other
        greenhouses hold copies (with labels) of our days.

Run:  cd research && PYTHONPATH="" <python> -u anal_f1_otherfarms.py
"""
import numpy as np
import pandas as pd

import env  # noqa: F401
import common

pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 80)
INPUTS = ["out_temp", "out_hum", "out_rad", "out_wspd", "in_temp", "in_hum", "in_co2", "in_rad",
          "act_vent", "act_side", "act_shade", "act_thermal", "act_valve", "act_heating", "act_circfan",
          "act_co2", "act_fog", "act_cool", "act_pump"]


def profile(tX, ty):
    a = tX.merge(ty[["row_id", "sub_temp", "sub_ec"]], on="row_id", how="left")
    rows = []
    for f, g in a.groupby("farm"):
        lt = g.sub_temp.dropna()
        rows.append(dict(
            farm=f, days=g.day.nunique(), rows=len(g), lab_t=int(g.sub_temp.notna().sum()),
            lab_ec=int(g.sub_ec.notna().sum()),
            miss_inputs=int(sum(g[c].isna().mean() > 0.99 for c in INPUTS)),
            sub_minus_air=float((g.sub_temp - g.in_temp).mean()),
            ec_mean=float(g.sub_ec.mean()), ec_sd=float(g.sub_ec.std()),
            int_label=float((np.abs(lt - lt.round()) < 1e-9).mean()) if len(lt) else np.nan,
            cold10=float((g.in_temp < 10).mean()), in_t_p5=float(g.in_temp.quantile(.05)),
            in_t_p50=float(g.in_temp.median())))
    return pd.DataFrame(rows).set_index("farm")


def day_vectors(df, col):
    p = df.pivot_table(index=["farm", "day"], columns="hour", values=col, aggfunc="first")
    p = p.reindex(columns=range(24))
    return p[p.notna().all(1)]


def nearest(A, B, centred=False):
    a, b = A.values.astype(float), B.values.astype(float)
    if centred:
        a = a - a.mean(1, keepdims=True)
        b = b - b.mean(1, keepdims=True)
    d2 = (a ** 2).sum(1)[:, None] + (b ** 2).sum(1)[None, :] - 2 * a @ b.T
    d = np.sqrt(np.maximum(d2, 0) / a.shape[1])
    j = d.argmin(1)
    return d[np.arange(len(a)), j], B.index[j]


def main():
    tX, ty, sX = common.load_raw()
    print("== Part A: greenhouse profiles ==")
    P = profile(tX, ty)
    print(P.round(3).to_string())

    allx = pd.concat([tX.assign(split="train"), sX.assign(split="test")], ignore_index=True)
    test_days = set(map(tuple, sX[["farm", "day"]].drop_duplicates().values))
    for col in ("in_temp", "in_hum"):
        V = day_vectors(allx, col)
        own = V[V.index.get_level_values(0).isin(["F13", "F47"])]
        oth = V[~V.index.get_level_values(0).isin(["F13", "F47"])]
        farms = sorted(set(oth.index.get_level_values(0)))
        rng = np.random.default_rng(0)
        # null: for each of 40 random other farms, nearest day in a DIFFERENT random other farm
        null_raw, null_c = [], []
        for f in rng.choice(farms, size=min(40, len(farms)), replace=False):
            q = oth.loc[[f]]
            ref = oth[oth.index.get_level_values(0) != f]
            null_raw += list(nearest(q, ref)[0])
            null_c += list(nearest(q, ref, True)[0])
        dr, jr = nearest(own, oth)
        dc, jc = nearest(own, oth, True)
        R = pd.DataFrame({"d_raw": dr, "partner_raw": [f"{a}_{b}" for a, b in jr],
                          "d_shape": dc, "partner_shape": [f"{a}_{b}" for a, b in jc]}, index=own.index)
        R["test"] = [k in test_days for k in R.index]
        nq = np.quantile(null_raw, [0.001, 0.01, 0.05, 0.5])
        nc = np.quantile(null_c, [0.001, 0.01, 0.05, 0.5])
        print("\n== Part B: %s, nearest day in another greenhouse ==" % col)
        print("null (other farm vs other farm) raw RMSE quantiles 0.1/1/5/50%%: %s" % np.round(nq, 3))
        print("null shape RMSE quantiles 0.1/1/5/50%%: %s" % np.round(nc, 3))
        for lab, m in (("F13/F47 training days", ~R.test), ("F13/F47 TEST days", R.test)):
            s = R[m]
            print("%s (%d): raw median %.3f, below null 0.1%%: %d | shape median %.3f, below null 0.1%%: %d"
                  % (lab, len(s), s.d_raw.median(), int((s.d_raw < nq[0]).sum()),
                     s.d_shape.median(), int((s.d_shape < nc[0]).sum())))
        print("closest 15 (raw):")
        print(R.sort_values("d_raw").head(15).round(3).to_string())
        pc = R.partner_raw.str[:3].value_counts().head(10)
        print("partner farms (raw nearest, all F13/F47 days):", dict(pc))


if __name__ == "__main__":
    main()
