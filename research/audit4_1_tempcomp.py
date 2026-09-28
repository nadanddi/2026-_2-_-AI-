# -*- coding: utf-8 -*-
"""Inspector 4-1: was rejecting temperature-compensated EC (6.28) premature?
Uses cached DIAG10 OOFs only (ec_tempcomp_v1_diag.npz, eval_v6_oof.npz).  stdout only.
All a-grids here are POST-HOC sensitivity, not selection."""
import env  # noqa: F401
import numpy as np
import pandas as pd
from harness import load
from common import rmse
from screen_v6 import boot

_, _, lab = load()
z = np.load(env.LOCAL + "/ec_tempcomp_v1_diag.npz", allow_pickle=True)
assert (z["row_id"] == lab.row_id.values).all()
t = np.load(env.LOCAL + "/eval_v6_oof.npz", allow_pickle=True)
assert (t["row_id"] == lab.row_id.values).all()
base, comp, orac = z["base"], z["comp"], z["oracle"]
That = t["F60ND__DIAG10"]
y = lab.sub_ec.values; T = lab.sub_temp.values
L = lab[["farm", "day", "hour", "t"]].copy().reset_index(drop=True)
g = ~np.isnan(base) & ~np.isnan(comp) & ~np.isnan(That)
print("rows %d" % g.sum())
key = [L.farm, L.day]


def dmean(x):
    return pd.Series(x).groupby(key).transform("mean").values


def expmean(x):
    return pd.Series(x).groupby(key).transform(lambda s: s.expanding().mean()).values


def rep(nm, p, ref=base):
    pr, lo, hi, pw = boot(lab[g].reset_index(drop=True), "sub_ec", ref[g], p[g])
    print("%-44s %.4f  d %+.4f [%+.4f,%+.4f] P(worse) %.2f" % (nm, rmse(p[g], y[g]), pr, lo, hi, pw))


rep("base (round 3)", base)
rep("comp (refit on ec25, a=0.02, T_hat)", comp)
rep("oracle comp (true T)", orac)
# decomposition of comp - base into day-level and within-day parts (non-causal diagnostic only)
hyb1 = dmean(base) + (comp - dmean(comp))   # base level, comp shape
hyb2 = dmean(comp) + (base - dmean(base))   # comp level, base shape
rep("[diag, non-causal] base level + comp shape", hyb1)
rep("[diag, non-causal] comp level + base shape", hyb2)
# shape-only multiplicative corrections on base
for a in (0.01, 0.02, 0.03):
    rep("[non-causal] base*(1+a(T -daymean T)) true T a=%.2f" % a, base * (1 + a * (T - dmean(T))))
    rep("[non-causal] base*(1+a(That-daymean)) a=%.2f" % a, base * (1 + a * (That - dmean(That))))
    rep("[causal] base*(1+a(That-expanding mean)) a=%.2f" % a, base * (1 + a * (That - expmean(That))))
# the model already carries half the slope; residual correction b = 0.02 - 0.0094 ~ 0.011
rep("[causal] a=0.011 (label slope - model slope)", base * (1 + 0.011 * (That - expmean(That))))
# day offset share
e = base - y
dm_ = dmean(e)
print("\nbase SSE share day-offset %.1f%%" % (100 * np.nansum(dm_[g] ** 2) / np.nansum(e[g] ** 2)))
# cold / warm days, bias
cold = pd.Series(That).groupby(key).transform("min").values < 10
for nm, m in (("That-daymin<10", cold & g), ("warm", ~cold & g), ("pass2", (L.day.values >= 179) & g)):
    print("%-16s n=%4d base %.4f bias %+.3f | comp %.4f bias %+.3f | oracle %.4f" % (nm, m.sum(), rmse(base[m], y[m]),
          (base - y)[m].mean(), rmse(comp[m], y[m]), (comp - y)[m].mean(), rmse(orac[m], y[m])))
# temperature error contribution: comp vs oracle differ only by T_hat vs T
print("T_hat rmse on these rows %.3f ; comp-oracle rmse gap %.4f" % (rmse(That[g], T[g]), rmse(comp[g], y[g]) - rmse(orac[g], y[g])))
