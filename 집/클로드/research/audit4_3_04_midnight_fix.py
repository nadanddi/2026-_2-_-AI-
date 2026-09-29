# -*- coding: utf-8 -*-
"""Inspector 4-3, step 4: quick validation of a midnight carry-over correction
for sub_temp on the cached DIAG10 out-of-fold predictions (F60ND__DIAG10).

Finding (step 3): residual y - oof at 00-05 h correlates -0.32..-0.36 with
g1 = mean in_temp(d-1, 20-23 h) - in_temp(d, 00 h), i.e. the model carries
the PREVIOUS RECORD DAY's evening (usually another source greenhouse) into
the new day.  g2 uses d-2 (often the same source).  Both are inputs of the
same greenhouse at earlier hours -> causal.

Correction  r_hat = (c1*g1 + c2*g2) * exp(-hour/tau), tau chosen inside the
training folds; coefficients fitted on training-fold OOF residuals only
(classic stacking; the held-out fold's rows never enter the fit).
Scored per DIAG10 fold; paired greenhouse-day block bootstrap (screen_v6.boot).
Run:  cd research && PYTHONPATH="" <python> -u audit4_3_04_midnight_fix.py
"""
import env  # noqa: F401
import numpy as np
import pandas as pd

import common
from common import split_mask, rmse
from harness import load
from anal_q1_errors import diag_folds
from screen_v6 import boot
import train_flags_v6 as TF

TAUS = (1.5, 2.5, 4.0, 6.0, 9.0)


def gaps():
    tX, _, sX = common.load_raw()
    b = pd.concat([tX, sX], ignore_index=True)
    b = b[b.farm.isin(["F13", "F47"])]
    h0 = b[b.hour == 0][["farm", "day", "in_temp"]].rename(columns={"in_temp": "a0"})
    e = b[b.hour.between(20, 23)].groupby(["farm", "day"]).in_temp.mean().rename("e").reset_index()
    for lag in (1, 2):
        x = e.copy(); x["day"] += lag
        h0 = h0.merge(x.rename(columns={"e": "e%d" % lag}), on=["farm", "day"], how="left")
    h0["g1"] = (h0.e1 - h0.a0).fillna(0.0)
    h0["g2"] = (h0.e2 - h0.a0).fillna(0.0)
    return h0[["farm", "day", "g1", "g2"]]


def design(df, tau, use2=True):
    k = np.exp(-df.hour.values / tau)
    cols = [df.g1.values * k] + ([df.g2.values * k] if use2 else [])
    return np.column_stack(cols)


def fit_predict(tr, va, use2):
    best = None
    # tau by inner leave-fold-out would be costly; choose tau on training rows' fit (few params)
    for tau in TAUS:
        X = design(tr, tau, use2)
        c, *_ = np.linalg.lstsq(X, tr.r.values, rcond=None)
        sse = float(((tr.r.values - X @ c) ** 2).sum())
        if best is None or sse < best[0]:
            best = (sse, tau, c)
    _, tau, c = best
    return design(va, tau, use2) @ c, tau, c


def main():
    _, lab0, _ = load()
    z = np.load(env.LOCAL + "/eval_v6_oof.npz", allow_pickle=True)
    oof = pd.Series(z["F60ND__DIAG10"], index=z["row_id"])
    lab = lab0[lab0.row_id.isin(oof.index)].reset_index(drop=True)
    lab = lab.merge(gaps(), on=["farm", "day"], how="left")
    lab[["g1", "g2"]] = lab[["g1", "g2"]].fillna(0.0)
    lab["p"] = oof.reindex(lab.row_id).values
    lab["r"] = lab.sub_temp - lab.p
    y = lab.sub_temp.values
    clean = TF.row_weights(lab, 0.0, radius=3) >= 1
    fds = diag_folds(lab)
    out = {}
    for nm, use2 in (("g1", False), ("g1+g2", True)):
        adj = np.full(len(lab), np.nan)
        info = []
        for fd in fds:
            trm, vam = split_mask(lab, fd)
            pr, tau, c = fit_predict(lab[trm], lab[vam], use2)
            adj[np.where(vam)[0]] = pr
            info.append((tau, np.round(c, 3)))
        out[nm] = lab.p.values + adj
        print("%s: fold taus/coefs %s" % (nm, info[:3]))
    base = lab.p.values
    for nm, p in out.items():
        per = []
        for fd in fds:
            _, vam = split_mask(lab, fd)
            per.append(rmse(p[vam], y[vam]) - rmse(base[vam], y[vam]))
        print("\n== %s ==" % nm)
        print("  all   base %.4f -> %.4f (%+.2f%%) | folds improved %d/%d"
              % (rmse(base, y), rmse(p, y), 100 * (rmse(p, y) / rmse(base, y) - 1), sum(d < 0 for d in per), len(per)))
        print("  clean base %.4f -> %.4f (%+.2f%%)" % (rmse(base[clean], y[clean]), rmse(p[clean], y[clean]),
                                                     100 * (rmse(p[clean], y[clean]) / rmse(base[clean], y[clean]) - 1)))
        for lo, hi in ((0, 5), (6, 11), (12, 23)):
            m = lab.hour.between(lo, hi).values
            print("  h%02d-%02d %.4f -> %.4f" % (lo, hi, rmse(base[m], y[m]), rmse(p[m], y[m])))
        for s, m in (("pass1", lab.day.values <= 178), ("pass2", lab.day.values >= 179)):
            print("  %s %.4f -> %.4f" % (s, rmse(base[m], y[m]), rmse(p[m], y[m])))
        pt, lo, hi, pw = boot(lab, "sub_temp", base, p)
        print("  block bootstrap delta %+.4f CI [%+.4f, %+.4f] P(worse) %.3f" % (pt, lo, hi, pw))
        m = clean
        pt, lo, hi, pw = boot(lab[m].reset_index(drop=True), "sub_temp", base[m], p[m])
        print("  clean-rows bootstrap delta %+.4f CI [%+.4f, %+.4f] P(worse) %.3f" % (pt, lo, hi, pw))
    np.savez(env.LOCAL + "/audit4_3_04_oof.npz", row_id=lab.row_id.values, base=base, **{k.replace("+", "_"): v for k, v in out.items()})


if __name__ == "__main__":
    main()
