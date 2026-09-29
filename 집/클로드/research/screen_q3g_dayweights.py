# -*- coding: utf-8 -*-
"""Step 3: down-weight the noise-injected training days, judged on the
calibrated validators.

anal_q3f_days.py (out of fold): the quarter of training days whose input
shape is least like the clean test inputs has double the day-level offset
(0.585 vs ~0.28, diff CI [0.19, 0.43]); their signature is CO2 changes 2.5-3.5x
larger than on test days and a rougher indoor temperature -- noise added to
the inputs of whole days.  Test inputs carry none.

Temperature (calibrated validator EXTRAP10, do-no-harm geometry A/B):
  plain   round-3 configuration
  w02     167 rule-flagged rows at weight 0.2              (known: EXTRAP -3.0%)
  d05     w02 + Q4 noisy days at weight 0.5
  d02     w02 + Q4 noisy days at weight 0.2
EC (calibrated validator geometry A, B secondary): plain / d05 / d02
(MLP member gets weights only if the installed sklearn supports it).

Each validator is scored three ways:
  all        every held-out row
  clean      rule-flagged rows +-3 h removed
  testlike   clean rows on Q1-Q3 days only (conditions closest to the test)

Run:  cd research && PYTHONPATH="" <python> -u screen_q3g_dayweights.py
"""
import inspect

import env  # noqa: F401
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor

from common import split_mask, rmse, USABLE, OUT_COLS, TARGET_FARMS
from harness import load, views, folds
import feat_temp74 as T74
import feat_new
import features_v4 as F4
from cold_v5 import THRESH
from screen_v6 import temp_members, collect, boot
from cleanw_v6 import weights
from ec_v6 import et, ltw, mlp
from make_submission_v3 import causal_shrink

MLP_SW = "sample_weight" in inspect.signature(MLPRegressor.fit).parameters


def day_weights(lab, w_q4):
    ds = pd.read_csv(env.LOCAL + "/anal_q3f_dayscore.csv")
    q4 = set(map(tuple, ds[ds.q == "Q4 train-like"][["farm", "day"]].values))
    m = np.array([(f, d) in q4 for f, d in zip(lab.farm.values, lab.day.values)])
    return np.where(m, w_q4, 1.0), m


def masks(lab):
    clean = weights(lab, 3, 0.0) >= 1
    _, q4 = day_weights(lab, 1.0)
    return {"all": np.ones(len(lab), bool), "clean": clean, "testlike": clean & ~q4}


def table(title, sets, variants, oof, y, M):
    print("\n== %s ==" % title)
    head = "%-6s" % ""
    for s, _ in sets:
        head += " | %-7s %-7s %-8s" % (s + ":all", "clean", "testlike")
    print(head)
    for v in variants:
        line = "%-6s" % v
        for s, _ in sets:
            o = oof[(v, s)]
            g = ~np.isnan(o)
            line += " | " + " ".join("%7.4f" % rmse(o[g & M[k]], y[g & M[k]]) for k in ("all", "clean", "testlike"))
        print(line)


def main():
    panel, lab_t0, lab_e0 = load()
    v = views(panel)
    ex, blocks = feat_new.build_extra()
    sg, ph, fp = F4.seg_features(), F4.phys_features(), F4.fp_features()
    lab_t = (lab_t0.merge(ex, on="row_id", how="left").merge(sg, on="row_id", how="left")
                   .merge(ph, on="row_id", how="left").merge(fp, on="row_id", how="left")).copy()
    lab_e = lab_e0.merge(fp, on="row_id", how="left").copy()
    f93 = T74.base74(v["temp"]) + list(blocks["dew"]) + list(blocks["event"])
    ct = f93 + F4.names(sg) + F4.names(fp)
    phc, fpc = F4.names(ph), F4.names(fp)
    f14 = [c for c in (list(USABLE) + ["day", "hr_sin", "hr_cos", "midnight"]) if c not in OUT_COLS]

    # ---------------- temperature ------------------------------------------
    Wf = weights(lab_t, 0, 0.2)
    Wt = {"plain": None, "w02": Wf,
          "d05": Wf * day_weights(lab_t, 0.5)[0], "d02": Wf * day_weights(lab_t, 0.2)[0]}
    dmin = lab_t.groupby(["farm", "day"]).ph_in_temp_3.min()
    cd = dmin[dmin < THRESH]
    ext = [{f: set(int(d) for (ff, d) in cd.index if ff == f) for f in TARGET_FARMS}]
    tsets = [("EXT10", ext), ("gA", folds("A")), ("gB", folds("B"))]
    toof = {}
    for vn, W in Wt.items():
        for s, fds in tsets:
            Mm = collect(lab_t, fds, lambda tr, va, m: temp_members(tr, va, ct, phc, None if W is None else W[m]))
            toof[(vn, s)] = 0.65 * Mm["res"] + 0.25 * Mm["ridge"] + 0.10 * Mm["nys"]
        print("  temp %s done" % vn, flush=True)
    yt = lab_t.sub_temp.values
    Mt = masks(lab_t)
    table("sub_temp  (primary: EXT10; gA/gB do-no-harm)", tsets, list(Wt), toof, yt, Mt)
    for vn in ("w02", "d05", "d02"):
        for s in ("EXT10", "gA", "gB"):
            a, b = toof[("plain", s)], toof[(vn, s)]
            keep = Mt["testlike"]
            pr, lo, hi, pw = boot(lab_t[keep].reset_index(drop=True), "sub_temp", a[keep], b[keep])
            print("  paired %-4s vs plain on %-5s testlike: %+.4f CI [%+.4f,%+.4f] P(worse)=%.3f"
                  % (vn, s, pr, lo, hi, pw))

    # ---------------- EC -------------------------------------------------------
    def ec_fn(W):
        def fn(tr, va, m):
            y = tr.sub_ec.values
            w = None if W is None else W[m]
            kw_et = {} if w is None else {"extratreesregressor__sample_weight": w}
            kw_l = {} if w is None else {"sample_weight": w}
            kw_m = {} if (w is None or not MLP_SW) else {"mlpregressor__sample_weight": w}
            p = (0.60 * et().fit(tr[f14 + fpc], y, **kw_et).predict(va[f14 + fpc])
                 + 0.30 * ltw().fit(tr[f14], y, **kw_l).predict(va[f14])
                 + 0.10 * mlp().fit(tr[f14], y, **kw_m).predict(va[f14]))
            return {"p": np.clip(causal_shrink(p, va, 0.5), 0.062, 3.46)}
        return fn

    We = {"plain": None, "d05": day_weights(lab_e, 0.5)[0], "d02": day_weights(lab_e, 0.2)[0]}
    esets = [("A", folds("A")), ("B", folds("B"))]
    eoof = {}
    for vn, W in We.items():
        for s, fds in esets:
            eoof[(vn, s)] = collect(lab_e, fds, ec_fn(W))["p"]
        print("  ec %s done (MLP weighted: %s)" % (vn, MLP_SW), flush=True)
    ye = lab_e.sub_ec.values
    Me = masks(lab_e)
    table("sub_ec  (primary: A; B secondary)", esets, list(We), eoof, ye, Me)
    for vn in ("d05", "d02"):
        for s, fds in esets:
            a, b = eoof[("plain", s)], eoof[(vn, s)]
            idx = [np.where(split_mask(lab_e, fd)[1])[0] for fd in fds]
            d = [rmse(b[i], ye[i]) - rmse(a[i], ye[i]) for i in idx]
            pr, lo, hi, pw = boot(lab_e, "sub_ec", a, b)
            print("  paired %s vs plain on %s all: %+.4f | folds %d/5 | CI [%+.4f,%+.4f] P(worse)=%.3f"
                  % (vn, s, pr, sum(x < 0 for x in d), lo, hi, pw))


if __name__ == "__main__":
    main()
