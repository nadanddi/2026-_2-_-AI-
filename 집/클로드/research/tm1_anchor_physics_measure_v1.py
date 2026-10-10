# -*- coding: utf-8 -*-
"""TM1 (2026-10-12 집 클로드; user: "연구자의 입장에서 다양하게, 모든 식에 이유"): MEASURE the pieces of the temperature
anchor instead of choosing them by convention.  Descriptive; uses linker v7 mirror-CV choices (local/linker_v7/
cv_all400_choice_v7m.csv) for labelled records, the round-11 recipe q (W40G-S + chain corrector) on the clean layout DIAG10.
Measured on PASS-1 records only (pass-2 reported separately as a check, never used to choose):
 (1) midnight transfer: y_0 - A  vs  dT_air (record 0 h air - previous 23 h air): intercept c0 and slope k (substrate follows a
     fraction of the air step across midnight).  Physical expectation: 0 < k < 1, c0 > 0 (substrate keeps cooling after 23 h).
 (2) hourly information decay: beta_h = slope of (y_h - q_h) on (yhat0 - q_0) per hour h, yhat0 = A - c0 + k dT_air.
 (3) linearity: binned mean of (y_0 - q_0) against (yhat0 - q_0).
 (4) confidence: |yhat0 - y_0| and anchor-better share by linker p bins.
Run:  cd 집/클로드/research && PYTHONPATH="" py -3.12 -u tm1_anchor_physics_measure_v1.py
"""
import os, sys, importlib.util
import env  # noqa: F401
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.argv = ["x"]
s = importlib.util.spec_from_file_location("ta2", os.path.join(HERE, "ta2_anchor_debiased_v1.py")); A2 = importlib.util.module_from_spec(s); s.loader.exec_module(A2)
LK = os.path.join(env.LOCAL, "linker_v7", "cv_all400_choice_v7m.csv")


def main():
    L = pd.read_csv(LK); L = L[L.prev_is_labelled.astype(bool) & L.anchor_st23.notna()].copy()
    L["farm"], L["day"] = L.rec.str[:3], L.rec.str[4:7].astype(int)
    R = A2.A1.frames(); M, cells = R["DIAG10"]
    q = np.mean(cells, 0); M = M.assign(q=q, rec=M.farm + "_" + M.day.map("{:03d}".format))
    P = M.pivot_table(index="rec", columns="hour", values=["sub_temp", "q"])
    L = L[L.rec.isin(P.index)].copy()
    y = P["sub_temp"].loc[L.rec].values; qq = P["q"].loc[L.rec].values
    L["y0"], L["q0"] = y[:, 0], qq[:, 0]
    p1 = (L.day < 179).values; hi = (L.p >= .8).values
    # (1) midnight transfer on confident pass-1 records
    m = p1 & hi & np.isfinite(L.dT_air.values) & np.isfinite(L.y0.values)
    Xd = np.c_[np.ones(m.sum()), L.dT_air.values[m]]; c, k = np.linalg.lstsq(Xd, (L.y0 - L.anchor_st23).values[m], rcond=None)[0]
    print("(1) y0 - A = %.3f + %.3f dT_air   (pass-1 confident n %d; median y0-A %.3f; resid sd %.3f vs no-dT sd %.3f)" % (
        c, k, m.sum(), float(np.median((L.y0 - L.anchor_st23).values[m])), np.std((L.y0 - L.anchor_st23).values[m] - Xd @ [c, k]), np.std((L.y0 - L.anchor_st23).values[m])))
    for nm, mm in (("pass-2 check", (~p1) & hi & np.isfinite(L.dT_air.values) & np.isfinite(L.y0.values)),):
        Xd2 = np.c_[np.ones(mm.sum()), L.dT_air.values[mm]]; c2, k2 = np.linalg.lstsq(Xd2, (L.y0 - L.anchor_st23).values[mm], rcond=None)[0]
        print("    %s: %.3f + %.3f dT_air (n %d)" % (nm, c2, k2, mm.sum()))
    yhat0 = L.anchor_st23.values + c + k * L.dT_air.values
    g = yhat0 - L.q0.values
    # (2) hourly decay of information
    print("(2) beta_h (pass-1 confident): ", end="")
    beta = []
    for h in range(24):
        mm = p1 & hi & np.isfinite(g) & np.isfinite(y[:, h])
        b = float((g[mm] * (y[mm, h] - qq[mm, h])).sum() / (g[mm] ** 2).sum()); beta.append(b)
    print(" ".join("%d:%.2f" % (h, b) for h, b in enumerate(beta)))
    mm2 = (~p1) & hi & np.isfinite(g)
    print("    pass-2 check: " + " ".join("%d:%.2f" % (h, float((g[mm2] * (y[mm2, h] - qq[mm2, h])).sum() / (g[mm2] ** 2).sum())) for h in range(0, 24, 3)))
    # (3) linearity
    mm = p1 & hi & np.isfinite(g)
    bins = pd.cut(g[mm], [-9, -2, -1, -.5, 0, .5, 1, 2, 9])
    T = pd.DataFrame({"g": g[mm], "r0": (L.y0 - L.q0).values[mm], "b": bins}).groupby("b", observed=True).agg(n=("g", "size"), gap=("g", "mean"), resid0=("r0", "mean"))
    T["ratio"] = T.resid0 / T.gap; print("(3) linearity (pass-1 confident):\n" + T.round(2).to_string())
    # (4) confidence bins
    L["e_anchor"] = np.abs(yhat0 - L.y0.values); L["e_model"] = np.abs(L.q0 - L.y0)
    L["pb"] = pd.cut(L.p, [0, .5, .6, .7, .8, .9, .95, 1.01])
    print("(4) by linker p (pass-1):\n" + L[p1].groupby("pb", observed=True).agg(n=("p", "size"), anchor=("e_anchor", "median"), model=("e_model", "median"),
          better=("e_anchor", lambda s: (s < L.loc[s.index, "e_model"]).mean())).round(3).to_string())
    np.save(os.path.join(env.LOCAL, "linker_v7", "tm1_params.npy"), np.array([c, k] + beta))


if __name__ == "__main__":
    main()
