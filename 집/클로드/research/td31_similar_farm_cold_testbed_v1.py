# -*- coding: utf-8 -*-
"""TD31: cold-extrapolation TESTBED on greenhouses similar to F13/F47 (2026-10-10 집 클로드, user: "F13·47과 다른 온실로
예상되는 애들은 빼버리고, 제안 후보 진행해 보자").  Diagnostic only (which model STRUCTURE extrapolates best below 6 C on
F13/F47-like substrates); no adoption.  Selection fixed before running, from training data only (TD30 table):
  similar = not heated (mean sub - in at in_temp >= 20 < 0) AND mean (sub - in) at in_temp 6-10 <= 2.5 C
            AND daily amplitude ratio >= .35 AND lag <= 5 h, excluding F13/F47.
  testbed = similar AND >= 60 rows with in_temp < 6.
Per testbed farm: features = re13 panel (in_temp/in_hum/in_co2 histories, vpd, dew gap; MASK-style from train_X), train
on that farm's rows with in_temp >= 8, test on rows with in_temp < 6 (and report 6-8).  Models:
  TREE    LightGBM (400 trees, lr .05, leaves 31, min_child 30)
  LIN     Ridge(alpha 10, standardized)
  LINRES  LIN + LightGBM residual (220 trees, leaves 12, min_child 100 - CODEX-like)
  MIX     0.4 TREE + 0.6 LINRES   (W40G-S weights on g = 0 rows, without TabPFN)
  GAPLIN  Ridge on the same features predicting (sub - in_temp), + in_temp
  NAIVE   in_temp + mean(sub - in_temp) of the training rows
Report RMSE and bias (pred - label) on < 6 C and 6-8 C rows per farm and pooled.
Run:  cd 집/클로드/research && PYTHONPATH="" py -3.12 -u td31_similar_farm_cold_testbed_v1.py
"""
import env  # noqa: F401
import os
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import re13_pooled51_v1 as R13
r = lambda e: float(np.sqrt(np.mean(np.square(e))))


def main():
    P = R13.build_panel()
    feats = [c for c in P.columns if c not in ("farm", "day", "hour", "row_id", "sub_temp", "farm_cat")]
    T = pd.read_csv(env.LOCAL + "/td26_farm_profiles_v1.csv", index_col=0)
    rows = []
    for f, x in P.groupby("farm"):
        d = x.sub_temp - x.in_temp
        rows.append(dict(farm=f, gw=d[x.in_temp >= 20].mean(), g610=d[(x.in_temp >= 6) & (x.in_temp < 10)].mean(), n6=int((x.in_temp < 6).sum())))
    S = pd.DataFrame(rows).set_index("farm").join(T[["amp", "lag"]])
    sim = S[(S.gw < 0) & (S.g610 <= 2.5) & (S.amp >= .35) & (S.lag <= 5) & ~S.index.isin(["F13", "F47"])]
    bed = sim[sim.n6 >= 60]
    print("닮은 온실:", list(sim.index), "| 시험대(6℃ 미만 ≥60행):", list(bed.index))
    print(sim.round(2).to_string())
    lin = lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=10))
    res = {}
    for f in bed.index:
        x = P[P.farm == f].dropna(subset=["in_temp"])
        tr, te6, te8 = x[x.in_temp >= 8], x[x.in_temp < 6], x[(x.in_temp >= 6) & (x.in_temp < 8)]
        Xtr, y = tr[feats], tr.sub_temp.values
        tree = lgb.LGBMRegressor(n_estimators=400, learning_rate=.05, num_leaves=31, min_child_samples=30, verbose=-1).fit(Xtr, y)
        L = lin().fit(Xtr, y)
        resm = lgb.LGBMRegressor(n_estimators=220, learning_rate=.035, num_leaves=12, min_child_samples=100, reg_lambda=15, verbose=-1).fit(Xtr, y - L.predict(Xtr))
        G = lin().fit(Xtr, y - tr.in_temp.values)
        mg = (y - tr.in_temp.values).mean()
        for nm, te in (("<6", te6), ("6-8", te8)):
            X = te[feats]; yy = te.sub_temp.values
            pr = {"TREE": tree.predict(X), "LIN": L.predict(X)}
            pr["LINRES"] = pr["LIN"] + resm.predict(X)
            pr["MIX"] = .4 * pr["TREE"] + .6 * pr["LINRES"]
            pr["GAPLIN"] = G.predict(X) + te.in_temp.values
            pr["NAIVE"] = te.in_temp.values + mg
            for k, p in pr.items():
                res.setdefault((nm, k), []).append((p - yy))
            print("  %s %-4s n%4d | " % (f, nm, len(te)) + " ".join("%s %.2f(%+.2f)" % (k, r(p - yy), (p - yy).mean()) for k, p in pr.items()))
    print("\n합산 (괄호: 평균 치우침 = 예측-정답)")
    for nm in ("<6", "6-8"):
        print("  %-4s " % nm + " | ".join("%s %.3f (%+.2f)" % (k, r(np.concatenate(res[(nm, k)])), np.concatenate(res[(nm, k)]).mean()) for k in ("TREE", "LIN", "LINRES", "MIX", "GAPLIN", "NAIVE")))


if __name__ == "__main__":
    main()
