# -*- coding: utf-8 -*-
"""EC stage-3 LA1 (diagnostic, fixed before running; 2026-10-03 집 클로드).
Survey D: audit input-label lag alignment per source and period.  If the label
of a source is shifted against its inputs (day level or hour level), a model
that uses same-day inputs mis-reads the state.  Training rows only, train_X
inputs only, no test_X.
Groups: farm (F13/F47) x pass (record day <179 / >=179) x parity (= source/동).
Day level: day means of 14 inputs (4 outdoor + 10 indoor/actuators), standardized.
  (a) label day mean y(d)  vs inputs X(d+L), L in -6,-4,-2,0,+2,+4,+6 record days
      (even L = same source); leave-one-day-out ridge(alpha 10) R^2.
  (b) R3S DIAG10 day residual e(d) (seed mean) vs X(d+L), same measure.
Hour level: label deviation from its day mean vs input deviations shifted by k
  hours within the day (k=-6..6), pooled, 5-fold-by-day ridge R^2.
Reading (fixed): a misalignment clue = best L (or k) != 0 with R^2 gain >= .05
over lag 0 in a group; otherwise alignment is fine.  Any clue -> a separate
pre-registered model test (only lags that are legal at evaluation: L<0 / k<=0).
"""
import env  # noqa: F401
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

V = ["out_temp", "out_hum", "out_rad", "out_wspd", "in_temp", "in_hum", "in_co2", "act_vent", "act_shade",
     "act_thermal", "act_heating", "act_circfan", "act_co2", "act_fog"]


def loo_r2(X, y, alpha=10.0):
    X, y = np.asarray(X), np.asarray(y)
    m = ~np.isnan(X).any(1) & ~np.isnan(y)
    X, y = X[m], y[m]
    if len(y) < 12:
        return np.nan, len(y)
    Xc = np.c_[np.ones(len(y)), X]
    A = Xc.T @ Xc + alpha * np.diag([0] + [1] * X.shape[1])
    H = Xc @ np.linalg.solve(A, Xc.T)
    res = (y - H @ y) / (1 - np.diag(H))
    return 1 - np.sum(res ** 2) / np.sum((y - y.mean()) ** 2), len(y)


def main():
    X = pd.read_csv(os.path.join(env.DATA, "train_X.csv"), usecols=["row_id"] + V)
    X = X[X.row_id.str[:3].isin(["F13", "F47"])].copy()
    X["farm"], X["day"], X["hour"] = X.row_id.str[:3], X.row_id.str[4:7].astype(int), X.row_id.str[8:10].astype(int)
    Y = pd.read_csv(os.path.join(env.DATA, "train_y.csv"), usecols=["row_id", "sub_ec"])
    H = X.merge(Y, on="row_id", how="left")
    DX = X.groupby(["farm", "day"])[V].mean()
    DX = (DX - DX.mean()) / DX.std()
    DY = H.groupby(["farm", "day"]).sub_ec.mean()
    O = pd.read_csv(os.path.join(env.LOCAL, "ec2_DC5_oof.csv"))
    O = O[O.validator == "DIAG10"]
    O["e"] = O[["r3s_7", "r3s_101", "r3s_2024"]].mean(1) - O.sub_ec
    DE = O.groupby(["farm", "day"]).e.mean()
    LAGS = (-6, -4, -2, 0, 2, 4, 6)
    print("DAY LEVEL: LOO ridge R^2 by lag (record days; even = same source)")
    clues = []
    for tgt_name, T in (("label y", DY), ("R3S resid e", DE)):
        print("\n[%s]" % tgt_name)
        print("%-4s %-5s %-4s %4s " % ("farm", "pass", "par", "n") + " ".join("%7s" % ("L%+d" % L) for L in LAGS))
        for f in ("F13", "F47"):
            for ps in (0, 1):
                for par in (0, 1):
                    days = [d for (ff, d) in T.index if ff == f and (d >= 179) == ps and d % 2 == par]
                    row, n0 = [], 0
                    for L in LAGS:
                        xs = np.array([DX.loc[(f, d + L)].values if (f, d + L) in DX.index and ((d + L) >= 179) == ps
                                       else np.full(len(V), np.nan) for d in days])
                        r2, n = loo_r2(xs, T.loc[[(f, d) for d in days]].values)
                        row.append(r2)
                        if L == 0:
                            n0 = n
                    row = np.array(row)
                    print("%-4s %-5s %-4d %4d " % (f, "late" if ps else "early", par, n0) + " ".join("%7.3f" % v for v in row))
                    b = int(np.nanargmax(row))
                    if LAGS[b] != 0 and row[b] - row[3] >= 0.05:
                        clues.append((tgt_name, f, ps, par, LAGS[b], round(row[b] - row[3], 3)))
    print("\nHOUR LEVEL: within-day deviations, 5-fold-by-day ridge R^2 by hour shift k (input at h+k)")
    H = H.sort_values(["farm", "day", "hour"])
    for v in V:
        H[v + "_dev"] = H[v] - H.groupby(["farm", "day"])[v].transform("mean")
    H["y_dev"] = H.sub_ec - H.groupby(["farm", "day"]).sub_ec.transform("mean")
    KS = range(-6, 7)
    print("%-4s %-5s %-4s " % ("farm", "pass", "par") + " ".join("%6s" % ("k%+d" % k) for k in KS))
    for f in ("F13", "F47"):
        for ps in (0, 1):
            for par in (0, 1):
                G = H[(H.farm == f) & ((H.day >= 179) == ps) & (H.day % 2 == par)]
                row = []
                for k in KS:
                    S = G.groupby("day")[[v + "_dev" for v in V]].shift(-k)
                    D = pd.concat([S, G[["y_dev", "day"]]], axis=1).dropna()
                    pr = np.zeros(len(D))
                    for a, b in GroupKFold(5).split(D, groups=D.day):
                        m = Ridge(alpha=10).fit(D.iloc[a, :len(V)], D.y_dev.iloc[a])
                        pr[b] = m.predict(D.iloc[b, :len(V)])
                    row.append(1 - np.sum((D.y_dev - pr) ** 2) / np.sum((D.y_dev - D.y_dev.mean()) ** 2))
                row = np.array(row)
                print("%-4s %-5s %-4d " % (f, "late" if ps else "early", par) + " ".join("%6.3f" % v for v in row))
                b = int(np.argmax(row))
                if list(KS)[b] != 0 and row[b] - row[6] >= 0.05:
                    clues.append(("hour", f, ps, par, list(KS)[b], round(row[b] - row[6], 3)))
    print("\nLA1 clues (best lag != 0 and gain >= .05):", clues if clues else "none")


if __name__ == "__main__":
    main()
