# -*- coding: utf-8 -*-
"""CH2: can the F47 test days 218-227 be attached to the high-EC source (chain 31)
or the normal source (chain 32) using RECORD-EARLIER inputs only?  Statistical
critic's single remaining check for the CH1 path (2026-10-02); if this fails
the path is closed by the regulation.  2026-10-02 집 클로드.

Design (fixed before running)
  Day signature = the day's 0-h values (legal for every hour of that day):
    in_temp, in_hum, in_co2, in_temp - out_temp, act_heating, act_thermal,
    act_shade, act_circfan, act_vent, act_co2, act_fog   (z-scored on train set)
  Training days: F47 days with deep_cal_11 chain 31 (high) or 32 (normal) and
    day < 218 (record-earlier than the test block) -> days 132-148 region.
  Classifier: nearest centroid (no tuning).  Identifiability rule:
    leave-one-out accuracy on the training days >= 0.80  AND  the record-later
    chain-31/other days 229-231 (not used for training) classified as their
    chain (descriptive check of transfer from the 1st to the 2nd segment).
  If identifiable: report the class of each test day 218-227 and of the
    labelled days 212-216 just before the block (which of their sources is
    the high one?).  No label is used for the test days; no model is changed.
Run:  cd 집/클로드/research && PYTHONPATH="" py -3.12 -u ch2_causal_source_id_v1.py
"""
import env  # noqa: F401
import os

import numpy as np
import pandas as pd

F = ["in_temp", "in_hum", "in_co2", "dTio", "act_heating", "act_thermal", "act_shade", "act_circfan", "act_vent",
     "act_co2", "act_fog"]


def main():
    tX = pd.read_csv(os.path.join(env.DATA, "train_X.csv"))
    sX = pd.read_csv(os.path.join(env.DATA, "test_X.csv"))
    X = pd.concat([tX, sX], ignore_index=True)
    p = X.row_id.str.split("_", expand=True)
    X["farm"], X["day"], X["hour"] = p[0], p[1].astype(int), p[2].astype(int)
    X = X[(X.farm == "F47") & (X.hour == 0)].copy()
    X["dTio"] = X.in_temp - X.out_temp
    C = pd.read_csv(os.path.join(env.LOCAL, "deep_cal_11_days.csv"))[["farm", "day", "cal", "chain", "ec_m", "is_test"]]
    D = X.merge(C[C.farm == "F47"], on=["farm", "day"], how="left")
    tr = D[D.chain.isin([31, 32]) & (D.day < 218)].copy()
    print("training days:", sorted(zip(tr.day, tr.chain.astype(int))))
    mu, sd = tr[F].mean(), tr[F].std().replace(0, 1)
    Z = lambda d: ((d[F] - mu) / sd).fillna(0).values
    zt, yt = Z(tr), tr.chain.values

    def predict(zf, ztr, ytr):
        c31, c32 = ztr[ytr == 31].mean(0), ztr[ytr == 32].mean(0)
        d31, d32 = np.linalg.norm(zf - c31, axis=1), np.linalg.norm(zf - c32, axis=1)
        return np.where(d31 < d32, 31, 32), d31, d32

    loo = []
    for i in range(len(zt)):
        m = np.arange(len(zt)) != i
        loo.append(predict(zt[i:i + 1], zt[m], yt[m])[0][0])
    acc = float(np.mean(np.array(loo) == yt))
    print("LOO accuracy %.2f (n=%d; chance ~%.2f)" % (acc, len(yt), max(np.mean(yt == 31), np.mean(yt == 32))))
    late = D[D.day.isin([229, 230, 231])].copy()
    pl, _, _ = predict(Z(late), zt, yt)
    late["pred"] = pl
    print("record-later check:", late[["day", "cal", "chain", "ec_m", "pred"]].to_string(index=False))
    transfer = all((r.chain == 31) == (r.pred == 31) for r in late.itertuples())
    ok = acc >= 0.80 and transfer
    print("identifiable:", ok)
    q = D[D.day.between(210, 227)].copy()
    pq, d31, d32 = predict(Z(q), zt, yt)
    q["pred"], q["d31"], q["d32"] = pq, d31, d32
    print("\nF47 days 210-227 (test days have no label):")
    print(q[["day", "cal", "chain", "is_test", "ec_m", "pred", "d31", "d32"]].round(2).to_string(index=False))
    print("\nper-feature z means: chain31 vs chain32 (training)")
    print(pd.DataFrame({"c31": zt[yt == 31].mean(0), "c32": zt[yt == 32].mean(0)}, index=F).round(2).T.to_string())


if __name__ == "__main__":
    main()
