# -*- coding: utf-8 -*-
"""Which model TYPE extrapolates best into cold it has never seen?  (Supports
or refutes G_C2, which moves weight to the Codex-type member below 8-10 C;
9.2% of test rows are <= 6 C with only 21 labelled F13/F47 rows there.)

Rehearsal on the other greenhouses (F13/F47/F32 excluded, >= 60 cold rows):
train on rows with in_temp > 8, predict rows with in_temp <= 6.
Features: in_temp/in_hum/in_co2 history (temp_transfer_v1.feats).
  TREE    LightGBM only                     (like the base / TabPFN: no extrapolation)
  LINRES  ridge on in_temp + EWMs, LightGBM on the residual (Codex type)
  LIN     ridge only
  MIX     0.4 TREE + 0.6 LINRES             (the G_C2 cold-row split)
Per greenhouse RMSE and bias on the cold rows; wins counted.  Analysis only.

Run:  cd research && PYTHONPATH="" <python> -u anal_cold_extrap_v2.py
"""
import env  # noqa: F401
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge

import common
from temp_transfer_v1 import feats

BASE = ["in_temp", "in_temp_e1", "in_temp_e3", "in_temp_e6", "in_temp_e12", "in_temp_e24"]


def lgb(seed=0):
    return LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=50,
                         verbosity=-1, n_jobs=6, random_state=seed)


def main():
    tX, ty, _ = common.load_raw()
    lab = ty[["row_id", "sub_temp"]].dropna()
    rows = []
    for f, g in tX.groupby("farm"):
        if f in ("F13", "F47", "F32"):
            continue
        F = feats(g).merge(lab, on="row_id").dropna(subset=["sub_temp", "in_temp"])
        cold, warm = F[F.in_temp <= 6], F[F.in_temp > 8]
        if len(cold) < 60:
            continue
        cols = [c for c in F.columns if c not in ("row_id", "farm", "sub_temp", "is_F13", "is_F47")]
        med = warm[BASE].median()
        Wb, Cb = warm[BASE].fillna(med), cold[BASE].fillna(med)
        yc = cold.sub_temp.values
        tree = lgb().fit(warm[cols], warm.sub_temp).predict(cold[cols])
        lin = Ridge(alpha=1.0).fit(Wb, warm.sub_temp)
        linres = lin.predict(Cb) + lgb().fit(warm[cols], warm.sub_temp - lin.predict(Wb)).predict(cold[cols])
        P = {"TREE": tree, "LINRES": linres, "LIN": lin.predict(Cb), "MIX": 0.4 * tree + 0.6 * linres}
        r = {"farm": f, "n": len(cold)}
        for k, p in P.items():
            r[k] = float(np.sqrt(((p - yc) ** 2).mean()))
            r[k + "_bias"] = float((p - yc).mean())
        rows.append(r)
    R = pd.DataFrame(rows).set_index("farm")
    print(R.round(2).to_string())
    ks = ["TREE", "LINRES", "LIN", "MIX"]
    print("\nmedian RMSE: %s" % {k: round(R[k].median(), 3) for k in ks})
    print("median bias: %s" % {k: round(R[k + "_bias"].median(), 3) for k in ks})
    print("best model count: %s" % R[ks].idxmin(axis=1).value_counts().to_dict())
    print("LINRES better than TREE in %d / %d greenhouses; MIX better than TREE in %d / %d"
          % ((R.LINRES < R.TREE).sum(), len(R), (R.MIX < R.TREE).sum(), len(R)))


if __name__ == "__main__":
    main()
