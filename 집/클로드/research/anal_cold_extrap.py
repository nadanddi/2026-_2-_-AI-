# -*- coding: utf-8 -*-
"""Cold-extrapolation rehearsal on the OTHER greenhouses (analysis).

F13/F47 have 21 labelled rows at <= 6 C while 9.2% of test rows are there
(catalog 6.54); the other greenhouses have hundreds.  Recreate the F13/F47
situation inside each other greenhouse: train only on rows with in_temp > 7,
predict its rows with in_temp <= 6, and measure the bias (pred - truth).

Model mirrors our temperature members: ridge physics base (in_temp and its
EWMs) + LightGBM residual on in_temp/in_hum/in_co2 history features
(temp_transfer_v1.feats, greenhouse flags dropped).

Question: does this kind of model UNDER-predict the cold rows (substrate
stays warmer than extrapolated, anal_cold_shape.py), consistently across
greenhouses, and does it depend on how strongly the greenhouse's offset
rises between 10-15 C and 8-10 C (F13/F47: weak, ~0.5 C)?

Run:  cd research && PYTHONPATH="" <python> -u anal_cold_extrap.py
"""
import env  # noqa: F401
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge

import common
from temp_transfer_v1 import feats

BASE = ["in_temp", "in_temp_e1", "in_temp_e3", "in_temp_e6", "in_temp_e12", "in_temp_e24"]


def main():
    tX, ty, _ = common.load_raw()
    lab = ty[["row_id", "sub_temp"]].dropna()
    rows = []
    for f, g in tX.groupby("farm"):
        if f in ("F13", "F47", "F32"):
            continue
        F = feats(g).merge(lab, on="row_id").dropna(subset=["sub_temp", "in_temp"])
        cold, warm = F[F.in_temp <= 6], F[F.in_temp > 7]
        if len(cold) < 60:
            continue
        cols = [c for c in F.columns if c not in ("row_id", "farm", "sub_temp", "is_F13", "is_F47")]
        Wb = warm[BASE].fillna(warm[BASE].median())
        base = Ridge(alpha=1.0).fit(Wb, warm.sub_temp)
        res = LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=50,
                            verbosity=-1, n_jobs=6, random_state=0)
        res.fit(warm[cols], warm.sub_temp - base.predict(Wb))
        Cb = cold[BASE].fillna(warm[BASE].median())
        p = base.predict(Cb) + res.predict(cold[cols])
        off = F.sub_temp - F.in_temp
        amp = off[(F.in_temp > 8) & (F.in_temp <= 10)].mean() - off[(F.in_temp > 10) & (F.in_temp <= 15)].mean()
        rows.append(dict(farm=f, n_cold=len(cold), bias=float((p - cold.sub_temp).mean()),
                         rmse=float(np.sqrt(((p - cold.sub_temp) ** 2).mean())), amp_8_10=float(amp),
                         ridge_only_bias=float((base.predict(Cb) - cold.sub_temp).mean())))
    R = pd.DataFrame(rows).set_index("farm").sort_values("amp_8_10")
    print(R.round(2).to_string())
    print("\ngreenhouses %d | bias < 0 (under-predicts cold) in %d | median bias %+.2f"
          % (len(R), int((R.bias < 0).sum()), R.bias.median()))
    lo = R[R.amp_8_10 < 0.9]
    print("weak-amplitude greenhouses (8-10C rise < 0.9, like F13 0.58 / F47 0.42): n=%d, median bias %+.2f, "
          "bias<0 in %d" % (len(lo), lo.bias.median(), int((lo.bias < 0).sum())))
    print("corr(amp_8_10, bias) = %.2f" % R[["amp_8_10", "bias"]].corr().iloc[0, 1])


if __name__ == "__main__":
    main()
