# -*- coding: utf-8 -*-
"""submission_32 temperature, stage 0 (2026-10-16 집 클로드; catalog 7.025 OOD1, 7.042 BND1).
Out-of-training-range shift toward the Codex member:  cell' = cell + s * (Codex - cell),
  s = sigmoid(a + b * clip(tmin - t, -20, 10) / 2 + c * clip(tmin - cm, -20, 10) / 2),  (a, b, c) = (-1.32, 0.29, 1.50)
  t = row indoor air temperature, cm = minimum indoor air temperature of the same record over hours 0..h (current/earlier only),
  tmin = minimum indoor air temperature over the farm's LABELLED training rows (training data only).  Missing t -> s = 0.
  (a, b, c) = OOD1 fit on all validation blocks (EXT12 + DIAG10); the two block groups learned the same curve (s ~ .6 at 2 C below the
  training minimum, ~.95 at 5 C below, ~0 inside the range).  No hand-set temperature threshold.
Codex = the NX Codex member (make_submission_v13_temp.py members, `second` removed as in NX / make_submission_v26_temp.py), refitted here.
Run:  cd code && python make_temp_v32_ood1.py <nx27 cell csv> <out.csv>      (data from env.DATA / AGRI_DATA)
"""
import os, sys
import env  # noqa: F401
import env_extra  # noqa: F401
import numpy as np, pandas as pd
import make_submission_v13_temp as V13
import train_flags_v6 as TF
import resid_reset_features as RR
from temp_mask_v1 import codex_fit_predict
import common

A_, B_, C_ = -1.32, 0.29, 1.50


def main():
    nxp, out = sys.argv[1:3]
    assert not os.path.exists(out), "refusing to overwrite " + out
    if "second" in RR.FEATURE_COLUMNS: RR.FEATURE_COLUMNS.remove("second")
    lab, test, ct, phc, sX = V13.build_frames()
    w = TF.row_weights(lab, V13.W_FLAG, w_noisy=V13.W_NOISY)
    ctr, cte = V13.codex_frame(lab), V13.codex_frame(test)
    codex = np.mean([codex_fit_predict(ctr, cte, w, s) for s in V13.CODEX_SEEDS], axis=0)
    tX, ty, sX0 = common.load_raw()
    L = tX.merge(ty[["row_id", "sub_temp"]], on="row_id"); L = L[L.sub_temp.notna() & L.farm.isin(["F13", "F47"])]
    tmin = L.groupby("farm").in_temp.min().to_dict()
    T = test[["row_id", "farm", "day", "hour", "in_temp"]].copy(); T["codex"] = codex
    T["cell"] = pd.read_csv(nxp).set_index("row_id").sub_temp.loc[T.row_id].values
    t = T.in_temp.values.astype(float); tt = np.nan_to_num(t, nan=15.0)
    T["tt"] = tt; T = T.sort_values(["farm", "day", "hour"]); T["cm"] = T.groupby(["farm", "day"]).tt.cummin(); T = T.sort_index()
    tm = T.farm.map(tmin).values
    z = A_ + B_ * np.clip(tm - T.tt.values, -20, 10) / 2 + C_ * np.clip(tm - T.cm.values, -20, 10) / 2
    s = np.where(np.isnan(t), 0.0, 1 / (1 + np.exp(-z)))
    new = T.cell.values + s * (T.codex.values - T.cell.values)
    O = pd.DataFrame({"row_id": T.row_id.values, "sub_temp": new}).set_index("row_id").loc[sX.row_id].reset_index()
    assert len(O) == 1440 and np.isfinite(O.sub_temp).all()
    O.to_csv(out, index=False, encoding="utf-8", float_format="%.6f")
    d = new - T.cell.values
    print("wrote %s | training min in_temp %s | rows with s > .1: %d | change mean %+.3f max |%.3f| | records changed %d"
          % (out, {k: round(v, 2) for k, v in tmin.items()}, int((s > .1).sum()), d.mean(), np.abs(d).max(), int(pd.Series(np.abs(d) > 1e-6).groupby(T.row_id.str[:7].values).any().sum())))


if __name__ == "__main__":
    main()
