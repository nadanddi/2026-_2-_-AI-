# -*- coding: utf-8 -*-
"""submission_28 temperature, last step = make_temp_v27_tx3.py with the constants re-shaped (CB4-T, catalog 6.955: c .40, T0 12, e0 .7;
chosen on cold layout CK2, judged on EXT10/EXT12 -2.0 / -0.75 %, P .0009; in the full assembly ASM2 6.957). Original description:
submission_27 temperature, last step (2026-10-15 집 클로드; catalog 6.799 TX2, 6.931 TX3 ADOPT, 5-point class F).
High-EC house, cold-row correction:  pred_h = q_h + .25 * w_h * g_h
  w_h = clip((cmEC_h - .8) / .4, 0, 1),  cmEC_h = mean of the submission's own EC base prediction NH (EC/output_27/ec_nh.csv) over hours 0..h of
        the same record (no labels; own record, current/earlier hours only);
  g_h = clip((10 - in_temp_h) / 2, 0, 1) (missing in_temp -> 0) — current indoor air temperature of the same row.
  .25 C = substrate-vs-input-model gap of high-EC houses measured on TRAINING labels (HX9, 6.797); not fitted to validation or evaluation data.
Local evidence (TX3, on the round-27 configuration): 11 layouts x 4 cells all better, cold EXT10 -.44 % (P .0047), warm layouts -.07..-.10 %.
Run:  python make_temp_v28_tx3.py <temp_in.csv> <ec_nh.csv> <test_X.csv> <temp_out.csv>
"""
import os, sys
import numpy as np, pandas as pd


def main():
    tp, ep, xp, out = sys.argv[1:5]
    assert not os.path.exists(out), "refusing to overwrite " + out
    T = pd.read_csv(tp); E = pd.read_csv(ep)[["row_id", "sub_ec"]]; X = pd.read_csv(xp)[["row_id", "in_temp"]]
    M = T[["row_id", "sub_temp"]].merge(E, on="row_id", how="left", validate="one_to_one").merge(X, on="row_id", how="left", validate="one_to_one")
    assert len(M) == 1440 and M.sub_ec.notna().all()
    M["rec"] = M.row_id.str[:7]; M["h"] = M.row_id.str[8:10].astype(int)
    cm = M.sort_values(["rec", "h"]).groupby("rec").sub_ec.transform(lambda s: s.expanding().mean()).reindex(M.index)
    w = np.clip((cm.values - .7) / .4, 0, 1); g = np.nan_to_num(np.clip((12 - M.in_temp.values) / 2, 0, 1), nan=0.0); add = .40 * w * g
    S = M[["row_id"]].copy(); S["sub_temp"] = M.sub_temp.values + add
    assert S.row_id.tolist() == T.row_id.tolist() and np.isfinite(S.sub_temp).all()
    S.to_csv(out, index=False, float_format="%.6f")
    print("wrote %s | rows changed %d (records %d), mean add %+.4f, max %+.3f" % (out, int((add > 1e-9).sum()), M.rec[add > 1e-9].nunique(), add.mean(), add.max()))


if __name__ == "__main__":
    main()
