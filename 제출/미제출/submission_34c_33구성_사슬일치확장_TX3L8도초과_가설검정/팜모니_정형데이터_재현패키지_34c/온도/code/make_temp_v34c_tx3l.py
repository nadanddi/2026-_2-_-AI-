# -*- coding: utf-8 -*-
"""submission_34c temperature, TX3 step = TX3L (data-chosen constants .5 / 14 C / EC 1.0, catalog 7.026) ONLY on rows with indoor air > 8 C;
rows at or below 8 C keep the round-27 hand TX3 (.25 / 10 C / EC .8) exactly as submission_33.
Why: on every validation layout TX3L changed no row at or below 8 C (validation high-EC records are not that cold), so TX3L's effect there is unvalidated;
in the evaluation set 61 of its 182 rows are <= 8 C (deep-cold high-EC records F47_221, F13_225, F47_223 ...).  8 C = the existing W40G-S gate start
(rediscovered by data, catalog 7.017).  On the validation layouts this file gives exactly the B34 numbers (7.048: 12/12 better, -1.17 %, P .019).
Run:  python make_temp_v34c_tx3l.py <temp_in.csv> <ec_nh.csv> <test_X.csv> <temp_out.csv>
"""
import os, sys
import numpy as np, pandas as pd


def add_tx3(cm, t, c, T0, e0):
    w = np.clip((cm - e0) / .4, 0, 1); g = np.nan_to_num(np.clip((T0 - t) / 2, 0, 1), nan=0.0); return c * w * g


def main():
    tp, ep, xp, out = sys.argv[1:5]
    assert not os.path.exists(out), "refusing to overwrite " + out
    T = pd.read_csv(tp); E = pd.read_csv(ep)[["row_id", "sub_ec"]]; X = pd.read_csv(xp)[["row_id", "in_temp"]]
    M = T[["row_id", "sub_temp"]].merge(E, on="row_id", how="left", validate="one_to_one").merge(X, on="row_id", how="left", validate="one_to_one")
    assert len(M) == 1440 and M.sub_ec.notna().all()
    M["rec"] = M.row_id.str[:7]; M["h"] = M.row_id.str[8:10].astype(int)
    cm = M.sort_values(["rec", "h"]).groupby("rec").sub_ec.transform(lambda s: s.expanding().mean()).reindex(M.index).values; t = M.in_temp.values
    add = np.where(np.nan_to_num(t, nan=99) > 8, add_tx3(cm, t, .5, 14.0, 1.0), add_tx3(cm, t, .25, 10.0, .8))
    S = M[["row_id"]].copy(); S["sub_temp"] = M.sub_temp.values + add
    assert S.row_id.tolist() == T.row_id.tolist() and np.isfinite(S.sub_temp).all()
    S.to_csv(out, index=False, float_format="%.6f")
    print("wrote %s | rows changed %d (records %d), mean add %+.4f, max %+.3f | <= 8C rows (hand TX3) %d, > 8C rows (TX3L) %d"
          % (out, int((add > 0).sum()), M.rec[add > 0].nunique(), add.mean(), add.max(), int(((add > 0) & (np.nan_to_num(t, nan=99) <= 8)).sum()), int(((add > 0) & (np.nan_to_num(t, nan=99) > 8)).sum())))


if __name__ == "__main__":
    main()
