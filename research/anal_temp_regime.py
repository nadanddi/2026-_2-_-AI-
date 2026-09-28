# -*- coding: utf-8 -*-
"""Temperature error by indoor-temperature band, reweighted to the TEST
composition (analysis only).  Candidate reference 0.8*MASK base + 0.2*Codex
(temp_mask_v1_oof.npz, seed 7 / 726).

Two error sources per band:
  DIAG10  ordinary cross-validation (cold days have warm neighbours in train)
  EXT10   all cold days held out at once (extrapolation, closer to the test)
Test band shares from test_X in_temp.  Projection = sqrt(sum share * MSE).
Compared with our LB 0.5456 (round 5).

Run:  cd research && PYTHONPATH="" <python> -u anal_temp_regime.py
"""
import env  # noqa: F401
import numpy as np
import pandas as pd

import common
from harness import load

BANDS = [-99, 6, 8, 10, 12, 15, 99]


def main():
    tX, ty, sX = common.load_raw()
    _, lab, _ = load()
    z = np.load(env.LOCAL + "/temp_mask_v1_oof.npz", allow_pickle=True)
    assert (lab.row_id.values == z["row_id"]).all()
    y = lab.sub_temp.values
    band = pd.cut(lab.in_temp.values, BANDS)
    tb = pd.cut(sX.in_temp.values, BANDS)
    share_te = pd.Series(tb).value_counts(normalize=True).sort_index()
    share_tr = pd.Series(band).value_counts(normalize=True).sort_index()
    rows = []
    for s in ("DIAG10", "EXT10", "EXT12"):
        ref = 0.8 * z["%s__MASK__7" % s] + 0.2 * z["%s__CODEX__726" % s]
        g = ~np.isnan(ref)
        e2 = pd.Series((ref - y)[g] ** 2).groupby(np.asarray(band)[g]).agg(["mean", "size"])
        e2["bias"] = pd.Series((ref - y)[g]).groupby(np.asarray(band)[g]).mean()
        for b, r in e2.iterrows():
            rows.append(dict(val=s, band=str(b), rmse=np.sqrt(r["mean"]), n=int(r["size"]), bias=r["bias"]))
    R = pd.DataFrame(rows)
    print(R.pivot(index="band", columns="val", values="rmse").round(3).to_string())
    print("\nrows per band:")
    print(R.pivot(index="band", columns="val", values="n").to_string())
    print("\nbias (pred - truth):")
    print(R.pivot(index="band", columns="val", values="bias").round(3).to_string())
    print("\nshare  train-label %s" % dict(share_tr.round(3)))
    print("share  test        %s" % dict(share_te.round(3)))
    D = R[R.val == "DIAG10"].set_index("band").rmse
    E = R[R.val == "EXT10"].set_index("band").rmse
    mix = D.copy()
    for b in E.index:          # use extrapolation error where cold days were held out as a block
        if b in mix.index and b in ("(-99, 6]", "(6, 8]", "(8, 10]"):
            mix[b] = E[b]
    for name, err in (("DIAG10 errors", D), ("DIAG10 warm + EXT10 cold", mix)):
        sh = share_te.rename(index=str).reindex(err.index).fillna(0)
        print("projection with %-25s -> %.4f (LB 0.5456)" % (name, np.sqrt((sh * err ** 2).sum() / sh.sum())))


if __name__ == "__main__":
    main()
