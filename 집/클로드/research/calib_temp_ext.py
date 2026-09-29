# -*- coding: utf-8 -*-
"""Second calibration point for the temperature extrapolation validator.

cold_v5.py: on the EXTRAP fold (all greenhouse-days colder than 10 C held out
at once) round 3 / round 2 = 0.858 vs 0.844 on the leaderboard.  That is one
point.  The other real change we have is round 1 -> round 2: 0.7450 -> 0.6666,
ratio 0.895, which the geometry CV missed (0.950 / 0.978).  If EXTRAP also
lands near 0.895 it is calibrated on both real temperature changes.

  R1  LightGBM huber on the 98-column features_v2 temperature view
  R2  93 columns, 0.65 LGB + 0.25 Ridge + 0.10 Nystroem
  (R2 on EXTRAP was 1.0589 in cold_v5.py; recomputed here on the same fold)

Run:  cd research && PYTHONPATH="" <python> -u calib_temp_ext.py
"""
import env  # noqa: F401
import numpy as np

from common import split_mask, rmse, TARGET_FARMS
from harness import load, views
import feat_temp74 as T74
import feat_new
import features_v4 as F4
from cold_v5 import lgbh, ridge, nys, THRESH


def main():
    panel, lab0, _ = load()
    v = views(panel)
    ex, blocks = feat_new.build_extra()
    ph = F4.phys_features()
    lab = lab0.merge(ex, on="row_id", how="left").merge(ph, on="row_id", how="left")
    f98 = v["temp"]
    f93 = T74.base74(f98) + list(blocks["dew"]) + list(blocks["event"])
    y = lab.sub_temp.values

    dmin = lab.groupby(["farm", "day"]).ph_in_temp_3.min()
    cd = dmin[dmin < THRESH]
    fd = {f: set(int(d) for (ff, d) in cd.index if ff == f) for f in TARGET_FARMS}
    trm, vam = split_mask(lab, fd)
    tr, va = lab[trm], lab[vam]
    yv = y[vam]

    p1 = lgbh().fit(tr[f98], tr.sub_temp.values).predict(va[f98])
    yt = tr.sub_temp.values
    p2 = (0.65 * lgbh().fit(tr[f93], yt).predict(va[f93])
          + 0.25 * ridge().fit(tr[f93], yt).predict(va[f93])
          + 0.10 * nys().fit(tr[f93], yt).predict(va[f93]))
    r1, r2 = rmse(p1, yv), rmse(p2, yv)
    print("EXTRAP (< %.0f C, %d rows): R1 %.4f  R2 %.4f  R2/R1 %.3f   (real 0.895)"
          % (THRESH, len(yv), r1, r2, r2 / r1))
    print("with the earlier point: R3/R2 on EXTRAP 0.858 (real 0.844)")


if __name__ == "__main__":
    main()
