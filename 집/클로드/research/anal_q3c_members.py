# -*- coding: utf-8 -*-
"""Analysis Q3c: WHICH blend member is distorted by the contaminated rows?

Down-weighting the 167 input-flagged training rows moves the calibrated
EXTRAP fold 0.9090 -> 0.8815.  The temperature blend has two linear parts --
the least-squares physics baseline (under the LGB residual) and Ridge -- plus
a Nystroem kernel ridge; the LGB residual learner uses a Huber loss and should
already be robust.  Linear least squares is outlier-sensitive and does the
extrapolation, so the hypothesis is that the linear parts carry the damage.

On the EXTRAP fold (single fold), each member plain vs down-weighted, clean
rows scored; plus the baseline coefficients with and without the flagged rows.

Run:  cd research && PYTHONPATH="" <python> anal_q3c_members.py
"""
import env  # noqa: F401
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression

from common import split_mask, rmse, TARGET_FARMS
from harness import load, views
import feat_temp74 as T74
import feat_new
import features_v4 as F4
from cold_v5 import THRESH
from screen_v6 import temp_members
from cleanw_v6 import weights


def main():
    panel, lab0, _ = load()
    v = views(panel)
    ex, blocks = feat_new.build_extra()
    sg, ph, fp = F4.seg_features(), F4.phys_features(), F4.fp_features()
    lab = (lab0.merge(ex, on="row_id", how="left").merge(sg, on="row_id", how="left")
               .merge(ph, on="row_id", how="left").merge(fp, on="row_id", how="left")).copy()
    f93 = T74.base74(v["temp"]) + list(blocks["dew"]) + list(blocks["event"])
    ct = f93 + F4.names(sg) + F4.names(fp)
    phc = F4.names(ph)
    W = weights(lab, 0, 0.2)
    clean = weights(lab, 3, 0.0) >= 1

    dmin = lab.groupby(["farm", "day"]).ph_in_temp_3.min()
    cd = dmin[dmin < THRESH]
    fd = {f: set(int(d) for (ff, d) in cd.index if ff == f) for f in TARGET_FARMS}
    trm, vam = split_mask(lab, fd)
    tr, va = lab[trm], lab[vam].reset_index(drop=True)
    yv = va.sub_temp.values
    ok = clean[vam]

    A = temp_members(tr, va, ct, phc)
    B = temp_members(tr, va, ct, phc, W[trm])
    imp = SimpleImputer(strategy="median").fit(tr[phc])
    base_a = LinearRegression().fit(imp.transform(tr[phc]), tr.sub_temp.values)
    base_b = LinearRegression().fit(imp.transform(tr[phc]), tr.sub_temp.values, sample_weight=W[trm])
    A["baseline only"] = base_a.predict(imp.transform(va[phc]))
    B["baseline only"] = base_b.predict(imp.transform(va[phc]))

    print("EXTRAP fold, clean held-out rows: %d" % int(ok.sum()))
    print("%-16s %9s %9s %8s" % ("member", "plain", "downwt", "change"))
    for k in ("baseline only", "res", "ridge", "nys"):
        a, b = rmse(A[k][ok], yv[ok]), rmse(B[k][ok], yv[ok])
        print("%-16s %9.4f %9.4f %+7.1f%%" % (k, a, b, 100 * (b / a - 1)))
    bl = 0.65 * A["res"] + 0.25 * A["ridge"] + 0.10 * A["nys"]
    bw = 0.65 * B["res"] + 0.25 * B["ridge"] + 0.10 * B["nys"]
    print("%-16s %9.4f %9.4f %+7.1f%%" % ("blend", rmse(bl[ok], yv[ok]), rmse(bw[ok], yv[ok]),
                                         100 * (rmse(bw[ok], yv[ok]) / rmse(bl[ok], yv[ok]) - 1)))
    # swap one member at a time to attribute the blend gain
    print("\nblend gain attribution (swap one member to its down-weighted version):")
    for k in ("res", "ridge", "nys"):
        mix = {kk: (B[kk] if kk == k else A[kk]) for kk in ("res", "ridge", "nys")}
        p = 0.65 * mix["res"] + 0.25 * mix["ridge"] + 0.10 * mix["nys"]
        print("  only %-6s down-weighted: %.4f" % (k, rmse(p[ok], yv[ok])))

    coef = pd.DataFrame({"plain": base_a.coef_, "downwt": base_b.coef_}, index=phc)
    coef["change"] = coef.downwt - coef.plain
    print("\nlinear physics baseline coefficients (largest changes):")
    print(coef.reindex(coef.change.abs().sort_values(ascending=False).index).head(6).round(3).to_string())


if __name__ == "__main__":
    main()
