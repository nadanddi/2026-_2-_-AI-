# -*- coding: utf-8 -*-
"""Is the high-EC under-prediction a SHRINKAGE problem or an INFORMATION one?

If trees under-predict y>1 only because a leaf average cannot reach beyond the
training range, then simply expanding the spread of the existing predictions
must reduce the bias, and there must exist an expansion factor that also
reduces RMSE.  This script measures the best achievable expansion / monotone
recalibration with the answer in hand (an oracle), which is an upper bound on
anything a legitimate post-processor could do.

  * variance inflation   p' = m + c (p - m)          grid over c
  * power warp           p' = m (p/m)^g              grid over g
  * isotonic             fitted on fold-set A's OOF, applied to fold-set B's
                         OOF and vice versa (honest cross-placement transfer)

A negative result here means the bias cannot be removed by any monotone
transform of the current predictions: the ordering itself is wrong, i.e. the
features do not identify the high-EC rows.  That would rule out the whole
"tree cannot extrapolate" explanation.

Run:  cd research && PYTHONPATH="" <python> -u model_calib_ec.py
"""
import env  # noqa: F401  MUST be first project import
import numpy as np
import lightgbm as lgb
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.pipeline import make_pipeline

from model_common import load, views, seg_stats, blend_factory
from harness import score
from common import rmse

DET = dict(deterministic=True, force_col_wise=True, n_jobs=4, verbose=-1)
E_HUB = dict(objective="huber", n_estimators=400, learning_rate=0.02,
             num_leaves=7, min_child_samples=240, subsample=0.7,
             subsample_freq=1, colsample_bytree=0.4, reg_lambda=5.0)
ET8 = dict(n_estimators=100, max_features=1.0, min_samples_leaf=8, n_jobs=4)


def main():
    panel, lab_t, lab_e = load()
    v = views(panel)
    fac = blend_factory([
        lambda s: make_pipeline(SimpleImputer(strategy="median"),
                                ExtraTreesRegressor(random_state=s, **ET8)),
        lambda s: lgb.LGBMRegressor(random_state=s, **DET, **E_HUB)])

    y = lab_e["sub_ec"].values.astype(float)
    oofs = {}
    for kind in ("A", "B"):
        (r, sd, per), oof = score(lab_e, "sub_ec", v["ec"], fac, kind=kind,
                                  return_oof=True)
        oofs[kind] = oof
        s = seg_stats(y[~np.isnan(oof)], oof[~np.isnan(oof)])
        print("submitted blend folds %s : RMSE %.4f (fold sd %.4f)  "
              "hi n=%d rmse %.4f bias %+.4f share %.3f"
              % (kind, r, sd, s["n"], s["rmse"], s["bias"], s["share"]))

    print("\n--- oracle variance inflation  p' = m + c(p-m) ---")
    print("%6s | %-26s | %-26s" % ("c", "folds A  rmse / hi-bias",
                                   "folds B  rmse / hi-bias"))
    for c in (1.0, 1.05, 1.1, 1.15, 1.2, 1.3, 1.4, 1.6, 2.0):
        line = "%6.2f |" % c
        for kind in ("A", "B"):
            oof = oofs[kind]
            g = ~np.isnan(oof)
            p, yy = oof[g], y[g]
            m = p.mean()
            q = m + c * (p - m)
            s = seg_stats(yy, q)
            line += " %8.4f / %+7.4f      |" % (rmse(q, yy), s["bias"])
        print(line)

    print("\n--- oracle power warp  p' = m (p/m)^g ---")
    for gpow in (1.0, 1.2, 1.4, 1.6, 2.0, 2.5):
        line = "%6.2f |" % gpow
        for kind in ("A", "B"):
            oof = oofs[kind]
            g = ~np.isnan(oof)
            p, yy = oof[g], y[g]
            m = p.mean()
            q = m * np.power(np.clip(p, 1e-6, None) / m, gpow)
            s = seg_stats(yy, q)
            line += " %8.4f / %+7.4f      |" % (rmse(q, yy), s["bias"])
        print(line)

    print("\n--- isotonic, fitted on the OTHER fold set (honest) ---")
    for src, dst in (("A", "B"), ("B", "A")):
        gs = ~np.isnan(oofs[src])
        gd = ~np.isnan(oofs[dst])
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(oofs[src][gs], y[gs])
        q = iso.predict(oofs[dst][gd])
        s0 = seg_stats(y[gd], oofs[dst][gd])
        s1 = seg_stats(y[gd], q)
        print("  fit %s -> apply %s : rmse %.4f -> %.4f | hi-bias %+.4f -> %+.4f"
              % (src, dst, rmse(oofs[dst][gd], y[gd]), rmse(q, y[gd]),
                 s0["bias"], s1["bias"]))

    print("\n--- ordering check: can the predictions even RANK the high rows? ---")
    for kind in ("A", "B"):
        oof = oofs[kind]
        g = ~np.isnan(oof)
        p, yy = oof[g], y[g]
        hi = yy > 1.0
        thr = np.quantile(p, 1 - hi.mean())
        flag = p > thr
        prec = float((hi & flag).sum()) / max(1, int(flag.sum()))
        print("  folds %s: spearman %.3f | top-%.1f%% of predictions capture "
              "%.1f%% of the true y>1 rows (precision %.2f)"
              % (kind,
                 float(np.corrcoef(np.argsort(np.argsort(p)),
                                   np.argsort(np.argsort(yy)))[0, 1]),
                 100 * hi.mean(),
                 100 * float((hi & flag).sum()) / max(1, int(hi.sum())), prec))
        # what would a perfectly-scaled model achieve on the high rows?
        a = float(np.sum(p[hi] * yy[hi]) / np.sum(p[hi] ** 2))
        print("        best single rescale of the hi rows alone: a=%.3f -> "
              "hi rmse %.4f (was %.4f)"
              % (a, rmse(a * p[hi], yy[hi]), rmse(p[hi], yy[hi])))


if __name__ == "__main__":
    main()
