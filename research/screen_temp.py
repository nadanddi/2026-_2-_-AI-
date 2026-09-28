# -*- coding: utf-8 -*-
"""sub_temp: screen candidate changes to the submission-2 temperature model.

Only 4 submissions remain for this team member, one of which must be kept to
restore the best file if an experiment scores worse.  So candidates are
screened here and only those that improve on BOTH geometry placements are
bundled.  Magnitudes are not trusted (the CV understated submission 2's real
temperature gain 2.3x); direction on both placements is the criterion.

Reference model: submission 2's temperature blend (93 cols, LGB .65 /
Ridge .25 / Nystroem .10), seed 7 for speed.

Candidates
  seg18    day-restarted filters (struct_segment.py: LGB alone -0.0026/-0.0069)
  stretch  range decompression p' = m + c (p - m); c fitted on the OTHER
           placement's OOF (honest), m = farm mean of training labels
  both     seg18 + stretch

Run:  cd research && PYTHONPATH="" <python> -u screen_temp.py
"""
import env  # noqa: F401
import numpy as np
import lightgbm as lgb
from sklearn.impute import SimpleImputer
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import split_mask, rmse
from harness import load, views, score, folds, blend_factory
import feat_temp74 as T74
import feat_new
from struct_segment import seg_features
from feat_lib import paired_block_boot

DET = dict(deterministic=True, force_col_wise=True, n_jobs=4, verbose=-1)
T_HUB = dict(objective="huber", n_estimators=1200, learning_rate=0.03,
             num_leaves=63, min_child_samples=40, subsample=0.8,
             subsample_freq=1, colsample_bytree=0.6, reg_lambda=1.0)
lgbh = lambda s: lgb.LGBMRegressor(random_state=s, **DET, **T_HUB)
ridge = lambda s: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                Ridge(alpha=100.0))
nys = lambda s: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                              Nystroem(gamma=0.005, n_components=500, random_state=s),
                              Ridge(alpha=1.0))
BLEND = blend_factory([lgbh, ridge, nys], [0.65, 0.25, 0.10])


def stretch(p, lab, c):
    m = lab.groupby("farm").sub_temp.transform("mean").values
    return m + c * (p - m)


def best_c(p, lab):
    g = ~np.isnan(p)
    cs = np.arange(0.95, 1.201, 0.01)
    r = [rmse(stretch(p, lab, c)[g], lab.sub_temp.values[g]) for c in cs]
    return float(cs[int(np.argmin(r))])


def main():
    panel, lab0, _ = load()
    v = views(panel)
    ex, blocks = feat_new.build_extra()
    sg = seg_features()
    lab = lab0.merge(ex, on="row_id", how="left").merge(sg, on="row_id", how="left")
    f93 = T74.base74(v["temp"]) + list(blocks["dew"]) + list(blocks["event"])
    seg = [c for c in sg.columns if c != "row_id"]
    y = lab.sub_temp.values

    oof = {}
    for nm, cols in (("base", f93), ("seg18", f93 + seg)):
        for kind in ("A", "B"):
            (r, _, _), o = score(lab, "sub_temp", cols, BLEND, kind=kind,
                                 seeds=(7,), return_oof=True)
            oof[(nm, kind)] = o
            print("  %-6s %s  %.4f" % (nm, kind, r), flush=True)

    # honest stretch: c from the other placement
    cA, cB = best_c(oof[("base", "A")], lab), best_c(oof[("base", "B")], lab)
    print("\n  stretch c fitted: on A %.2f | on B %.2f" % (cA, cB))
    for base_nm in ("base", "seg18"):
        for kind, c in (("A", cB), ("B", cA)):
            oof[(base_nm + "+stretch", kind)] = stretch(oof[(base_nm, kind)], lab, c)

    print("\n== paired vs submission-2 temperature blend ==")
    for nm in ("seg18", "base+stretch", "seg18+stretch"):
        for kind in ("A", "B"):
            a, b = oof[("base", kind)], oof[(nm, kind)]
            g = ~np.isnan(a) & ~np.isnan(b)
            idx = [np.where(split_mask(lab, fd)[1])[0] for fd in folds(kind)]
            d = [rmse(b[i], y[i]) - rmse(a[i], y[i]) for i in idx]
            sub = lab[g].reset_index(drop=True)
            pr, lo, hi, pw = paired_block_boot(sub, "sub_temp", a[g], b[g],
                                               n_boot=2000, seed=0, level="row")
            print("  %-14s %s  %.4f | 폴드 %d/5 | %+.4f CI [%+.4f,%+.4f] P(worse)=%.3f"
                  % (nm, kind, rmse(b[g], y[g]), sum(x < 0 for x in d),
                     pr, lo, hi, pw))


if __name__ == "__main__":
    main()
