# -*- coding: utf-8 -*-
"""Is v5's EC win from its feature set, or from leaf=2?  Cross the two factors.

If 68 sensor-grounded features at leaf=2 match v5, we can keep the model that
reasons from greenhouse state instead of the one that leans 39% on crop stage.
"""
import numpy as np

from common import USABLE, split_mask, rmse
from model_v2 import get_panel
from make_submission import DOMAIN_MARKS
from geometry_cv import geometry_folds, score, et_fp
import features_v2 as F2


def main():
    panel, tX, ty, sX = get_panel()
    panel["midnight"] = (panel.hour == 0).astype(float)
    folds = geometry_folds()
    lab = panel[(~panel.is_test) & panel.sub_ec.notna()].reset_index(drop=True)

    hist = [c for c in F2.view(panel, "sub_ec") if not any(m in c for m in DOMAIN_MARKS)]
    v5 = USABLE + ["day", "hr_sin", "hr_cos", "midnight"]
    hist_noday = [c for c in hist if c not in ("day", "day_par")]

    sets = [("v5 18개 (원본+달력)", v5),
            ("이력 68개", hist),
            ("이력 68개, day 제거", hist_noday)]
    print("%-26s %8s %8s %8s %8s" % ("피처셋", "leaf=2", "leaf=4", "leaf=8", "leaf=16"))
    for name, cols in sets:
        row = []
        for leaf in (2, 4, 8, 16):
            p = dict(n_estimators=100, max_features=1.0, min_samples_leaf=leaf, n_jobs=4)
            row.append(score(lab, folds, "sub_ec", et_fp(cols, p))[0])
        print("%-26s %8.4f %8.4f %8.4f %8.4f" % ((name,) + tuple(row)))

    print("\n원본 14개만 (달력 전부 제거):")
    for leaf in (2, 4, 8):
        p = dict(n_estimators=100, max_features=1.0, min_samples_leaf=leaf, n_jobs=4)
        print("   leaf=%-2d  %.4f" % (leaf, score(lab, folds, "sub_ec", et_fp(USABLE, p))[0]))


if __name__ == "__main__":
    main()
