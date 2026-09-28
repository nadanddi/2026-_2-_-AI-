# -*- coding: utf-8 -*-
"""sub_temp: do the two confirmed gains stack?

They were measured on different bases and never combined:
  * feature side  : drop prev_day+mem_long (98 -> 74), then +dew+event (93)
                    A 0.8310 / B 0.7723   (feat_temp74.py confirm)
  * model side    : LGB 0.65 / Ridge 0.25 / Nystroem 0.10 on the 98-col view
                    A 0.8439 / B 0.7733   (model_temp_paired.py)
Ridge and Nystroem are smooth global fits, so removing 24 noisy columns may
change how much they contribute.  Additivity cannot be assumed; measure it.

Run:  cd research && PYTHONPATH="" <python> -u temp_combine.py
"""
import env  # noqa: F401  MUST be first project import
import numpy as np
import lightgbm as lgb
from sklearn.impute import SimpleImputer
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from harness import load, score, views, folds, blend_factory
from common import split_mask, rmse
from feat_lib import paired_block_boot
import feat_temp74 as T74
import feat_new

DET = dict(deterministic=True, force_col_wise=True, n_jobs=4, verbose=-1)
T_HUB = dict(objective="huber", n_estimators=1200, learning_rate=0.03,
             num_leaves=63, min_child_samples=40, subsample=0.8,
             subsample_freq=1, colsample_bytree=0.6, reg_lambda=1.0)
SEEDS = (7, 101, 2024)


def LGBH(s):
    return lgb.LGBMRegressor(random_state=s, **DET, **T_HUB)


def RIDGE(s):
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         Ridge(alpha=100.0))


def NYS(s):
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         Nystroem(gamma=0.005, n_components=500, random_state=s),
                         Ridge(alpha=1.0))


def main():
    panel, lab_t0, lab_e = load()
    v = views(panel)
    cols98 = v["temp"]
    cols74 = T74.base74(cols98)
    ex, blocks = feat_new.build_extra()
    lab_t = lab_t0.merge(ex, on="row_id", how="left")
    cols93 = cols74 + list(blocks["dew"]) + list(blocks["event"])
    print("columns: 98f=%d  74f=%d  93f=%d" % (len(cols98), len(cols74), len(cols93)))

    W = [0.65, 0.25, 0.10]
    cands = [
        ("98f LGB alone (current submission)", cols98, LGBH),
        ("93f LGB alone", cols93, LGBH),
        ("98f 3-way .65/.25/.10", cols98, blend_factory([LGBH, RIDGE, NYS], W)),
        ("93f 3-way .65/.25/.10", cols93, blend_factory([LGBH, RIDGE, NYS], W)),
    ]
    oof = {}
    print("\n%-38s %8s %8s" % ("candidate", "A", "B"))
    print("-" * 58)
    for nm, cols, fac in cands:
        row = []
        for kind in ("A", "B"):
            (r, sd, per), o = score(lab_t, "sub_temp", cols, fac, kind=kind,
                                    seeds=SEEDS, return_oof=True)
            oof[(nm, kind)] = o
            row.append(r)
        print("%-38s %8.4f %8.4f" % (nm, row[0], row[1]))

    ref = "98f LGB alone (current submission)"
    y = lab_t["sub_temp"].values.astype(float)
    for nm, _, _ in cands[1:]:
        print("\n== paired: %s vs %s ==" % (ref, nm))
        for kind in ("A", "B"):
            a, b = oof[(ref, kind)], oof[(nm, kind)]
            g = ~np.isnan(a) & ~np.isnan(b)
            idx = [np.where(split_mask(lab_t, fd)[1])[0] for fd in folds(kind)]
            d = [rmse(b[i], y[i]) - rmse(a[i], y[i]) for i in idx]
            nb = sum(1 for x in d if x < 0)
            sub = lab_t[g].reset_index(drop=True)
            pr, lo, hi, pw = paired_block_boot(sub, "sub_temp", a[g], b[g],
                                               n_boot=2000, seed=0, level="row")
            print("  배치 %s | per-fold %s (%d/5) | row %+.4f CI [%+.4f,%+.4f] P(worse)=%.3f"
                  % (kind, " ".join("%+.3f" % x for x in d), nb, pr, lo, hi, pw))


if __name__ == "__main__":
    main()
