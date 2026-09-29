# -*- coding: utf-8 -*-
"""Verify the source-fingerprint features on the harness folds.

The time-structure agent reported temp 0.794->0.767 and EC 0.257->0.226 on its
own GroupKFold.  Re-measure on the geometry folds A/B with paired tests, on
our actual configurations, and audit causality first.

Run:  cd research && PYTHONPATH="" <python> -u verify_fp.py
"""
import env  # noqa: F401
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

import common
from common import split_mask, rmse, USABLE, OUT_COLS, TARGET_FARMS
from harness import load, views, score, folds
import feat_temp74 as T74
import feat_new
import fp_features as FP
from feat_lib import paired_block_boot

DET = dict(deterministic=True, force_col_wise=True, n_jobs=4, verbose=-1)
T_HUB = dict(objective="huber", n_estimators=1200, learning_rate=0.03,
             num_leaves=63, min_child_samples=40, subsample=0.8,
             subsample_freq=1, colsample_bytree=0.6, reg_lambda=1.0)
ET1 = dict(n_estimators=600, max_features=1.0, min_samples_leaf=1, n_jobs=4)
LGBP = dict(n_estimators=800, learning_rate=0.03, num_leaves=31,
            min_child_samples=40, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.8, reg_lambda=1.0)


def causality_audit():
    base = FP.build().set_index("row_id").sort_index()
    orig = common.load_raw
    tX, ty, sX = orig()
    ok = True
    for farm in TARGET_FARMS:
        ts = np.sort(pd.concat([tX, sX]).query("farm == @farm").t.unique())
        T = int(ts[len(ts) * 2 // 3]) + 7          # cut mid-day, not at midnight
        def loader(farm=farm, T=T):
            a, b, c = orig()
            rng = np.random.RandomState(0)
            for df in (a, c):
                m = ((df.farm == farm) & (df.t > T)).values
                for col in USABLE:
                    df.loc[m, col] = df.loc[m, col] + rng.normal(0, 5.0, int(m.sum()))
                df.loc[m & (rng.rand(len(df)) < 0.33), USABLE] = np.nan
            return a, b, c
        common.load_raw = loader
        try:
            alt = FP.build().set_index("row_id").sort_index()
        finally:
            common.load_raw = orig
        rid = pd.concat([tX, sX]).set_index("row_id")
        past = [r for r in base.index if rid.loc[r, "farm"] == farm and rid.loc[r, "t"] <= T]
        a, b = base.loc[past], alt.loc[past]
        same = (a.values == b.values) | (a.isna().values & b.isna().values)
        bad = int((~same).sum())
        print("  causality %s cut t=%d (mid-day): %s" % (farm, T, "PASS" if bad == 0 else "FAIL %d" % bad))
        ok &= bad == 0
    return ok


def paired(lab, target, a, b, kind, tag):
    y = lab[target].values
    g = ~np.isnan(a) & ~np.isnan(b)
    idx = [np.where(split_mask(lab, fd)[1])[0] for fd in folds(kind)]
    d = [rmse(b[i], y[i]) - rmse(a[i], y[i]) for i in idx]
    sub = lab[g].reset_index(drop=True)
    pr, lo, hi, pw = paired_block_boot(sub, target, a[g], b[g], n_boot=2000, seed=0, level="row")
    print("  %-28s %s  %.4f -> %.4f | 폴드 %d/5 | %+.4f CI [%+.4f,%+.4f] P(worse)=%.3f"
          % (tag, kind, rmse(a[g], y[g]), rmse(b[g], y[g]), sum(x < 0 for x in d),
             pr, lo, hi, pw), flush=True)


def main():
    print("== 1. 인과성 검사 ==")
    causality_audit()

    panel, lab_t0, lab_e0 = load()
    v = views(panel)
    fp = FP.build()
    fpc = FP.names(fp)
    ex, blocks = feat_new.build_extra()
    lab_e = lab_e0.merge(fp, on="row_id", how="left")
    lab_t = lab_t0.merge(ex, on="row_id", how="left").merge(fp, on="row_id", how="left")
    f14 = [c for c in (list(USABLE) + ["day", "hr_sin", "hr_cos", "midnight"]) if c not in OUT_COLS]
    f93 = T74.base74(v["temp"]) + list(blocks["dew"]) + list(blocks["event"])
    print("fingerprint cols: %d" % len(fpc))

    et = lambda s: make_pipeline(SimpleImputer(strategy="median"), ExtraTreesRegressor(random_state=s, **ET1))
    ltw = lambda s: lgb.LGBMRegressor(random_state=s, objective="tweedie",
                                      tweedie_variance_power=1.5, **DET, **LGBP)
    lgbh = lambda s: lgb.LGBMRegressor(random_state=s, **DET, **T_HUB)

    print("\n== 2. EC (14열 기준) ==")
    for nm, fac in (("ET600/1", et), ("LGB tweedie", ltw)):
        for kind in ("A", "B"):
            _, a = score(lab_e, "sub_ec", f14, fac, kind=kind, seeds=(7,), return_oof=True)
            _, b = score(lab_e, "sub_ec", f14 + fpc, fac, kind=kind, seeds=(7,), return_oof=True)
            paired(lab_e, "sub_ec", a, b, kind, nm + " +지문")

    print("\n== 3. 온도 (93열 LGB 기준) ==")
    for kind in ("A", "B"):
        _, a = score(lab_t, "sub_temp", f93, lgbh, kind=kind, seeds=(7,), return_oof=True)
        _, b = score(lab_t, "sub_temp", f93 + fpc, lgbh, kind=kind, seeds=(7,), return_oof=True)
        paired(lab_t, "sub_temp", a, b, kind, "LGB93 +지문")


if __name__ == "__main__":
    main()
