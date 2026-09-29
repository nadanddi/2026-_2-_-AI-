# -*- coding: utf-8 -*-
"""Candidate submission 3: stack everything verified so far and test it whole.

Verified pieces (paired, harness folds A/B):
  * source fingerprint features (fp_features.py): EC ET -9.4%/-4.3%,
    temp LGB -1.5%/-1.4%, causality PASS
  * temp: day-restarted filters (seg18) + physics-baseline residual target,
    resid blend 0.8068/0.7583 vs submission 2 0.8211/0.7718 (cold_temp.py)
Gains measured separately need not add up, so the combinations are measured
directly against the submission-2 configuration of each target.

Run:  cd research && PYTHONPATH="" <python> -u combine_v4.py
"""
import env  # noqa: F401
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import common
from common import split_mask, rmse, USABLE, OUT_COLS
from harness import load, views, folds
import feat_temp74 as T74
import feat_new
import fp_features as FP
from struct_segment import seg_features
from struct_temp import series, build as phys_build
from cold_temp import PHYS
from make_submission_v3 import causal_shrink
from feat_lib import paired_block_boot

SEED = 7
DET = dict(deterministic=True, force_col_wise=True, n_jobs=4, verbose=-1)
T_HUB = dict(objective="huber", n_estimators=1200, learning_rate=0.03,
             num_leaves=63, min_child_samples=40, subsample=0.8,
             subsample_freq=1, colsample_bytree=0.6, reg_lambda=1.0)
ET1 = dict(n_estimators=600, max_features=1.0, min_samples_leaf=1, n_jobs=4)
LGBP = dict(n_estimators=800, learning_rate=0.03, num_leaves=31,
            min_child_samples=40, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.8, reg_lambda=1.0)


def lgbh():
    return lgb.LGBMRegressor(random_state=SEED, **DET, **T_HUB)


def ridge():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=100.0))


def nys():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         Nystroem(gamma=0.005, n_components=500, random_state=SEED), Ridge(alpha=1.0))


def et():
    return make_pipeline(SimpleImputer(strategy="median"), ExtraTreesRegressor(random_state=SEED, **ET1))


def ltw():
    return lgb.LGBMRegressor(random_state=SEED, objective="tweedie", tweedie_variance_power=1.5, **DET, **LGBP)


def mlp():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         MLPRegressor(hidden_layer_sizes=(128, 64), alpha=1e-2, learning_rate_init=1e-3,
                                      max_iter=800, early_stopping=True, n_iter_no_change=25,
                                      validation_fraction=0.12, random_state=SEED))


def fp_(m, cols, tr, va, y):
    return m.fit(tr[cols], y).predict(va[cols])


def temp_blend(tr, va, cols):
    y = tr.sub_temp.values
    return 0.65 * fp_(lgbh(), cols, tr, va, y) + 0.25 * fp_(ridge(), cols, tr, va, y) \
        + 0.10 * fp_(nys(), cols, tr, va, y)


def temp_resid(tr, va, cols, phc):
    imp = SimpleImputer(strategy="median").fit(tr[phc])
    base = LinearRegression().fit(imp.transform(tr[phc]), tr.sub_temp.values)
    b_tr, b_va = base.predict(imp.transform(tr[phc])), base.predict(imp.transform(va[phc]))
    y = tr.sub_temp.values
    r = fp_(lgbh(), cols, tr, va, y - b_tr)
    return 0.65 * (b_va + r) + 0.25 * fp_(ridge(), cols, tr, va, y) + 0.10 * fp_(nys(), cols, tr, va, y)


def ec_blend(tr, va, c_et, c_rest):
    y = tr.sub_ec.values
    p = 0.60 * fp_(et(), c_et, tr, va, y) + 0.30 * fp_(ltw(), c_rest, tr, va, y) \
        + 0.10 * fp_(mlp(), c_rest, tr, va, y)
    return np.clip(causal_shrink(p, va, 0.5), 0.062, 3.46)


def cv(lab, fn, kind):
    o = np.full(len(lab), np.nan)
    for fd in folds(kind):
        trm, vam = split_mask(lab, fd)
        o[np.where(vam)[0]] = fn(lab[trm], lab[vam].reset_index(drop=True))
    return o


def report(lab, target, ref, alt, kind, tag):
    y = lab[target].values
    g = ~np.isnan(ref) & ~np.isnan(alt)
    idx = [np.where(split_mask(lab, fd)[1])[0] for fd in folds(kind)]
    d = [rmse(alt[i], y[i]) - rmse(ref[i], y[i]) for i in idx]
    sub = lab[g].reset_index(drop=True)
    pr, lo, hi, pw = paired_block_boot(sub, target, ref[g], alt[g], n_boot=2000, seed=0, level="row")
    print("  %-30s %s %.4f -> %.4f (%+.1f%%) | 폴드 %d/5 | CI [%+.4f,%+.4f] P(worse)=%.3f"
          % (tag, kind, rmse(ref[g], y[g]), rmse(alt[g], y[g]),
             100 * (rmse(alt[g], y[g]) / rmse(ref[g], y[g]) - 1), sum(x < 0 for x in d), lo, hi, pw),
          flush=True)


def main():
    panel, lab_t0, lab_e0 = load()
    v = views(panel)
    ex, blocks = feat_new.build_extra()
    sg = seg_features()
    fp = FP.build()
    fpc = FP.names(fp)
    tX, ty, sX = common.load_raw()
    ph = phys_build(series(tX, sX), PHYS)
    phc0 = [c for c in ph.columns if c != "row_id"]
    ph = ph.rename(columns={c: "ph_" + c for c in phc0})
    phc = ["ph_" + c for c in phc0]
    lab_t = (lab_t0.merge(ex, on="row_id", how="left").merge(sg, on="row_id", how="left")
                   .merge(fp, on="row_id", how="left").merge(ph, on="row_id", how="left"))
    lab_e = lab_e0.merge(fp, on="row_id", how="left")
    f93 = T74.base74(v["temp"]) + list(blocks["dew"]) + list(blocks["event"])
    seg = [c for c in sg.columns if c != "row_id"]
    f14 = [c for c in (list(USABLE) + ["day", "hr_sin", "hr_cos", "midnight"]) if c not in OUT_COLS]

    print("== 배지 온도 (기준: 2회차 구성) ==")
    for kind in ("A", "B"):
        ref = cv(lab_t, lambda tr, va: temp_blend(tr, va, f93), kind)
        for tag, fn in (
            ("잔차블렌드 +seg18", lambda tr, va: temp_resid(tr, va, f93 + seg, phc)),
            ("잔차블렌드 +seg18 +지문", lambda tr, va: temp_resid(tr, va, f93 + seg + fpc, phc)),
            ("2회차블렌드 +지문", lambda tr, va: temp_blend(tr, va, f93 + fpc)),
        ):
            report(lab_t, "sub_temp", ref, cv(lab_t, fn, kind), kind, tag)

    print("\n== 배지 EC (기준: 2회차 구성) ==")
    for kind in ("A", "B"):
        ref = cv(lab_e, lambda tr, va: ec_blend(tr, va, f14, f14), kind)
        for tag, fn in (
            ("지문 ET에만", lambda tr, va: ec_blend(tr, va, f14 + fpc, f14)),
            ("지문 전 구성원", lambda tr, va: ec_blend(tr, va, f14 + fpc, f14 + fpc)),
        ):
            report(lab_e, "sub_ec", ref, cv(lab_e, fn, kind), kind, tag)


if __name__ == "__main__":
    main()
