# -*- coding: utf-8 -*-
"""sub_temp: can the other 49 greenhouses help?

Why re-open a direction the handoff closed:
  * Only the indoor sensors are genuine measurements.  24-h vectors of the
    outdoor weather repeat on 70-93% of days (train AND test); in_temp,
    in_hum, in_co2 never repeat (restored.py).  The informative inputs are
    exactly the ones every greenhouse has.
  * The top permutation-importance feature is in_temp_ewm6 (+0.53).
  * The handoff's rejection used a CV that we now know understated the
    temperature gain of submission 2 by 2.3x (calib_cv.py).  All three CVs
    did get the SIGN right for sub_temp, so this script is read for
    direction, not magnitude.
  * Other farms' sub_temp labels are integer-rounded: noise std 1/sqrt(12)
    = 0.289, below the current error (0.667 real, ~0.8 CV).

Design: features from features_v2 built on all 51 greenhouses, restricted to
columns derivable from the indoor sensors + clock.  Held-out rows are always
F13/F47 geometry-fold days; other farms are always training rows (never
scored).  Compare target-only vs pooled training on identical features, then
test the pooled model as a blend partner for the 93-column submission model.

Run:  cd research && PYTHONPATH="" <python> -u pool_temp.py
"""
import env  # noqa: F401
import numpy as np
import pandas as pd
import lightgbm as lgb

import common
import features_v2 as F2
from common import split_mask, rmse, TARGET_FARMS
from harness import load, views, folds
import feat_temp74 as T74
import feat_new
from feat_lib import paired_block_boot

DET = dict(deterministic=True, force_col_wise=True, n_jobs=4, verbose=-1)
T_HUB = dict(objective="huber", n_estimators=1200, learning_rate=0.03,
             num_leaves=63, min_child_samples=40, subsample=0.8,
             subsample_freq=1, colsample_bytree=0.6, reg_lambda=1.0)
POOL = dict(objective="huber", n_estimators=1500, learning_rate=0.03,
            num_leaves=127, min_child_samples=100, subsample=0.7,
            subsample_freq=1, colsample_bytree=0.7, reg_lambda=1.0)
INDOOR = ("in_temp", "in_hum", "in_co2", "vpd_in")
CLOCK = ["hour", "hr_sin", "hr_cos", "day"]


def indoor_cols(panel):
    return sorted([c for c in panel.columns
                   if c.startswith(INDOOR) and panel[c].dtype.kind in "fi"]) + CLOCK


def main():
    tX, ty, sX = common.load_raw()
    farms = sorted(tX.farm.unique())
    print("building features_v2 on %d greenhouses ..." % len(farms))
    big = F2.build(tX, sX, farms=farms)
    big = big.merge(ty[["row_id", "sub_temp"]], on="row_id", how="left")
    big["is_test"] = big.row_id.isin(set(sX.row_id))
    big["farm_cat"] = big.farm.str[1:].astype(int)
    lab_all = big[(~big.is_test) & big.sub_temp.notna()].reset_index(drop=True)
    ic = indoor_cols(big)
    print("labelled rows: all %d | target %d | indoor feature cols %d"
          % (len(lab_all), int(lab_all.farm.isin(TARGET_FARMS).sum()), len(ic)))

    # submission-2 LightGBM member on the target panel, for the blend test
    panel, lab_t0, _ = load()
    v = views(panel)
    ex, blocks = feat_new.build_extra()
    lab_t = lab_t0.merge(ex, on="row_id", how="left")
    f93 = T74.base74(v["temp"]) + list(blocks["dew"]) + list(blocks["event"])

    is_tgt = lab_all.farm.isin(TARGET_FARMS).values
    tgt_idx = np.where(is_tgt)[0]
    tgt = lab_all.iloc[tgt_idx].reset_index(drop=True)
    y_t = tgt.sub_temp.values

    res = {}
    for kind in ("A", "B"):
        fds = folds(kind)
        o_tonly = np.full(len(tgt), np.nan)
        o_pool = np.full(len(tgt), np.nan)
        o_poolw = np.full(len(tgt), np.nan)
        for k, fd in enumerate(fds):
            trm, _ = split_mask(lab_all, fd)
            _, vam_t = split_mask(tgt, fd)
            va = tgt[vam_t]
            tr_all = lab_all[trm]
            tr_tgt = tr_all[tr_all.farm.isin(TARGET_FARMS)]
            m = lgb.LGBMRegressor(random_state=7, **DET, **T_HUB)
            m.fit(tr_tgt[ic], tr_tgt.sub_temp)
            o_tonly[vam_t] = m.predict(va[ic])
            cols = ic + ["farm_cat"]
            m = lgb.LGBMRegressor(random_state=7, **DET, **POOL)
            m.fit(tr_all[cols], tr_all.sub_temp, categorical_feature=["farm_cat"])
            o_pool[vam_t] = m.predict(va[cols])
            w = np.where(tr_all.farm.isin(TARGET_FARMS), 10.0, 1.0)
            m = lgb.LGBMRegressor(random_state=7, **DET, **POOL)
            m.fit(tr_all[cols], tr_all.sub_temp, sample_weight=w,
                  categorical_feature=["farm_cat"])
            o_poolw[vam_t] = m.predict(va[cols])
            print("  [%s] fold %d done" % (kind, k), flush=True)
        g = ~np.isnan(o_tonly)
        print("\n######## 배치 %s (실내센서+시계 %d열) ########" % (kind, len(ic)))
        print("  대상 2온실만 학습        %.4f" % rmse(o_tonly[g], y_t[g]))
        print("  51온실 통합              %.4f" % rmse(o_pool[g], y_t[g]))
        print("  51온실 통합 + 대상 x10   %.4f" % rmse(o_poolw[g], y_t[g]))
        res[kind] = (o_tonly, o_pool, o_poolw)

    # blend test against the 93-col submission member (align by row_id)
    print("\n######## 제출 모델(93열 LGB) 과의 블렌드 ########")
    from harness import score
    for kind in ("A", "B"):
        (r, _, _), o93 = score(lab_t, "sub_temp", f93,
                               lambda s: lgb.LGBMRegressor(random_state=s, **DET, **T_HUB),
                               kind=kind, seeds=(7,), return_oof=True)
        key = pd.Series(np.arange(len(tgt)), index=tgt.row_id.values)
        pos = key.reindex(lab_t.row_id.values).values
        ok = ~np.isnan(pos)
        pool_on_t = np.full(len(lab_t), np.nan)
        pool_on_t[ok] = res[kind][2][pos[ok].astype(int)]
        g = ~np.isnan(o93) & ~np.isnan(pool_on_t)
        y = lab_t.sub_temp.values
        print("  배치 %s: 93열 단독 %.4f" % (kind, rmse(o93[g], y[g])))
        idx = [np.where(split_mask(lab_t, fd)[1])[0] for fd in folds(kind)]
        for w in (0.2, 0.3, 0.4, 0.5):
            b = (1 - w) * o93 + w * pool_on_t
            d = [rmse(b[i], y[i]) - rmse(o93[i], y[i]) for i in idx]
            sub = lab_t[g].reset_index(drop=True)
            pr, lo, hi, pw = paired_block_boot(sub, "sub_temp", o93[g], b[g],
                                               n_boot=2000, seed=0, level="row")
            print("    +통합 w=%.1f  %.4f | 폴드 %d/5 | %+.4f CI [%+.4f,%+.4f] P(worse)=%.3f"
                  % (w, rmse(b[g], y[g]), sum(x < 0 for x in d), pr, lo, hi, pw))


if __name__ == "__main__":
    main()
