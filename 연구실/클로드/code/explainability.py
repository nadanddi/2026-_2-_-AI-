# -*- coding: utf-8 -*-
"""Is the model reasoning, or looking up nearby days?

Test 1 (day-jitter): at PREDICTION time only, shift `day` by +-1..3.  A model
that interpolates a slow seasonal drift barely moves; a model that memorises
"day 143 had EC 0.61" moves a lot.  Nothing about the truth changes, so any
change is the model keying on the date as an index rather than as crop stage.

Test 2 (grouped gain importance): which physical quantities carry the model.

Test 3 (monotonicity): does substrate temperature respond to air temperature
in the direction physics requires?
"""
import collections
import re

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

from common import rmse, split_mask, TARGET_FARMS, USABLE
from model_v2 import get_panel
from make_submission import T_HUB, E_HUB, ET, V5_ET, DOMAIN_MARKS
from geometry_cv import geometry_folds
import features_v2 as F2

SEED = 7


def fit_et(tr, cols, params, target="sub_ec"):
    m = make_pipeline(SimpleImputer(strategy="median"),
                      ExtraTreesRegressor(random_state=SEED, **params))
    m.fit(tr[cols], tr[target])
    m.steps[-1][1].n_jobs = 1
    return m


def fit_lgb(tr, cols, params, target):
    m = lgb.LGBMRegressor(random_state=SEED, n_jobs=4, verbose=-1,
                          deterministic=True, force_col_wise=True, **params)
    m.fit(tr[cols], tr[target])
    return m


def jitter_test(lab, folds, cols, fitter, target, name):
    """RMSE, and how far predictions move when `day` is shifted at predict time."""
    base_err, moves = [], collections.defaultdict(list)
    for fd in folds:
        trm, vam = split_mask(lab, fd)
        tr, va = lab[trm], lab[vam]
        m = fitter(tr, cols)
        p0 = m.predict(va[cols])
        base_err.append(rmse(p0, va[target].values))
        for k in (1, 2, 3):
            for s in (-k, k):
                v2 = va.copy()
                v2["day"] = v2.day + s
                moves[k].append(np.abs(m.predict(v2[cols]) - p0).mean())
    sd = lab[target].std()
    print("  %-30s RMSE %.4f | 예측 이동폭:" % (name, np.mean(base_err)), end="")
    for k in (1, 2, 3):
        mv = np.mean(moves[k])
        print("  %+dd %.4f(%.0f%%)" % (k, mv, 100 * mv / sd), end="")
    print()


def grouped_importance(lab, folds, cols, fitter, target, name):
    tot = collections.Counter()
    for fd in folds:
        trm, _ = split_mask(lab, fd)
        m = fitter(lab[trm], cols)
        if hasattr(m, "booster_"):
            imp = m.booster_.feature_importance("gain")
        else:
            imp = m.steps[-1][1].feature_importances_
        for c, v in zip(cols, imp):
            tot[base_var(c)] += float(v)
    s = pd.Series(tot)
    s = (100 * s / s.sum()).sort_values(ascending=False)
    print("\n  [%s] 변수별 기여도 %%" % name)
    for k, v in s.head(10).items():
        print("     %5.1f%%  %s" % (v, k))


def base_var(c):
    if c in ("day", "day_par"):
        return "day (작기 일차)"
    if c in ("hour", "hr_sin", "hr_cos", "midnight"):
        return "hour (시각)"
    if c in ("farm_id", "farm_code"):
        return "farm (온실 구분)"
    c = re.sub(r"_(ewm|lag|r|dev)\d+[ms]?$", "", c)
    c = re.sub(r"_(d1|d3|d24|dev24|dev6)$", "", c)
    c = re.sub(r"_(dm|dmax|dmin)?_?pd\d?(_7m|7m)?$", "", c)
    c = re.sub(r"_(tdmean|tdsum|cum|cumrate|todaymean|todaysum)$", "", c)
    c = re.sub(r"_(dm|dmax|dmin)$", "", c)
    return c


def main():
    panel, tX, ty, sX = get_panel()
    panel["midnight"] = (panel.hour == 0).astype(float)
    folds = geometry_folds()
    v_temp = F2.view(panel, "sub_temp")
    v_ec_cur = [c for c in F2.view(panel, "sub_ec") if not any(m in c for m in DOMAIN_MARKS)]
    v5 = USABLE + ["day", "hr_sin", "hr_cos", "midnight"]
    lab_t = panel[(~panel.is_test) & panel.sub_temp.notna()].reset_index(drop=True)
    lab_e = panel[(~panel.is_test) & panel.sub_ec.notna()].reset_index(drop=True)

    print("=== 테스트 1. day 흔들기 (정답은 그대로, 예측만 얼마나 움직이나) ===")
    print("  이동폭이 크면 = 날짜를 '색인'으로 외운 것. %는 타깃 표준편차 대비.\n")
    jitter_test(lab_e, folds, v5, lambda tr, c: fit_et(tr, c, V5_ET), "sub_ec",
                "EC 2회차 v5 (ET leaf=2, 18개)")
    jitter_test(lab_e, folds, v_ec_cur, lambda tr, c: fit_et(tr, c, ET), "sub_ec",
                "EC 1회차 ET (leaf=8, 68개)")
    jitter_test(lab_e, folds, v_ec_cur, lambda tr, c: fit_lgb(tr, c, E_HUB, "sub_ec"),
                "sub_ec", "EC 1회차 LGB (68개)")
    jitter_test(lab_t, folds, v_temp, lambda tr, c: fit_lgb(tr, c, T_HUB, "sub_temp"),
                "sub_temp", "온도 제출 구성 (98개)")

    print("\n=== 테스트 2. 무엇이 모델을 움직이나 ===")
    grouped_importance(lab_t, folds, v_temp,
                       lambda tr, c: fit_lgb(tr, c, T_HUB, "sub_temp"), "sub_temp",
                       "온도 제출 구성")
    grouped_importance(lab_e, folds, v5, lambda tr, c: fit_et(tr, c, V5_ET), "sub_ec",
                       "EC 2회차 v5")
    grouped_importance(lab_e, folds, v_ec_cur, lambda tr, c: fit_et(tr, c, ET), "sub_ec",
                       "EC 1회차 ET")

    print("\n=== 테스트 3. 물리 방향성: 실내온도를 ±2℃ 옮기면 배지온도 예측은? ===")
    fd = folds[0]
    trm, vam = split_mask(lab_t, fd)
    m = fit_lgb(lab_t[trm], v_temp, T_HUB, "sub_temp")
    va = lab_t[vam]
    p0 = m.predict(va[v_temp])
    for d in (-2, -1, 1, 2):
        v2 = va.copy()
        for c in v_temp:
            if c.startswith("in_temp"):
                v2[c] = v2[c] + d
        print("     실내온도 %+d℃ → 배지온도 예측 %+.3f℃ (일관 방향 %.0f%%)"
              % (d, (m.predict(v2[v_temp]) - p0).mean(),
                 100 * np.mean(np.sign(m.predict(v2[v_temp]) - p0) == np.sign(d))))


if __name__ == "__main__":
    main()
