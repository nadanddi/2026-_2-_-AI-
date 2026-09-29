# -*- coding: utf-8 -*-
"""Same direction check on BOTH EC models, plus the confounding it may reflect.

If the 68-feature model behaves the same way, the "seasonal proxy instead of
mechanism" finding is a property of the data, not of the model we picked.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

from common import USABLE, split_mask, TARGET_FARMS
from model_v2 import get_panel
from make_submission import V5_ET, ET, E_HUB, DOMAIN_MARKS
from geometry_cv import geometry_folds
import features_v2 as F2

VARS = ["out_temp", "out_rad", "out_hum", "act_vent", "act_circfan"]


def pdp_delta(model, X, col, lo, hi):
    a, b = X.copy(), X.copy()
    a[col] = lo
    b[col] = hi
    return model.predict(b).mean() - model.predict(a).mean()


def main():
    panel, tX, ty, sX = get_panel()
    panel["midnight"] = (panel.hour == 0).astype(float)
    lab = panel[(~panel.is_test) & panel.sub_ec.notna()].reset_index(drop=True)
    trm, vam = split_mask(lab, geometry_folds()[0])
    tr, va = lab[trm], lab[vam]

    v5 = USABLE + ["day", "hr_sin", "hr_cos", "midnight"]
    hist = [c for c in F2.view(panel, "sub_ec") if not any(m in c for m in DOMAIN_MARKS)]

    models = {}
    m = make_pipeline(SimpleImputer(strategy="median"), ExtraTreesRegressor(random_state=7, **V5_ET))
    models["v5 ET 18개"] = (m.fit(tr[v5], tr.sub_ec), v5)
    m = make_pipeline(SimpleImputer(strategy="median"), ExtraTreesRegressor(random_state=7, **ET))
    models["1회차 ET 68개"] = (m.fit(tr[hist], tr.sub_ec), hist)
    g = lgb.LGBMRegressor(random_state=7, n_jobs=4, verbose=-1, deterministic=True,
                          force_col_wise=True, **E_HUB)
    models["1회차 LGB 68개"] = (g.fit(tr[hist], tr.sub_ec), hist)

    print("=== 5→95 백분위 이동 시 EC 예측 변화 (문헌 기대 방향 표기) ===")
    hdr = "%-12s %7s" % ("변수", "기대")
    for n in models:
        hdr += " %14s" % n
    print(hdr)
    expect = {"out_temp": "+", "out_rad": "+", "out_hum": "-",
              "act_vent": "?", "act_circfan": "?"}
    for c in VARS:
        lo, hi = np.percentile(lab[c].dropna(), [5, 95])
        row = "%-12s %7s" % (c, expect[c])
        for n, (mm, cols) in models.items():
            # the raw column may not exist in the derived view under its own name
            if c not in cols:
                row += " %14s" % "n/a"
                continue
            row += " %+14.3f" % pdp_delta(mm, va[cols].copy(), c, lo, hi)
        print(row)

    print("\n=== 계절 교란 확인: 각 변수와 day / sub_ec 의 상관 ===")
    d = lab[VARS + ["day", "sub_ec"]].copy()
    daily = d.groupby(lab.day).mean()
    print("%-12s %10s %10s" % ("변수", "corr(day)", "corr(EC)"))
    for c in VARS:
        print("%-12s %+10.3f %+10.3f"
              % (c, daily[c].corr(daily.day), daily[c].corr(daily.sub_ec)))

    print("\n=== 계절을 통제하면 관계가 남는가 (일별 평균을 day 추세로 제거 후) ===")
    res = daily.copy()
    for c in VARS + ["sub_ec"]:
        z = res[[c]].join(res.day).dropna()
        k = np.polyfit(z.day, z[c], 3)
        res[c] = res[c] - np.poly1d(k)(res.day)
    print("%-12s %10s" % ("변수", "부분상관"))
    for c in VARS:
        print("%-12s %+10.3f" % (c, res[c].corr(res.sub_ec)))


if __name__ == "__main__":
    main()
