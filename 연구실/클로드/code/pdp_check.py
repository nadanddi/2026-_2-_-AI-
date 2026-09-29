# -*- coding: utf-8 -*-
"""Does the EC model move in the direction the agronomy says it should?

Literature (RDA/nongsaro fertigation guidance, soilless-culture reviews):
  - fertigation EC setpoint is ramped up through the crop cycle, and unabsorbed
    salts accumulate  -> substrate EC should RISE with crop day;
  - transpiration concentrates the substrate solution, drainage dilutes it, and
    drainage grows when radiation is low / humidity high / temperature low
    -> higher evaporative demand should push EC UP.
Partial dependence is computed by sweeping one feature across the observed
range on real rows and averaging the prediction (other features untouched).
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

from common import USABLE, split_mask
from model_v2 import get_panel
from make_submission import V5_ET
from geometry_cv import geometry_folds


def pdp(model, X, col, grid):
    out = []
    for g in grid:
        Z = X.copy()
        Z[col] = g
        out.append(model.predict(Z).mean())
    return np.array(out)


def main():
    panel, tX, ty, sX = get_panel()
    panel["midnight"] = (panel.hour == 0).astype(float)
    v5 = USABLE + ["day", "hr_sin", "hr_cos", "midnight"]
    lab = panel[(~panel.is_test) & panel.sub_ec.notna()].reset_index(drop=True)

    trm, vam = split_mask(lab, geometry_folds()[0])
    tr, va = lab[trm], lab[vam]
    m = make_pipeline(SimpleImputer(strategy="median"),
                      ExtraTreesRegressor(random_state=7, **V5_ET))
    m.fit(tr[v5], tr.sub_ec)
    X = va[v5].copy()

    print("=== 작기일차(day)에 따른 EC 예측 — 문헌 예측: 상승 ===")
    grid = np.linspace(lab.day.min(), lab.day.max(), 10).round().astype(int)
    p = pdp(m, X, "day", grid)
    obs = lab.groupby(pd.cut(lab.day, 9)).sub_ec.mean()
    for g, v in zip(grid, p):
        print("   %3d일차  예측 %.3f" % (g, v))
    print("   → 기울기 %+.4f /일,  단조 상승 구간 %d/9"
          % (np.polyfit(grid, p, 1)[0], int((np.diff(p) > 0).sum())))
    print("   실측 일별평균 추세: %.3f → %.3f" % (obs.iloc[0], obs.iloc[-1]))

    print("\n=== 증산 동인에 따른 EC 예측 — 문헌 예측: 증산↑ → EC↑ ===")
    for col, unit in [("out_temp", "℃"), ("out_rad", "W/m2"), ("out_hum", "%")]:
        lo, hi = np.percentile(lab[col].dropna(), [5, 95])
        grid = np.linspace(lo, hi, 7)
        p = pdp(m, X, col, grid)
        print("   %-9s %6.0f%s → %6.0f%s :  EC %.3f → %.3f  (%+.3f)"
              % (col, lo, unit, hi, unit, p[0], p[-1], p[-1] - p[0]))
    print("   * out_hum 은 습도가 높을수록 증산이 줄므로 EC가 내려가야 방향이 맞음")

    print("\n=== 구동기 ===")
    for col in ["act_vent", "act_circfan"]:
        grid = np.linspace(0, 100, 6)
        p = pdp(m, X, col, grid)
        print("   %-12s 0%% → 100%% :  EC %.3f → %.3f  (%+.3f)"
              % (col, p[0], p[-1], p[-1] - p[0]))


if __name__ == "__main__":
    main()
