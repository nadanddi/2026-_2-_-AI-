# -*- coding: utf-8 -*-
"""Fair retest with a rule simple enough for 13 samples.

9 features on 13 greenhouses overfits by construction.  The physically natural
single predictor is the ADJACENT band (8-10C) -- chosen for being adjacent, not
for its correlation.  Also tried: two features, and the slope of the offset
across the warm bands extrapolated downward.
"""
import numpy as np
import pandas as pd

from predict_cold_offset import profile, BANDS, MIN_COLD
from common import load_raw


def loo(X, y, ridge=1.0):
    """Leave-one-out prediction with a tiny ridge, X already 2-D."""
    p = np.zeros(len(y))
    for i in range(len(y)):
        m = np.ones(len(y), bool); m[i] = False
        A = np.c_[np.ones(m.sum()), X[m]]
        w = np.linalg.solve(A.T @ A + ridge * np.eye(A.shape[1]), A.T @ y[m])
        p[i] = np.r_[1, X[i]] @ w
    return p


def main():
    tX, ty, sX = load_raw()
    d = tX.merge(ty[["row_id", "sub_temp"]], on="row_id", how="left")
    t = pd.DataFrame({f: profile(g.copy()) for f, g in d.groupby("farm")}).T
    fit = t[t.cold.notna()].drop(index=[f for f in ("F13", "F47", "F32") if f in t.index],
                                 errors="ignore")
    y = fit.cold.astype(float).values
    print("=== 온실 %d곳, 추운구간 오프셋 중앙 %+.2f, 표준편차 %.2f ===" % (len(y), np.median(y), y.std()))

    cands = {
        "규칙 없음 (다른 곳 중앙값)": None,
        "옆 구간(8~10℃) 하나만": ["b8_10"],
        "옆 구간 + 그 다음(10~15℃)": ["b8_10", "b10_15"],
        "따뜻한 구간 기울기 하나만": ["slope_warm"],
        "야간 온도차 하나만": ["night_gap"],
    }
    print("\n%-28s %8s %8s %10s" % ("규칙", "RMSE", "R2", "더 가까운 곳"))
    base = None
    for nm, cols in cands.items():
        if cols is None:
            p = np.array([np.median(np.delete(y, i)) for i in range(len(y))])
        else:
            X = fit[cols].astype(float).values
            X = (X - X.mean(0)) / (X.std(0) + 1e-9)
            p = loo(X, y)
        r = float(np.sqrt(np.mean((p - y) ** 2)))
        q = 1 - np.mean((p - y) ** 2) / np.var(y)
        if base is None:
            base, base_p = r, p
            print("%-28s %8.3f %8.3f %10s" % (nm, r, q, "-"))
        else:
            w = int((np.abs(p - y) < np.abs(base_p - y)).sum())
            print("%-28s %8.3f %8.3f %8d/%d" % (nm, r, q, w, len(y)))

    print("\n=== 표본이 몇 곳이면 판단할 수 있나 ===")
    print("  현재 %d곳, 오프셋 표준편차 %.2f℃" % (len(y), y.std()))
    print("  상관 0.65를 유의하게 잡으려면 대략 %d곳 이상 필요 (양측 5%%)" % 17)
    print("  → ≤6℃ 정답 %d행 이상인 온실이 %d곳뿐이라 더 늘릴 수 없음" % (MIN_COLD, len(y)))


if __name__ == "__main__":
    main()
