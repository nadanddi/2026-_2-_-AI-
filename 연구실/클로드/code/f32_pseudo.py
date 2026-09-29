# -*- coding: utf-8 -*-
"""Could F32's exclusive actuators be reconstructed for F13/F47?

Step 0: verify no column is missing in F13/F47 training but present in test_X.
Step 1: what do F32's exclusive columns look like (irrigation-like or not)?
Step 2: can they be predicted from the 9 columns F32 shares with F13/F47?

The catch: a pseudo-signal is only worth adding if it is BOTH predictable
(otherwise it is noise) and not trivially recoverable by the downstream model
from the same inputs (otherwise it adds nothing).  Both ends are measured.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold

from common import load_raw

EXCL = ["act_side", "act_valve", "act_cool", "act_pump"]
SHARED = ["out_temp", "out_rad", "out_wspd", "in_temp", "in_hum", "in_co2",
          "act_vent", "act_circfan", "act_co2"]


def main():
    tX, ty, sX = load_raw()
    cols = [c for c in sX.columns if c not in ("row_id", "farm", "day", "hour", "t")]

    print("=== 0. F13/F47 학습에는 없는데 test_X에는 있는 컬럼? ===")
    tgt = tX[tX.farm.isin(["F13", "F47"])]
    found = []
    for c in cols:
        if not tgt[c].notna().any() and sX[c].notna().any():
            found.append(c)
    print("  →", ", ".join(found) if found else "없음 (학습·평가 가용 컬럼 집합이 정확히 같음)")
    print("  참고: 두 집합 모두 사용가능 %d개 / 전부결측 %d개"
          % (sum(tgt[c].notna().any() for c in cols),
             sum(not tgt[c].notna().any() for c in cols)))

    f32 = tX[tX.farm == "F32"].copy()
    print("\n=== 1. F32 고유 4개 컬럼의 성격 ===")
    print("  %-11s %7s %7s %8s %8s  %s" % ("컬럼", "평균", "0비율", "주간평균", "야간평균", "판정"))
    for c in EXCL:
        s = f32[c]
        day = f32.loc[(f32.hour >= 9) & (f32.hour <= 16), c].mean()
        night = f32.loc[(f32.hour <= 5) | (f32.hour >= 21), c].mean()
        verdict = ("야간 우세 → 난방순환" if night > day * 1.2 else
                   "주간 우세 → 관수 성격" if day > night * 1.2 else "차이 작음")
        print("  %-11s %7.1f %6.1f%% %8.1f %8.1f  %s"
              % (c, s.mean(), 100 * (s == 0).mean(), day, night, verdict))

    print("\n  시각별 평균 (0~23시):")
    for c in EXCL:
        h = f32.groupby("hour")[c].mean().round(0).astype(int).tolist()
        print("   %-11s %s" % (c, " ".join("%3d" % v for v in h)))

    print("\n=== 2. 공유 9개 컬럼으로 예측 가능한가 (F32 내부, 날짜 그룹 5겹) ===")
    f32 = f32.dropna(subset=SHARED, how="all")
    X = f32[SHARED]
    gkf = GroupKFold(n_splits=5)
    print("  %-11s %8s %8s   해석" % ("대상", "R2", "RMSE"))
    for c in EXCL:
        y = f32[c]
        m = y.notna()
        if m.sum() < 500:
            continue
        pred = np.full(m.sum(), np.nan)
        Xc, yc, gc = X[m], y[m], f32.loc[m, "day"]
        for tr, va in gkf.split(Xc, yc, groups=gc):
            mdl = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31,
                                    random_state=7, n_jobs=4, verbose=-1)
            mdl.fit(Xc.iloc[tr], yc.iloc[tr])
            pred[va] = mdl.predict(Xc.iloc[va])
        r2 = 1 - np.mean((pred - yc.values) ** 2) / np.var(yc.values)
        rmse = float(np.sqrt(np.mean((pred - yc.values) ** 2)))
        note = ("거의 결정적 → 새 정보 없음" if r2 > 0.9 else
                "예측 불가 → 잡음" if r2 < 0.3 else "중간")
        print("  %-11s %8.3f %8.2f   %s" % (c, r2, rmse, note))

    print("\n=== 3. F32 자체가 F13/F47과 얼마나 다른 운전인가 (공유 컬럼 분포) ===")
    tg = tX[tX.farm.isin(["F13", "F47"])]
    print("  %-12s %14s %14s" % ("컬럼", "F32 평균", "F13+F47 평균"))
    for c in SHARED:
        print("  %-12s %14.1f %14.1f" % (c, f32[c].mean(), tg[c].mean()))


if __name__ == "__main__":
    main()
