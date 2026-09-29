# -*- coding: utf-8 -*-
"""Can a greenhouse's COLD-band behaviour be predicted from its WARM-band one?

The blocker so far: to pick donor greenhouses resembling F13/F47 in the cold,
we would need to know how F13/F47 behave in the cold -- which is the unknown.
This turns the question around.  Warm-band behaviour is measured on thousands
of rows for every greenhouse, F13/F47 included.  If a rule maps warm behaviour
to cold behaviour across the greenhouses that have both, that rule can be
applied to F13/F47.

Leave-one-out over greenhouses with a measurable <=6C offset; compared against
simply using the median of the others (which is what "no rule" would give).
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV

from common import load_raw

BANDS = [("b8_10", 8, 10), ("b10_15", 10, 15), ("b15_20", 15, 20), ("b20", 20, 99)]
MIN_COLD = 20


def profile(g):
    g = g.sort_values("t")
    p = g.set_index("t").reindex(np.arange(g.t.min(), g.t.max() + 1))
    p["air"] = p.in_temp.ewm(halflife=3, ignore_na=True).mean()
    p = p.dropna(subset=["sub_temp", "air"])
    diff = p.sub_temp - p.air
    r = {}
    for nm, a, b in BANDS:
        m = (p.air >= a) & (p.air < b)
        r[nm] = diff[m].median() if m.sum() >= 30 else np.nan
        r["n_" + nm] = int(m.sum())
    m = p.air < 6
    r["cold"] = diff[m].median() if m.sum() >= MIN_COLD else np.nan
    r["n_cold"] = int(m.sum())
    # extra describable traits, all from the warm side
    warm = p[p.air >= 8]
    r["slope_warm"] = np.polyfit(warm.air, warm.sub_temp, 1)[0] if len(warm) > 100 else np.nan
    r["lag_corr"] = max((p.sub_temp.corr(p.in_temp.shift(L)), L) for L in range(0, 7))[1]
    r["night_gap"] = float((p.loc[(p.index % 24 <= 5) | (p.index % 24 >= 21), "sub_temp"]
                            - p.loc[(p.index % 24 <= 5) | (p.index % 24 >= 21), "air"]).median())
    r["amp"] = float(p.groupby(p.index // 24).sub_temp.apply(lambda s: s.max() - s.min()).median())
    r["hum"] = float(p.in_hum.median())
    return r


def main():
    tX, ty, sX = load_raw()
    d = tX.merge(ty[["row_id", "sub_temp"]], on="row_id", how="left")
    rows = {f: profile(g.copy()) for f, g in d.groupby("farm")}
    t = pd.DataFrame(rows).T

    FEATS = ["b8_10", "b10_15", "b15_20", "b20", "slope_warm", "lag_corr",
             "night_gap", "amp", "hum"]
    fit = t[t.cold.notna() & t[FEATS].notna().all(axis=1)]
    fit = fit.drop(index=[f for f in ("F13", "F47", "F32") if f in fit.index])
    print("=== 규칙을 학습할 수 있는 온실: %d곳 (≤6℃ 정답 %d행 이상) ===" % (len(fit), MIN_COLD))
    print("  추운 구간 오프셋 분포: 중앙 %+.2f, 범위 %+.2f ~ %+.2f, 표준편차 %.2f"
          % (fit.cold.median(), fit.cold.min(), fit.cold.max(), fit.cold.std()))

    X = fit[FEATS].astype(float).values
    y = fit.cold.astype(float).values
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-9)

    pred_rule, pred_med = np.zeros(len(y)), np.zeros(len(y))
    for i in range(len(y)):
        m = np.ones(len(y), bool); m[i] = False
        pred_med[i] = np.median(y[m])
        pred_rule[i] = RidgeCV(alphas=np.logspace(-1, 3, 20)).fit(Xs[m], y[m]).predict(Xs[i:i + 1])[0]

    def sc(p):
        return (float(np.sqrt(np.mean((p - y) ** 2))),
                1 - np.mean((p - y) ** 2) / np.var(y))
    r_med, q_med = sc(pred_med)
    r_rule, q_rule = sc(pred_rule)
    print("\n=== leave-one-out: 추운 구간 오프셋을 맞힐 수 있나 ===")
    print("  %-28s RMSE %.3f   R2 %+.3f" % ("규칙 없음(다른 곳 중앙값)", r_med, q_med))
    print("  %-28s RMSE %.3f   R2 %+.3f" % ("따뜻한 구간으로 예측(Ridge)", r_rule, q_rule))
    better = int((np.abs(pred_rule - y) < np.abs(pred_med - y)).sum())
    print("  규칙이 더 가까운 곳 %d/%d" % (better, len(y)))

    # single strongest warm-band correlate
    print("\n=== 각 따뜻한 구간 지표와 추운 구간 오프셋의 상관 ===")
    for c in FEATS:
        print("  %-12s r = %+.3f" % (c, np.corrcoef(fit[c].astype(float), y)[0, 1]))

    if q_rule > 0:
        full = RidgeCV(alphas=np.logspace(-1, 3, 20)).fit(Xs, y)
        print("\n=== F13/F47에 적용 ===")
        for f in ("F13", "F47"):
            if t.loc[f, FEATS].notna().all():
                z = ((t.loc[f, FEATS].astype(float).values - X.mean(0)) / (X.std(0) + 1e-9))
                print("  %s 추정 추운구간 오프셋 %+.2f℃   (관측 %s, %d행)"
                      % (f, full.predict(z.reshape(1, -1))[0],
                         "%+.2f" % t.loc[f, "cold"] if np.isfinite(t.loc[f, "cold"]) else "측정불가",
                         t.loc[f, "n_cold"]))
        print("  * 예측 오차 %.2f℃ 를 감안해야 함" % r_rule)
    else:
        print("\n=== 규칙이 중앙값보다 낫지 않음 → F13/F47에 적용할 근거 없음 ===")
    t.to_csv("cold_offset_profile.csv")


if __name__ == "__main__":
    main()
