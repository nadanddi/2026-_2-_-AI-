# -*- coding: utf-8 -*-
"""What does an RMSE of 0.55 C actually mean for substrate temperature?"""
import numpy as np
import pandas as pd
import lightgbm as lgb

from common import load_raw, rmse, split_mask, TARGET_FARMS
from geometry_cv import geometry_folds
from make_submission import T_HUB, SEEDS
import features_v2 as F2

R = 0.5456   # the score the current best temperature submission received


def main():
    tX, ty, sX = load_raw()
    d = tX.merge(ty[["row_id", "sub_temp"]], on="row_id", how="left")
    t = d[d.farm.isin(TARGET_FARMS) & d.sub_temp.notna()].copy()

    print("=== 1. 맞혀야 하는 값 자체의 크기 ===")
    print("  배지 온도 범위      %.2f ~ %.2f ℃" % (t.sub_temp.min(), t.sub_temp.max()))
    print("  표준편차            %.2f ℃" % t.sub_temp.std())
    print("  하루 안 변동폭 중앙  %.2f ℃" % t.groupby([t.farm, t.day]).sub_temp
          .apply(lambda s: s.max() - s.min()).median())

    print("\n=== 2. 자연스러운 비교 기준 ===")
    ch = []
    for f, g in t.groupby("farm"):
        g = g.sort_values("t")
        dd = g.sub_temp.diff().where(g.t.diff() == 1).abs()
        ch.append(dd[(g.hour != 0)])
    ch = pd.concat(ch).dropna()
    print("  배지 온도의 1시간 변화 중앙값   %.2f ℃   ← 오차 0.55와 거의 같음" % ch.median())
    print("  라벨 기록 해상도              0.01 ℃")
    print("  다른 47개 온실의 라벨 해상도    1 ℃ (반올림 잡음 %.2f℃)" % (1 / np.sqrt(12)))

    print("\n=== 3. 다른 방법으로 찍으면 ===")
    a3 = t.groupby("farm", group_keys=False).apply(
        lambda g: g.sort_values("t").in_temp.ewm(halflife=3, ignore_na=True).mean())
    t = t.assign(a3=a3.values)
    for nm, p in [("온실별 평균으로 전부 채우기", t.farm.map(t.groupby("farm").sub_temp.mean())),
                  ("현재 실내온도 그대로", t.in_temp),
                  ("실내온도 3시간 평활 그대로", t.a3),
                  ("실내온도 3시간 평활 − 0.7℃", t.a3 - 0.7)]:
        z = pd.concat([t.sub_temp, p], axis=1).dropna()
        print("  %-26s RMSE %.3f" % (nm, rmse(z.iloc[:, 1], z.iloc[:, 0])))

    print("\n=== 4. 오차 0.55는 실제로 어떤 모습인가 (블록 CV 오차분포를 0.55로 환산) ===")
    panel = F2.build(tX, sX).merge(ty[["row_id", "sub_temp"]], on="row_id", how="left")
    panel["is_test"] = panel.row_id.isin(set(sX.row_id))
    v = F2.view(panel, "sub_temp")
    lab = panel[(~panel.is_test) & panel.sub_temp.notna()].reset_index(drop=True)
    oof = np.full(len(lab), np.nan)
    for fd in geometry_folds():
        trm, vam = split_mask(lab, fd)
        tr, va = lab[trm], lab[vam]
        ps = [lgb.LGBMRegressor(random_state=s, n_jobs=4, verbose=-1, **T_HUB)
              .fit(tr[v], tr.sub_temp).predict(va[v]) for s in SEEDS]
        oof[np.where(vam)[0]] = np.mean(ps, axis=0)
    got = ~np.isnan(oof)
    e = np.abs(oof[got] - lab.sub_temp.values[got])
    scale = R / float(np.sqrt(np.mean(e ** 2)))
    e = e * scale
    print("  (CV 오차분포를 실제 점수 %.4f 에 맞춰 축소, 배율 %.2f)" % (R, scale))
    print("  오차 중앙값          %.2f ℃" % np.median(e))
    print("  오차 평균            %.2f ℃" % e.mean())
    for th in (0.2, 0.5, 1.0, 1.5, 2.0):
        print("  %.1f℃ 이내로 맞힌 비율   %5.1f%%" % (th, 100 * (e <= th).mean()))
    print("  상위 1%% 최악 오차      %.2f ℃ 이상" % np.percentile(e, 99))

    print("\n=== 5. 순위 맥락 ===")
    print("  1위 딸기모찌26  0.4985")
    print("  우리 최고       0.5456   ← 차이 %.4f ℃" % (0.5456 - 0.4985))
    print("  1회차           0.7450")
    print("  기준선(온실평균)  4.62")


if __name__ == "__main__":
    main()
