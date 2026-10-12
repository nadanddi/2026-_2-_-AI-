# -*- coding: utf-8 -*-
"""BME1 (2026-10-13 집 클로드; 사용자: 'EC 좀 더 보자') 서술: 현 EC 구성 기준 EV 남은 오차 지도.
EV EC 행(ev_validator/ec/out_v1/v2_ev_ec_rows.csv, BASE 계절, 시드 3개 평균): SG2 = 앵커 없음(12회차 EC 에 해당, W2 는 평가 순수 창 전용), EA2 = v7 앵커(13회차 EC).
보고: 전체/2차 RMSE, 하루 수준 몫, 시각대별, 하루 평균 EC 수준별(보통 / 0.8~1 / 고EC ≥ 1) SSE 몫, 앵커 여부별, 상위 기록 집중, 오라클(하루 수준 완벽, 0시 완벽, 고EC 기록 완벽), 상위 15기록.
"""
import os
import env  # noqa: F401
import numpy as np, pandas as pd
H = os.path.dirname(os.path.abspath(__file__))
R = pd.read_csv(os.path.join(H, "ev_validator", "ec", "out_v1", "v2_ev_ec_rows.csv")); R = R[(R.arm == "BASE") & (R.seed.astype(str) != "avg")]
G = R.groupby(["row_id", "rec", "day", "hour", "farm"], as_index=False).agg(y=("y", "first"), SG2=("SG2", "mean"), EA2=("EA2", "mean"), anchored=("anchored", "first"))
rm = lambda e: float(np.sqrt(np.mean(np.square(e))))
G["p2"] = G.day >= 179; ym = G.groupby("rec").y.transform("mean"); G["lvl"] = pd.cut(ym, [-1, .8, 1.0, 9], labels=["normal<0.8", "0.8-1", "high>=1"])
for col in ("SG2", "EA2"):
    e = G[col] - G.y; G["e_" + col] = e
    print(f"\n== {col}: RMSE all {rm(e):.4f} | pass-2 {rm(e[G.p2]):.4f}")
    for nm, S in (("all", G), ("pass-2", G[G.p2])):
        ee = S["e_" + col]; dm = S.groupby("rec")["e_" + col].transform("mean")
        print(f"   {nm}: day-level share {np.sum(dm ** 2) / np.sum(ee ** 2):.2f}")
        print(f"   {nm}: SSE share by level:", {str(k): round(float(np.sum(g['e_' + col] ** 2) / np.sum(ee ** 2)), 2) for k, g in S.groupby("lvl", observed=True)},
              "| records by level:", S.groupby("lvl", observed=True).rec.nunique().to_dict())
        print(f"   {nm}: SSE share anchored records {np.sum(S[S.anchored]['e_' + col] ** 2) / np.sum(ee ** 2):.2f} (records {S[S.anchored].rec.nunique()})")
        sse = S.groupby("rec")["e_" + col].apply(lambda s: np.sum(s ** 2)).sort_values(ascending=False); n = len(sse)
        print(f"   {nm}: top 5/10% records SSE share {sse.iloc[:max(1, n // 20)].sum() / sse.sum():.2f} / {sse.iloc[:max(1, n // 10)].sum() / sse.sum():.2f}")
        e0 = S.groupby("rec")["e_" + col].transform("first")
        hi = S.lvl == "high>=1"
        print(f"   {nm}: oracle RMSE now {rm(ee):.4f} | day level perfect {rm(ee - dm):.4f} | 0h level perfect (subtract 0h error all day) {rm(ee - e0):.4f} | high-EC records perfect {rm(np.where(hi, 0, ee)):.4f}")
    print("   by hour (all):", {h: round(rm(g["e_" + col]), 3) for h, g in G.groupby(pd.cut(G.hour, [-1, 0, 5, 11, 17, 23]))})
T = G.groupby("rec").agg(y=("y", "mean"), p2=("p2", "first"), anchored=("anchored", "first"), errSG2=("e_SG2", "mean"), errEA2=("e_EA2", "mean"),
                         sseEA2=("e_EA2", lambda s: float(np.sum(s ** 2))))
T["share"] = T.sseEA2 / T.sseEA2.sum()
pd.set_option("display.width", 200); print("\ntop 15 records (EA2):"); print(T.sort_values("share", ascending=False).head(15).round(3).to_string())
