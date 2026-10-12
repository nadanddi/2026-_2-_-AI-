# -*- coding: utf-8 -*-
"""BM1 (2026-10-12 집 클로드; 사용자: '병목을 더 해결해 보자' — 13회차 이후 온도 남은 오차 지도).  서술.
EV(평가 기하) 저장 결과 ev_validator/temp/local_ckpt/ev_all.pkl, 13회차 구성 = v22 (4칸 평균, 'a' 열).  y = sub_temp.
보고: (1) 전체·2차 RMSE, 하루 수준(기록 평균 오차) 몫 vs 하루 안 몫, (2) 시각대별 RMSE, (3) 실내온도 구간별(추운 행) RMSE·행 비중·SSE 몫,
(4) 앵커 기록(v22 ≠ v18) vs 아닌 기록, (5) 기록 집중(상위 5·10·20% 기록의 SSE 몫), (6) 오라클: 하루 수준 완벽 / 0시 완벽(0시 오차를 하루 내내 감쇠 없이 빼기) /
상위 10% 기록 완벽 시 RMSE, (7) 큰 오차 기록 상위 15 (하루 평균 오차, 0시 오차, 실내 평균, 2차 여부, 앵커 여부).
"""
import os, sys
import env  # noqa: F401
import numpy as np, pandas as pd
H = os.path.dirname(os.path.abspath(__file__))
D = pd.read_pickle(os.path.join(H, "ev_validator", "temp", "local_ckpt", "ev_all.pkl"))
cells = ["7A", "7B", "101A", "101B"]
D["q"] = D[[f"v22_a_{c}" for c in cells]].mean(1); D["q18"] = D[[f"v18_a_{c}" for c in cells]].mean(1)
D["rec"] = D.row_id.str[:7]; D["e"] = D.q - D.sub_temp; D["p2"] = D.day >= 179
rm = lambda e: float(np.sqrt(np.mean(np.square(e))))
anc = D.groupby("rec").apply(lambda g: bool((g.q - g.q18).abs().max() > 1e-6)); D["anc"] = D.rec.map(anc)
pd.set_option("display.width", 200)
for nm, S in (("all", D), ("pass-2", D[D.p2])):
    dm = S.groupby("rec").e.transform("mean")
    print(f"(1) {nm}: RMSE {rm(S.e):.4f} | day-level share {np.sum(dm**2)/np.sum(S.e**2):.2f} | records {S.rec.nunique()}")
print("(2) by hour:", {h: round(rm(g.e), 3) for h, g in D.groupby(pd.cut(D.hour, [-1, 0, 2, 5, 7, 11, 15, 19, 23]))})
print("(3) by indoor temp band: RMSE / row share / SSE share")
for b, g in D.groupby(pd.cut(D.in_temp, [-50, 6, 8, 10, 14, 20, 60])):
    print(f"   {str(b):12s} RMSE {rm(g.e):.3f} rows {len(g)/len(D):.3f} SSE {np.sum(g.e**2)/np.sum(D.e**2):.3f}")
print("(4) anchored records:", int(anc.sum()), "| RMSE anchored %.3f vs not %.3f" % (rm(D.e[D.anc]), rm(D.e[~D.anc])))
sse = D.groupby("rec").e.apply(lambda s: np.sum(s ** 2)).sort_values(ascending=False); n = len(sse)
print("(5) top 5/10/20%% records SSE share: %.2f / %.2f / %.2f" % tuple([sse.iloc[:max(1, int(n * f))].sum() / sse.sum() for f in (.05, .10, .20)]))
dm = D.groupby("rec").e.transform("mean"); e0 = D.groupby("rec").e.transform("first")
top = set(sse.index[:int(n * .1)])
print("(6) oracle RMSE: now %.4f | day level perfect %.4f | 0h perfect (subtract 0h error all day) %.4f | 0h perfect decaying exp(-h/4.5) %.4f | top-10%% records perfect %.4f" % (
    rm(D.e), rm(D.e - dm), rm(D.e - e0), rm(D.e - e0 * np.exp(-D.hour / 4.5)), rm(np.where(D.rec.isin(top), 0, D.e))))
R = D.groupby("rec").agg(day_err=("e", "mean"), err0=("e", "first"), rmse=("e", rm), tin=("in_temp", "mean"), tmin=("in_temp", "min"), p2=("p2", "first"), anc=("anc", "first"), y=("sub_temp", "mean"))
R["sse_share"] = sse / sse.sum()
print("(7) top 15 records:"); print(R.sort_values("sse_share", ascending=False).head(15).round(2).to_string())
print("\n   day_err vs indoor mean temp (spearman) %.2f | vs indoor min %.2f" % (R.day_err.rank().corr(R.tin.rank()), R.day_err.rank().corr(R.tmin.rank())))
