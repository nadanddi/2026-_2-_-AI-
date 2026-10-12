# -*- coding: utf-8 -*-
"""NI2 (2026-10-12 집 클로드; 사용자: '온도도 봐봐' = 떨어진 날(같은 동 앞뒤 정답 없음) 온도는 맞추기 힘든가).  서술.
NI1(6.686)의 온도판.  학습 기록을 NF1(nf1_v1.csv, 진단용 연결) 동 수준으로 나눔: 앞뒤 모두 정답 / 앞만 / 뒤만 / 둘 다 없음.
온도 예측 q = DIAG10 OOF(ta2 → A1.frames, 11회차 구성 4칸 평균, 앵커 없음).  보고: 집단별 RMSE(전체, 0~7시, 8~23시), 하루 평균 오차 |mean| 중앙,
1·2차별, 그리고 떨어진 기록 목록(하루 평균 오차·실내 최저온).
"""
import os, sys, importlib.util
import env  # noqa: F401
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.argv = ["x"]
s = importlib.util.spec_from_file_location("ta2", os.path.join(HERE, "ta2_anchor_debiased_v1.py")); A2 = importlib.util.module_from_spec(s); s.loader.exec_module(A2)
M, cells = A2.A1.frames()["DIAG10"]
M = M.reset_index(drop=True); M["rec"] = M.farm + "_" + M.day.map("{:03d}".format); q = np.mean(cells, 0); M["e"] = q - M.sub_temp.values
NF = pd.read_csv(os.path.join(HERE, "ec0h", "rise", "nf1_v1.csv")); NF = NF[NF.split == "train"].copy()
hp = NF.true_prev_lab.fillna(False).astype(bool); hn = NF.true_next_lab.fillna(False).astype(bool)
NF["grp"] = np.select([hp & hn, hp & ~hn, ~hp & hn], ["prev & next", "prev only", "next only"], "neither (isolated)")
M["grp"] = M.rec.map(dict(zip(NF.rec, NF.grp))); M = M.dropna(subset=["grp"])
rm = lambda e: float(np.sqrt(np.mean(np.square(e))))
rows = []
for (g, p2), G in M.groupby(["grp", M.day >= 179]):
    dm = G.groupby("rec").e.mean()
    rows.append(dict(group=g, pass2=p2, records=G.rec.nunique(), rmse=rm(G.e), rmse_0_7=rm(G.e[G.hour <= 7]), rmse_8_23=rm(G.e[G.hour >= 8]), med_abs_daymean=float(dm.abs().median())))
for g, G in M.groupby("grp"):
    dm = G.groupby("rec").e.mean()
    rows.append(dict(group=g, pass2="all", records=G.rec.nunique(), rmse=rm(G.e), rmse_0_7=rm(G.e[G.hour <= 7]), rmse_8_23=rm(G.e[G.hour >= 8]), med_abs_daymean=float(dm.abs().median())))
pd.set_option("display.width", 200); print(pd.DataFrame(rows).round(3).to_string(index=False))
I = M[M.grp == "neither (isolated)"]
print("\nisolated records: day-mean error (pred - truth), truth day mean")
print(I.groupby("rec").agg(err=("e", "mean"), y=("sub_temp", "mean")).round(2).sort_values("err").T.to_string())
