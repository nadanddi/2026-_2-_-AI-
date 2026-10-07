# -*- coding: utf-8 -*-
"""DT0 — 2차 기록의 날짜 확정 경로 분류와 SG2 오차 (전제 검산, 진단 전용) — 2026-10-07 연구실 클로드
외기 24h 완전 쌍둥이(z-RMSE ≤ .05, 표준화는 학습 기록 기준)로 날짜 묶음을 만들고 2차 기록을 나눈다:
  C1 같은 온실 ID 1차 정답 기록과 쌍둥이 (SG2 가 이미 날짜를 앎)
  C2 C1 아님, 다른 온실 ID 1차 정답 기록과 쌍둥이 (날짜는 알 수 있으나 SG2 는 안 씀)
  C3 1차와는 쌍둥이 없음, 다른 2차/평가 기록과만 쌍둥이
  C4 쌍둥이 없음
평가 60개(입력만)와 2차 정답 56개의 분류 수, SG2 OOF(3 검증기) 오차를 분류별로.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", u"집", u"클로드", "research"))
import env  # noqa
import numpy as np, pandas as pd

R = os.path.dirname(os.path.abspath(__file__)); H = os.path.join(env.ROOT, u"집", u"클로드", "research", "local")
W = ["out_temp", "out_hum", "out_wspd", "out_rad"]
TR = pd.read_csv(os.path.join(env.DATA, "train_X.csv"), usecols=["row_id"] + W); TE = pd.read_csv(os.path.join(env.DATA, "test_X.csv"), usecols=["row_id"] + W)
TR["is_test"] = False; TE["is_test"] = True
X = pd.concat([TR, TE]); X["farm"], X["day"], X["hour"] = X.row_id.str[:3], X.row_id.str[4:7].astype(int), X.row_id.str[8:10].astype(int)
X = X[X.farm.isin(["F13", "F47"])]
WV = X.pivot_table(index=["farm", "day"], columns="hour", values=W)
istest = X.groupby(["farm", "day"]).is_test.first()
trk = [k for k in WV.index if not istest[k]]
mu, sd = WV.loc[trk].mean(), WV.loc[trk].std()
WZ = (WV - mu) / sd
keys = list(WZ.index); A = WZ.values
D = np.sqrt(np.nanmean((A[:, None, :] - A[None, :, :]) ** 2, axis=2))
cls = {}
for i, (f, d) in enumerate(keys):
    if d < 179:
        continue
    tw = [keys[j] for j in np.where(D[i] <= .05)[0] if j != i]
    own1 = [k for k in tw if k[0] == f and k[1] < 179 and not istest[k]]
    oth1 = [k for k in tw if k[0] != f and k[1] < 179 and not istest[k]]
    cls[(f, d)] = "C1" if own1 else "C2" if oth1 else "C3" if tw else "C4"
C = pd.Series(cls, name="cls")
print("평가 60:", C[[k for k in C.index if istest[k]]].value_counts().sort_index().to_dict())
print("2차 정답:", C[[k for k in C.index if not istest[k]]].value_counts().sort_index().to_dict())
S = pd.read_csv(os.path.join(H, "ec3_SG2_all.csv")); S["sg"] = S[["sg_23", "sg_808", "sg_9090"]].mean(axis=1)
S["cls"] = [cls.get((f, d)) for f, d in zip(S.farm, S.day)]; S["dy"] = S.groupby(["validator", "farm", "day"]).sub_ec.transform("mean")
S["e"] = S.sg - S.sub_ec
rm = lambda x: np.sqrt(np.mean(np.square(x)))
for v, g in S.groupby("validator"):
    T = (g.e ** 2).sum()
    print(" %-8s " % v + " | ".join("%s %d일 RMSE %.3f(일반 %.3f) 몫 %.0f%%" % (c, q.groupby(["farm", "day"]).ngroups, rm(q.e), rm(q[q.dy < 1].e) if (q.dy < 1).any() else np.nan, 100 * (q.e ** 2).sum() / T)
                                       for c, q in g.groupby("cls")))
C.rename_axis(["farm", "day"]).reset_index().assign(test=lambda x: [istest[(f, d)] for f, d in zip(x.farm, x.day)]).to_csv(
    os.path.join(R, "..", "results", "ec_dt0_date_class_v1.csv"), index=False, encoding="utf-8-sig")
