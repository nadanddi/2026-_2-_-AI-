# -*- coding: utf-8 -*-
"""HS2 (2026-10-12 집 클로드; HS1 후속).  서술.  HS1 에서 본 고EC 동의 운영 일정 지문:
  차광·보온 커튼이 9시에 닫히기 시작해 16시까지(8시 = 0, 10시 = 100, 17시 = 0), CO2 공급이 9·11·13시에 짧게(< 30).
정의 SIG = shade[8]==0 & shade[10]>=95 & shade[17]==0 & max(co2[8..14]) < 30.  (자기 기록 입력만)
보고: (1) 날짜별로 SIG 기록 수·그 기록의 정답 하루평균(학습)·12회차 예측(평가), (2) SIG 여부 x 고EC(하루평균 ≥ 1) 교차표(학습),
(3) 지문 일부만 쓴 변형(커튼만 / CO2만)의 교차표."""
import sys, os
sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env
import numpy as np, pandas as pd
here = os.path.dirname(os.path.abspath(__file__))
X = pd.concat([pd.read_csv(os.path.join(env.DATA, f)).assign(split=s) for f, s in (("train_X.csv", "train"), ("test_X.csv", "eval"))])
X = X[X.row_id.str[:3].isin(["F13", "F47"])]; X["rec"] = X.row_id.str[:7]; X["h"] = X.row_id.str[8:].astype(int)
sh = X.pivot(index="rec", columns="h", values="act_shade"); co = X.pivot(index="rec", columns="h", values="act_co2")
Y = pd.read_csv(os.path.join(env.DATA, "train_y.csv")); Y["rec"] = Y.row_id.str[:7]; y = Y.groupby("rec").sub_ec.mean()
P = pd.read_csv(r"C:\work\farmai\제출\12회차_2026-10-11(나)\submission_19.csv"); P["rec"] = P.row_id.str[:7]; pm = P.groupby("rec").sub_ec.mean()
C = pd.read_csv(os.path.join(here, "dm11_calendar_v1.csv")).set_index("rec")
D = pd.DataFrame(index=sh.index)
D["curtain"] = (sh[8] == 0) & (sh[10] >= 95) & (sh[17] == 0); D["co2"] = co.loc[:, 8:14].max(1) < 30
D["SIG"] = D.curtain & D.co2
D["day"] = C.day; D["split"] = C.split; D["y"] = y; D["pred"] = pm; D["val"] = D.y.fillna(D.pred)
T = D[D.split == "train"]
for c in ("SIG", "curtain", "co2"):
    print(f"\n{c} x high-EC (train, day-mean >= 1):"); print(pd.crosstab(T[c], T.y >= 1, margins=True).to_string())
print("\nby calendar day: SIG records (train y / eval pred), and highest non-SIG train y that day")
for d, G in D.dropna(subset=["day"]).groupby("day"):
    S = G[G.SIG]; ns = G[~G.SIG & (G.split == "train")]
    if len(S) == 0 and not (ns.y >= 1).any(): continue
    s = ", ".join(f"{r}({'y' if G.at[r,'split']=='train' else 'p'} {G.at[r,'val']:.2f})" for r in S.index)
    print(f"  day {int(d):3d}: SIG [{s}]  | max non-SIG train y {ns.y.max() if len(ns) else float('nan'):.2f}")
D.to_csv("hs2_v1.csv")
