# -*- coding: utf-8 -*-
"""FL1 (2026-10-12 집 클로드; '규정 안에서 닿을 수 있는 최선' 계산, 서술·추정).
규정: 평가 기록은 자기 0..h 입력 + 같은 ID 앞번호 평가 입력 + train 전체(앞뒤 날짜 모두)만.  뒤 평가 입력 사용 금지.
같은 동의 정답이 규정상 '닿는' 경우: 하루 전/후 날짜(달력 DM11)에 그 동이 train 기록으로 있음.  동 식별은 완벽하다고 가정(낙관적 하한).
확률: 날짜 d±k 묶음에서 train 기록 비율 = 그 동이 정답 있는 기록일 확률(동 4개 중 무작위 근사).
0시 오차 sigma: 1일 이웃 정답 → .05(EV 앵커 기록 0시 RMSE .037~.059), 2일 → .12(k=2 정보 −48%), 그 밖 → .23(EV 2차 앵커 없는 기록).
환산식(EV 2차 오라클에서 맞춤): LB ~ .622 * sqrt(.1059² + .846 sigma_eff²), sigma_eff² = 평균 sigma_i² (기댓값).  검산: 현재(앵커 21, 나머지 .23) → .1255 (실제 .1271).
"""
import sys, os
sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env
import numpy as np, pandas as pd
H = os.path.dirname(os.path.abspath(__file__))
C = pd.read_csv(os.path.join(H, "..", "rise", "dm11_calendar_v1.csv"))
g = C.dropna(subset=["day"]).groupby("day").agg(n=("rec", "size"), lab=("split", lambda s: int((s == "train").sum())))
frac = (g.lab / g.n).to_dict()
E = C[C.split == "eval"].copy()
lb = lambda s2: .622 * np.sqrt(.1059 ** 2 + .846 * s2)
rows = []
for r in E.itertuples():
    d = r.day
    if not np.isfinite(d): rows.append(dict(rec=r.rec, day=np.nan, p1=0, p2=0)); continue
    pb1, pf1 = frac.get(d - 1, 0), frac.get(d + 1, 0)
    pb2, pf2 = frac.get(d - 2, 0), frac.get(d + 2, 0)
    pf1_only_train = pf1  # 다음날 train 기록은 규정상 사용 가능(뒤 '평가' 입력이 아님)
    p1 = 1 - (1 - pb1) * (1 - pf1_only_train)
    p2 = (1 - p1) * (1 - (1 - pb2) * (1 - pf2))
    rows.append(dict(rec=r.rec, day=d, p1=p1, p2=p2, pb1=pb1, pf1=pf1))
R = pd.DataFrame(rows); R["p3"] = 1 - R.p1 - R.p2
for nm, s1 in (("낙관(1일 이웃 sigma .05)", .05), ("보수(1일 이웃 sigma .08)", .08)):
    s2 = (R.p1 * s1 ** 2 + R.p2 * .12 ** 2 + R.p3 * .23 ** 2).mean()
    s2b = (R.p1 * s1 ** 2 + R.p2 * .23 ** 2 + R.p3 * .23 ** 2).mean()
    s2c = ((R.p1 + 0) * s1 ** 2 + (1 - R.p1) * .23 ** 2).mean()
    print(f"{nm}: 기대 1일 이웃 {R.p1.sum():.1f}기록, 2일 {R.p2.sum():.1f}, 그 밖 {R.p3.sum():.1f} | 하한 LB ~ {lb(s2):.4f} (2일 정보 안 쓰면 {lb(s2b):.4f})")
print("현재 검산(앵커 21, 나머지 sigma .23):", round(lb((21 * .04 ** 2 + 39 * .23 ** 2) / 60), 4))
print("0시 완벽(sigma 0):", round(lb(0), 4))
print("\n뒤쪽(다음날) train 정답만으로 1일 이웃이 생길 기대 기록 수(앞쪽엔 없을 때):", round(((1 - R.pb1) * R.pf1).sum(), 1))
print("앞뒤 모두 1일 이웃 정답 확률 0인 기록:", list(R[(R.p1 == 0)].rec))
R.round(3).to_csv(os.path.join(H, "fl1_floor_v1.csv"), index=False)
