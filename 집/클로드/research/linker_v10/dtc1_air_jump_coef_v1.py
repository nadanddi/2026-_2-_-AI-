# DTC1 (2026-10-13 집 클로드; 2단계 준비) 자정 공기 점프 보정 계수: y0(오늘 0시 배지) − A(전날 23시 배지) = a + c·dT, dT = 오늘 0시 실내 − 전날 23시 실내.
# 학습 확실 연결(진단) 중 1차 대상(번호 < 179)으로 농장별 최소제곱; 2차는 서술만. 출력 local/linker_v10/dtc1_coef.csv (farm, a, c).
import sys, os; sys.path.insert(0, r"C:\work\farmai\집\클로드\research\linker_v4"); sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env
from feat4 import *
TL = pd.read_pickle(S3 + r"\truth.pkl"); TL = TL[TL.cert]
R = pd.DataFrame([dict(farm=FARM[int(b)], p2=NUM[int(b)] >= 179, dy=ST[int(b), 0] - ST[int(a), 23], dT=IN_[int(b), 0, 0] - IN_[int(a), 23, 0]) for a, b in zip(TL.a1, TL.b) if TRAIN[int(a)] and TRAIN[int(b)]]).dropna()
rows = []
for (f, p2), G in R.groupby(["farm", "p2"]):
    c, a = np.polyfit(G.dT, G.dy, 1); res = G.dy - (a + c * G.dT)
    print(f"{f} {'pass-2' if p2 else 'pass-1'}: n {len(G)} | a {a:+.3f} c {c:.3f} | resid sd {res.std():.3f} vs raw sd {G.dy.std():.3f} | |dT|>1.5 n {int((G.dT.abs() > 1.5).sum())}")
    if not p2: rows.append(dict(farm=f, a=a, c=c))
pd.DataFrame(rows).to_csv(os.path.join(env.LOCAL, "linker_v10", "dtc1_coef.csv"), index=False)
