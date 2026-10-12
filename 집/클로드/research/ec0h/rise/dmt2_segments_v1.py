# -*- coding: utf-8 -*-
"""DMT2 (2026-10-13 집 클로드; 사용자: '그 순서대로' 1단계 — 규정 안 달력).  학습 기록만으로 만든 DMT(6.753)에 '끊김' 표시.
DMT 는 평가만 있는 날(순수 2차 창)을 모르므로 그 앞뒤를 붙여 거짓 인접을 만듦 → 학습 입력만으로 탐지:
  이웃 날짜 k−1 → k 의 외기 자정 연속 비용 = min over (a on k−1, b on k) Σ w·|X[a,23] + .5(X[a,23]−X[a,22]) − X[b,0]| / sdO (v7 oj 식, w = 1,1,.3,.5).
  기준: 학습 확실 연결(진단, 같은 동 연속 날) 의 oj 분포 99% 분위 T. 비용 > T 이면 '끊김'.
출력 local/dmt2_days.csv: rec, day(DMT), seg(끊김으로 나뉜 구간 번호). 날짜 차는 같은 seg 안에서만 정의.
보고: T, 끊김 위치(앞뒤 날짜의 기록들), DM11(평가 입력 사용 달력)과 비교 — 끊김이 순수 창 경계를 잡는지(서술용 대조만).
"""
import sys, os
sys.path.insert(0, r"C:\work\farmai\집\클로드\research\linker_v4"); sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env
from feat4 import *
here = os.path.dirname(os.path.abspath(__file__))
rid = lambda i: f"{FARM[i]}_{NUM[i]:03d}"; idx = {rid(i): i for i in range(N)}
T = pd.read_csv(os.path.join(here, "dmt_calendar_v1.csv")); T = T[T.split == "train"]
DAY = dict(zip(T.rec, T.day))
W = np.array([1, 1, .3, .5])
oj = lambda a, b: float(np.sum(W * np.abs(X[a, 23] + .5 * (X[a, 23] - X[a, 22]) - X[b, 0]) / sdO))
TL = pd.read_pickle(S3 + r"\truth.pkl"); TLc = TL[TL.cert]
base = [oj(int(a), int(b)) for a, b in zip(TLc.a1, TLc.b) if TRAIN[int(a)] and TRAIN[int(b)]]
Tq = float(np.quantile(base, .99)); print(f"certain-link midnight outdoor cost: median {np.median(base):.2f}, 99% {Tq:.2f}, max {max(base):.2f}")
byday = T.dropna(subset=["day"]).groupby("day").rec.apply(list).to_dict(); days = sorted(byday)
breaks = []
seg = {}; s = 0
for k, d in enumerate(days):
    if k > 0:
        dp = days[k - 1]
        cost = min(oj(idx[a], idx[b]) for a in byday[dp] for b in byday[d])
        if d - dp != 1 or cost > Tq: s += 1; breaks.append((dp, d, round(cost, 2), byday[dp][:3], byday[d][:3]))
    seg[d] = s
print(f"DMT days {len(days)} | breaks {len(breaks)}:")
for b in breaks: print("   ", b)
out = T[["rec", "day"]].copy(); out["seg"] = out.day.map(seg)
os.makedirs(env.LOCAL, exist_ok=True); out.to_csv(os.path.join(env.LOCAL, "dmt2_days.csv"), index=False)
D11 = pd.read_csv(os.path.join(here, "dm11_calendar_v1.csv")).set_index("rec").day
print("\ncheck vs DM11 (descriptive only): breaks located between DM11 days:", [(D11.get(b[3][0]), D11.get(b[4][0])) for b in breaks])
