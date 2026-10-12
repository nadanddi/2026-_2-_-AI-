# -*- coding: utf-8 -*-
"""DM5 (2026-10-12 집 클로드, 서술): 같은 날짜 형제 대비 '이어 붙임' 스캔.  기록 i 와 같은 날짜(외기 90% 동일) 형제 j 들에 대해
실내 3변수(in_temp, in_hum, in_co2; 표준화) 시각별 거리 d_t = min_j |V_i,t − V_j,t|.  분할 시각 h(2..22)마다 앞(0..h−1)·뒤(h..23) 평균.
흔적(사전): 한쪽 평균 >= 3 이고 다른 쪽 <= 1.2, 그리고 차이가 최대인 h.  추가로 '외기 날짜 자체가 번호 위치와 어긋남'(DM2 의 F47_178·F47_074).
또 앞 번호 기록 23시→0시 실내 연속(이어 붙은 앞조각이 앞 기록에서 온 것인지)도 표시."""
import sys
sys.path.insert(0, r"C:\work\farmai\집\클로드\research\linker_v4"); sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env
from feat4 import *
rid = lambda i: f"{FARM[i]}_{NUM[i]:03d}" + ("" if TRAIN[i] else "*")
V = IN_.copy(); sd = np.nanstd(V.reshape(-1, 3), 0); Vz = V / sd
Xf = X.reshape(N, -1)
sibs = {i: [j for j in range(N) if j != i and (np.abs(Xf[j] - Xf[i]) < 1e-6).mean() >= .9] for i in range(N)}
def prevrec(i):
    c = [j for j in range(N) if FARM[j] == FARM[i] and NUM[j] == NUM[i] - 1]; return c[0] if c else None
out = []
for i in range(N):
    if not sibs[i]: continue
    d = np.nanmin(np.stack([np.nansum(np.abs(Vz[i] - Vz[j]), 1) for j in sibs[i]]), 0)   # [24]
    best = None
    for h in range(2, 23):
        a, b = np.nanmean(d[:h]), np.nanmean(d[h:])
        if (a >= 3 and b <= 1.2) or (b >= 3 and a <= 1.2):
            if best is None or abs(a - b) > abs(best[1] - best[2]): best = (h, a, b)
    if best:
        p = prevrec(i); cont = float(np.nansum(np.abs(Vz[p, 23] - Vz[i, 0]))) if p is not None else np.nan
        out.append((rid(i), best[0], best[1], best[2], cont, ", ".join(rid(j) for j in sibs[i]), " ".join(f"{x:.1f}" for x in d)))
print(f"records with a sibling-distance split (one side >=3, other <=1.2): {len(out)}")
for r in sorted(out, key=lambda x: -abs(x[2] - x[3])):
    print(f"  {r[0]:9s} split {r[1]:2d}h | dist before {r[2]:.1f} after {r[3]:.1f} | prev-rec 23h->0h indoor jump {r[4]:.1f} | siblings {r[5]}")
    print(f"            hourly min-dist: {r[6]}")
# whole-day outliers: records far from all siblings all day (different house state or misdated indoor)
far = []
for i in range(N):
    if not sibs[i]: continue
    d = np.nanmin(np.stack([np.nansum(np.abs(Vz[i] - Vz[j]), 1) for j in sibs[i]]), 0)
    far.append((np.nanmean(d), rid(i)))
far.sort(reverse=True)
print("\nwhole-day farthest from all same-date siblings (mean min-dist; median of all records %.2f):" % np.median([f[0] for f in far]))
print("  " + ", ".join(f"{r} {m:.1f}" for m, r in far[:20]))
