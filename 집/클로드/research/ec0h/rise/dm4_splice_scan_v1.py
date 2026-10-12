# -*- coding: utf-8 -*-
"""DM4 (2026-10-12 집 클로드, 서술): 기록 안 '이어 붙임' 흔적 전수 스캔 (460 기록).
흔적 정의(사전): 시각 h(1..23)에서 실내 변수(in_temp, in_hum, in_co2)와 구동기(보온·차광·난방·팬)의 동시 급변 점수
  S_h = Σ_v |Δv_h| / sd_v(같은 시각 전체 기록의 Δ 표준편차)  — 외기 급변이 큰 시각은 외기 점수 O_h 로 따로 표시.
  기록 점수 = max_h (S_h − O_h); 상위 기록을 나열하고, 그 시각 전후로 (a) 앞 기록 23시와의 연속(앞쪽 조각이 앞 기록에서 이어짐?),
  (b) 같은 날짜 형제와의 거리 변화(이어 붙인 뒤쪽이 형제와 맞는지)를 표시.  정답(학습)은 배지온·EC 의 같은 시각 급변도 함께 표시.
"""
import sys, os
sys.path.insert(0, r"C:\work\farmai\집\클로드\research\linker_v4"); sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env
from feat4 import *
rid = lambda i: f"{FARM[i]}_{NUM[i]:03d}" + ("" if TRAIN[i] else "*"); idx = {f"{FARM[i]}_{NUM[i]:03d}": i for i in range(N)}
V = np.concatenate([IN_, AC[:, :, 1:5]], 2)          # in_temp,in_hum,in_co2, shade,thermal,heating,circfan
dV = np.abs(np.diff(V, axis=1)); sdV_h = np.nanstd(dV, 0) + 1e-6           # per hour, per var
S = np.nansum(dV / sdV_h, 2)                                               # [N, 23]
dO = np.abs(np.diff(X, axis=1)); O = np.nansum(dO / (np.nanstd(dO, 0) + 1e-6), 2)
sc = S - O; h_star = np.nanargmax(sc, 1) + 1; best = np.nanmax(sc, 1)
# typical: hour-specific distribution
z = (sc - np.nanmedian(sc, 0)) / (np.nanstd(sc, 0) + 1e-6); zb = np.nanmax(z, 1)
Xf = X.reshape(N, -1)
sibs = {i: [j for j in range(N) if j != i and (np.abs(Xf[j] - Xf[i]) < 1e-6).mean() >= .9] for i in range(N)}
def prevrec(i):
    c = [j for j in range(N) if FARM[j] == FARM[i] and NUM[j] == NUM[i] - 1]; return c[0] if c else None
rows = []
for i in np.argsort(-zb)[:25]:
    h = h_star[i]; p = prevrec(i)
    cont = np.nansum(np.abs(V[p, 23, :3] - V[i, 0, :3]) / np.nanstd(V[:, 0, :3], 0)) if p is not None else np.nan
    # sibling distance before vs after the splice hour (indoor 3 vars)
    sd3 = np.nanstd(V[:, :, :3].reshape(-1, 3), 0)
    def sdist(a, b, hs): return float(np.nanmean(np.abs(V[a][hs, :3] - V[b][hs, :3]) / sd3))
    sb = [(sdist(i, j, slice(0, h)), sdist(i, j, slice(h, 24))) for j in sibs[i]]
    sbtxt = ", ".join(f"{rid(j)} {a:.1f}->{b:.1f}" for j, (a, b) in zip(sibs[i], sb))
    lab = f" | dST {ST[i,h]-ST[i,h-1]:+.1f} dEC {EC[i,h]-EC[i,h-1]:+.2f}" if TRAIN[i] else ""
    rows.append(f"  {rid(i):9s} z {zb[i]:5.1f} at {h:2d}h | in_temp {IN_[i,h-1,0]:.1f}->{IN_[i,h,0]:.1f} hum {IN_[i,h-1,1]:.0f}->{IN_[i,h,1]:.0f} co2 {IN_[i,h-1,2]:.0f}->{IN_[i,h,2]:.0f}{lab} | prev-rec 23h->0h indoor jump {cont:.1f} | siblings (dist before->after splice) {sbtxt}")
print("top 25 records by within-record simultaneous indoor/actuator jump not explained by outdoor (z over same-hour distribution):")
print("\n".join(rows))
print("\nz distribution: 99%", np.round(np.nanquantile(zb, .99), 1), "95%", np.round(np.nanquantile(zb, .95), 1), "median", np.round(np.nanmedian(zb), 1))
pd.DataFrame(dict(rec=[rid(i) for i in range(N)], z=zb, hour=h_star)).sort_values("z", ascending=False).to_csv("dm4_splice_scores_v1.csv", index=False)
