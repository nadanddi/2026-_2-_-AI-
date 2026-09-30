# -*- coding: utf-8 -*-
"""재분석 11 (2026-10-01, 집 클로드): 평가일 실내 기록이 학습 자료 어딘가에 (거의) 그대로 있는가 — 탐색.
평가 60일(온실×날) 각각의 in_temp·in_hum·in_co2 24시간 벡터와 학습 51개 온실 모든 날(24행 완비)의
거리(시간별 RMSE, 변수별 표준화)를 재서 최근접 날을 찾는다. 비교 기준: 학습 F13·F47 날끼리의 최근접 거리.
외기는 같은 날짜 공유 구조(1.1)라 제외. 결과 local/re11_test_match_scan.txt
"""
import env  # noqa
import os, numpy as np, pandas as pd
X = pd.read_csv(os.path.join(env.DATA, "train_X.csv"))
T = pd.read_csv(os.path.join(env.DATA, "test_X.csv"))
Y = pd.read_csv(os.path.join(env.DATA, "train_y.csv"))
cols = ["in_temp", "in_hum", "in_co2"]
def daymat(df):
    k = df.row_id.str.extract(r"^(F\d+)_(\d+)_(\d+)$")
    df = df.assign(farm=k[0], day=k[1].astype(int), hour=k[2].astype(int))
    g = df.groupby(["farm", "day"])
    ok = g.hour.transform("size") == 24
    df = df[ok].sort_values(["farm", "day", "hour"])
    keys = df[["farm", "day"]].drop_duplicates().values
    M = df[cols].values.reshape(len(keys), 24, 3)
    return keys, M
kx, MX = daymat(X); kt, MT = daymat(T)
sd = np.nanstd(MX.reshape(-1, 3), 0)
def dist(A, B):  # A (n,24,3), B (m,24,3) → (n,m) 시간별 표준화 RMSE, 결측 무시
    A = A / sd; B = B / sd
    out = np.full((len(A), len(B)), np.nan)
    for i in range(len(A)):
        d = (B - A[i]) ** 2
        out[i] = np.sqrt(np.nanmean(d.reshape(len(B), -1), 1))
    return out
D = dist(MT, MX)
out = open(env.LOCAL + "/re11_test_match_scan.txt", "w", encoding="utf-8")
def p(s): print(s); out.write(s + "\n")
lab = set(Y.row_id.str[:7])
# 기준: 학습 F13/F47 날의 최근접(자기 제외)
sel = np.isin(kx[:, 0], ["F13", "F47"])
Dtt = dist(MX[sel], MX); 
for i, j in enumerate(np.where(sel)[0]): Dtt[i, j] = np.inf
base = np.nanmin(Dtt, 1)
p(f"기준: 학습 F13·F47 날 → 학습 전체 최근접 거리 중앙 {np.median(base):.3f}, 하위 5% {np.percentile(base,5):.3f}, 최소 {base.min():.3f}")
best = np.nanargmin(D, 1); bd = D[np.arange(len(D)), best]
p(f"평가 60일 → 학습 전체 최근접 거리 중앙 {np.median(bd):.3f}, 최소 {bd.min():.3f}")
p("평가일별 최근접 (거리 작은 순 상위 25):")
o = np.argsort(bd)
for i in o[:25]:
    j = best[i]
    p(f"  {kt[i][0]}_{kt[i][1]:03d} → {kx[j][0]}_{kx[j][1]:03d}  거리 {bd[i]:.3f}")
p(f"다른 온실이 최근접인 평가일 수: {int(sum(kx[best[i]][0] not in ('F13','F47') for i in range(len(bd))))}/60")
# 시간 이동 허용(±3h) 최근접도 확인: 일 경계 무시한 전체 시계열 슬라이딩은 생략, 결과가 흥미로우면 후속
