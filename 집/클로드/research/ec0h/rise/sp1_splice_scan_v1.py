# -*- coding: utf-8 -*-
"""SP1 (2026-10-12 집 클로드; 사용자: 평가 기록 이어 붙임 점검).  서술(진단 — 형제 비교 사용, 예측 특징 아님).
F47_178 형 흔적: 앞부분(0..k−1시) 실내 입력이 '날짜가 이어지지 않는' 앞번호 기록(n−1)의 23시에서 이어지고, k시부터 같은 날짜 형제와 비슷해짐.
기록마다:
  c_prev = n−1 기록 23시 → n 기록 0시 실내 3변수 표준화 차 합(작을수록 이어짐), adj = n−1 의 날짜가 n 날짜 − 1 인지(DM11)
  sib_early / sib_late = 0..3시 / 8..23시 실내의 형제 최소 거리 평균(같은 날짜 외기 90% 동일 기록)
  break_k = 실내 3변수 동시 변화가 가장 큰 시각(1..12)과 그 크기(같은 시각 분포 대비 z)
  night_gap = 0..5시 평균(실내 − 외기) — 겨울 난방 온실에서 크게 음수면 물리적으로 어색
흔적 점수: c_prev 작음(≤ 1.5) & adj 아님 & sib_early ≥ 2.5 & sib_late ≤ 1.5.  학습 기록은 정답(배지온·EC)이 앞부분에서 형제와 맞는지도 표시.
"""
import sys, os
sys.path.insert(0, r"C:\work\farmai\집\클로드\research\linker_v4"); sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env
from feat4 import *
rid = lambda i: f"{FARM[i]}_{NUM[i]:03d}"; idx = {rid(i): i for i in range(N)}
CAL = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "dm11_calendar_v1.csv")); DAY = dict(zip(CAL.rec, CAL.day))
day_of = np.array([DAY.get(rid(i), np.nan) for i in range(N)])
sd3 = np.nanstd(IN_.reshape(-1, 3), 0); Vz = IN_ / sd3
Xf = X.reshape(N, -1)
sibs = {i: [j for j in range(N) if j != i and (np.abs(Xf[j] - Xf[i]) < 1e-6).mean() >= .9] for i in range(N)}
dV = np.abs(np.diff(Vz, axis=1)).sum(2); zdV = (dV - np.nanmedian(dV, 0)) / (np.nanstd(dV, 0) + 1e-9)
rows = []
for i in range(N):
    p = idx.get(f"{FARM[i]}_{NUM[i]-1:03d}")
    c_prev = float(np.nansum(np.abs(Vz[p, 23] - Vz[i, 0]))) if p is not None else np.nan
    adj = (p is not None) and np.isfinite(day_of[p]) and np.isfinite(day_of[i]) and day_of[p] in (day_of[i] - 1, day_of[i])
    if sibs[i]:
        d = np.nanmin(np.stack([np.nansum(np.abs(Vz[i] - Vz[j]), 1) for j in sibs[i]]), 0)
        se, sl = float(np.nanmean(d[:4])), float(np.nanmean(d[8:]))
    else: se = sl = np.nan
    k = int(np.nanargmax(zdV[i, :12])) + 1; kz = float(np.nanmax(zdV[i, :12]))
    ng = float(np.nanmean(IN_[i, :6, 0] - X[i, :6, 0]))
    lab_early = np.nan
    if TRAIN[i] and sibs[i]:
        ls = [j for j in sibs[i] if TRAIN[j]]
        if ls: lab_early = float(np.nanmin([np.nanmean(np.abs(ST[i, :4] - ST[j, :4])) for j in ls]))
    flag = bool(np.isfinite(c_prev) and c_prev <= 1.5 and not adj and np.isfinite(se) and se >= 2.5 and sl <= 1.5)
    rows.append(dict(rec=rid(i) + ("" if TRAIN[i] else "*"), split="train" if TRAIN[i] else "eval", c_prev=c_prev, prev_adjacent=adj, sib_early=se, sib_late=sl,
                     break_h=k, break_z=kz, night_in_minus_out=ng, label_early_sib_diff=lab_early, flag=flag))
D = pd.DataFrame(rows); D.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "sp1_scan_v1.csv"), index=False)
pd.set_option("display.width", 250)
print("FLAGGED (F47_178-type):"); print(D[D.flag].round(2).to_string(index=False))
print("\nnear-flag (any 3 of 4 conditions) or early-far-from-siblings & continuous with non-adjacent prev:")
cond = pd.DataFrame(dict(a=(D.c_prev <= 1.5), b=~D.prev_adjacent, c=(D.sib_early >= 2.5), d=(D.sib_late <= 1.5)))
near = D[(cond.sum(1) >= 3) & ~D.flag & (D.sib_early >= 2)]
print(near.sort_values("sib_early", ascending=False).round(2).to_string(index=False))
print("\neval records: early-hours far from siblings (sib_early >= 2):")
print(D[(D.split == "eval") & (D.sib_early >= 2)].sort_values("sib_early", ascending=False).round(2).to_string(index=False))
print("\nnight indoor colder than outdoor by > 3C (0..5h mean):", D[D.night_in_minus_out < -3][["rec", "night_in_minus_out", "c_prev", "prev_adjacent", "sib_early"]].round(2).to_string(index=False))
