# -*- coding: utf-8 -*-
# vl3: 연결 선택 편향 없는 판단 — 같은 기록 안에서 (1) 밤 시간 1시간 변화 분포 (2) 하루 평균 - 23시 값 분포, 23시<.9 인데 평균>=1.1 비율
import sys
sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env
sys.path.insert(0, r"C:\work\farmai\집\클로드\research\linker_v4")
import numpy as np
from feat4 import EC, TRAIN, FARM
tr = np.where(TRAIN & np.isin(FARM, ["F13", "F47"]))[0]; E = EC[tr]
for h0, h1 in ((22, 23), (0, 1), (21, 22)):
    d = np.abs(E[:, h1] - E[:, h0]); d = d[~np.isnan(d)]; print(f"|dEC| {h0}->{h1}: q50 {np.quantile(d,.5):.3f} q99 {np.quantile(d,.99):.3f} max {d.max():.3f} share>=.25 {(d>=.25).mean():.4f}")
mm = np.nanmean(E, 1); e23 = E[:, 23]; e0 = E[:, 0]
dd = mm - e23; print("mean - EC23: q01/q50/q99", np.round(np.nanquantile(dd, [.01, .5, .99]), 3))
m = e23 < .9; print("EC23<.9: n", int(m.sum()), "share mean>=1.1", round(float((mm[m] >= 1.1).mean()), 4), "max mean", round(float(mm[m].max()), 3))
m = (e23 < .9) & (e0 < .9); print("EC0<.9 & EC23<.9: n", int(m.sum()), "share mean>=1.1", round(float((mm[m] >= 1.1).mean()), 4), "max mean", round(float(mm[m].max()), 3))
