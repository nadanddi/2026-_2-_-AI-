# -*- coding: utf-8 -*-
"""SM1b — SM1 의 결합 약화가 '같은 센서 + 잡음'만으로 재현되는가 (반증 시뮬레이션) — 2026-10-08 연구실 클로드
깨끗한 F47 날(거칠지 않음)에 평균 0 흰잡음(in_temp σT, in_hum σH)을 더해, 거친 날의 차분 자기상관(in_temp .41, in_hum .27 수준)을 맞춘 뒤 K1~K4 를 다시 계산.
판독(사전): 잡음만으로 K1·K2·K4 의 거친 날 수준이 재현되고 K3(하루 평균 수준 차)는 안 바뀌면 → K1·K2·K4 차이는 잡음으로 설명, 'K3 차이'만이 다른 센서 근거로 남음.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", u"집", u"클로드", "research"))
import env  # noqa
import numpy as np, pandas as pd
R = os.path.dirname(os.path.abspath(__file__)); rng = np.random.default_rng(8)
K = pd.read_csv(os.path.join(R, "..", "results", "ec_sm1_days_v1.csv"))
TR = pd.read_csv(os.path.join(env.DATA, "train_X.csv")).merge(pd.read_csv(os.path.join(env.DATA, "train_y.csv")), on="row_id")
TR["farm"] = TR.row_id.str[:3]; TR["day"] = TR.row_id.str[4:7].astype(int); TR["hour"] = TR.row_id.str[8:10].astype(int)
TR = TR[(TR.farm == "F47") & TR.sub_ec.notna()].sort_values(["day", "hour"])
clean = set(K[(K.farm == "F47") & ~K.rough].day); AL = 1 - 0.5 ** (1 / 4)
mu, sd = K.loc[K.day < 179, "K3raw"].mean(), K.loc[K.day < 179, "K3raw"].std()
def ewm(x):
    o = np.empty_like(x); o[0] = x[0]
    for i in range(1, len(x)): o[i] = o[i - 1] + AL * (x[i] - o[i - 1])
    return o
def metrics(st, it, ih):
    e = ewm(it); m = np.arange(len(st)) >= 3
    k1 = np.corrcoef(st[m], e[m])[0, 1]; b = np.polyfit(e[m], st[m], 1); k2 = np.sqrt(np.mean((st[m] - np.polyval(b, e[m])) ** 2))
    k3 = abs((np.mean(st - it) - mu) / sd); k4 = np.corrcoef(it, ih)[0, 1]
    dt, dh = np.diff(it), np.diff(ih); ac = lambda d: np.corrcoef(d[:-1], d[1:])[0, 1]
    return k1, k2, k3, k4, ac(dt), ac(dh)
print("거친 날 실제(중앙값): K1 .951 K2 .541 K3 1.032 K4 -.725 | 차분 자기상관 in_temp .41 in_hum .27")
for sT, sH in [(0, 0), (.2, 1), (.3, 2), (.5, 3), (.8, 4)]:
    res = []
    for d, g in TR.groupby("day"):
        if d not in clean or g[["sub_temp", "in_temp", "in_hum"]].isna().any().any() or len(g) != 24: continue
        for _ in range(5):
            res.append(metrics(g.sub_temp.values, g.in_temp.values + rng.normal(0, sT, 24), g.in_hum.values + rng.normal(0, sH, 24)))
    r = np.median(np.array(res), axis=0)
    print(f"σT {sT:.1f} σH {sH:.0f}: K1 {r[0]:.3f} K2 {r[1]:.3f} K3 {r[2]:.3f} K4 {r[3]:+.3f} | 자기상관 T {r[4]:+.2f} H {r[5]:+.2f}")
