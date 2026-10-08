# -*- coding: utf-8 -*-
"""SM2 — '실내 입력이 다른 센서' 직접 검사: 같은 날짜 묶음 교차 대조 + 환기 공변량 (SM1 최종 비평 권고, 실행 전 고정) — 2026-10-08 연구실 클로드
(1) 각 정답 날 d 의 sub_temp 를, 같은 날짜 묶음(외기 24h 근사일치 ≤.05, 학습·평가 기록 모두)의 다른 기록 r 의 in_temp EWMA(반감기 4h)와 결합(K1, t≥3)시켜
    own K1 과 비교. '바뀜 신호' = max_r K1(d, r) > own K1 + .02. 사전 예측(가설 참): 거친 날의 바뀜 비율 > 거친 않은 날, Fisher 단측 p < .01.
(2) 환기 공변량: K1(Fisher z) ~ rough + 하루 환기>0 비율 + 순환팬 평균 + 차광 평균 (OLS), rough 계수가 남는지(p < .01).
(3) 온도 영향: DIAG10 sub_temp 오차는 저장 OOF 가 없으면 생략(서술).
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", u"집", u"클로드", "research"))
import env  # noqa
import numpy as np, pandas as pd
from scipy.stats import fisher_exact
from scipy.stats import norm
R = os.path.dirname(os.path.abspath(__file__))
def load(fn):
    d = pd.read_csv(os.path.join(env.DATA, fn)); d["farm"] = d.row_id.str[:3]; d["day"] = d.row_id.str[4:7].astype(int); d["hour"] = d.row_id.str[8:10].astype(int); return d
X = pd.concat([load("train_X.csv").merge(pd.read_csv(os.path.join(env.DATA, "train_y.csv")), on="row_id", how="left"), load("test_X.csv")])
X = X[X.farm.isin(["F13", "F47"])].sort_values(["farm", "day", "hour"])
W = ["out_temp", "out_hum", "out_wspd", "out_rad"]
WV = X.pivot_table(index=["farm", "day"], columns="hour", values=W); WZ = (WV - WV.mean()) / WV.std()
keys = list(WZ.index); A = WZ.values; Dm = np.sqrt(np.nanmean((A[:, None, :] - A[None, :, :]) ** 2, axis=2))
AL = 1 - 0.5 ** (1 / 4)
def ewm(x):
    x = pd.Series(x).ffill().bfill().values; o = np.empty(len(x)); o[0] = x[0]
    for i in range(1, len(x)): o[i] = o[i - 1] + AL * (x[i] - o[i - 1])
    return o
IT = {k: g.set_index("hour").in_temp.reindex(range(24)).values for k, g in X.groupby(["farm", "day"])}
ST = {k: g.set_index("hour").sub_temp.reindex(range(24)).values for k, g in X.groupby(["farm", "day"]) if g.sub_temp.notna().sum() >= 20}
def k1(st, it):
    e = ewm(it); m = (np.arange(24) >= 3) & ~np.isnan(st) & ~np.isnan(e)
    return np.corrcoef(st[m], e[m])[0, 1] if m.sum() > 8 else np.nan
K = pd.read_csv(os.path.join(R, "..", "results", "ec_sm1_days_v1.csv")).set_index(["farm", "day"])
rows = []
for i, k in enumerate(keys):
    if k not in ST or k not in K.index: continue
    own = k1(ST[k], IT[k]); nb = [keys[j] for j in np.where(Dm[i] <= .05)[0] if keys[j] != k]
    oth = [(r, k1(ST[k], IT[r])) for r in nb]
    best = max(oth, key=lambda t: t[1]) if oth else (None, np.nan)
    rows.append((k[0], k[1], own, len(nb), best[0], best[1], bool(K.loc[k, "rough"])))
M = pd.DataFrame(rows, columns=["farm", "day", "own", "nnb", "best_rec", "best", "rough"])
H = M[M.nnb > 0].copy(); H["swap"] = H.best > H.own + .02
print(f"같은 날짜 묶음 짝이 있는 정답 날 {len(H)} / {len(M)} (거친 {int(H.rough.sum())}/{int(M.rough.sum())})")
t = pd.crosstab(H.rough, H.swap); print(t)
if t.shape == (2, 2):
    print("Fisher 단측(거친 날에 바뀜 많음) p", round(fisher_exact(t.values[::-1], alternative="greater")[1], 4))
print("거친 날:", [(f"{r.farm}_{r.day}", round(r.own, 3), f"{r.best_rec[0]}_{r.best_rec[1]}" if r.best_rec else None, round(r.best, 3)) for r in H[H.rough].itertuples()])
print("바뀜 신호 날(전체):", [(f"{r.farm}_{r.day}", round(r.own, 3), f"{r.best_rec[0]}_{r.best_rec[1]}", round(r.best, 3), 'R' if r.rough else '') for r in H[H.swap].itertuples()])
print(f"own − best 중앙값: 거친 {(H[H.rough].own - H[H.rough].best).median():+.3f}, 나머지 {(H[~H.rough].own - H[~H.rough].best).median():+.3f}")
# (2) 환기 공변량
g = X[X.sub_ec.notna()].groupby(["farm", "day"]).agg(vent=("act_vent", lambda s: (s > 0).mean()), fan=("act_circfan", "mean"), shade=("act_shade", "mean"))
D = K.join(g, how="inner").dropna(subset=["K1"]); D["z"] = np.arctanh(D.K1.clip(-.999, .999))
for cols in (["rough"], ["rough", "vent", "fan", "shade"], ["rough", "vent", "fan", "shade", "f47"]):
    D["f47"] = (D.index.get_level_values(0) == "F47").astype(float)
    Xm = np.column_stack([np.ones(len(D))] + [D[c].astype(float).values for c in cols]); y = D.z.values
    b = np.linalg.lstsq(Xm, y, rcond=None)[0]; e = y - Xm @ b; n, k = Xm.shape; XtXi = np.linalg.inv(Xm.T @ Xm)
    V = XtXi @ (Xm.T * e ** 2) @ Xm @ XtXi * n / (n - k); se = np.sqrt(np.diag(V)); pv = 2 * norm.sf(np.abs(b / se))
    P = dict(zip(["c"] + cols, zip(b, pv)))
    print(f"K1(z) ~ {'+'.join(cols)}: rough 계수 {P['rough'][0]:+.3f} p {P['rough'][1]:.2e} | " + " ".join(f"{c} {P[c][0]:+.3f}(p {P[c][1]:.1e})" for c in cols[1:]))
print("거친 날 vs 나머지 환기>0 비율 중앙", round(D[D.rough].vent.median(), 2), round(D[~D.rough].vent.median(), 2), "| 팬", round(D[D.rough].fan.median(), 1), round(D[~D.rough].fan.median(), 1))
M.to_csv(os.path.join(R, "..", "results", "ec_sm2_cross_v1.csv"), index=False)
