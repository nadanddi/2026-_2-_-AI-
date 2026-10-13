# -*- coding: utf-8 -*-
"""DG1 (2026-10-14 집 클로드; 사용자: '2번(빈 날 감쇠 측정)부터').  서술(판정 아님).  EC.
'g 일 전 같은 동 값이 오늘을 얼마나 알려주나' — RNN 상태 감쇠·앵커 사용 일수의 설계값.
같은 동 쌍: CP2 트랙의 1일 연결(cert 기반, 학습 정답 연속) 만 이어 만든 사슬 안에서 날짜 차 g = 1..6 인 (a, b) — 확실한 쌍.  비교: 빈 날 연결(지문 추정) 쌍.
기본 예측 = DIAG10 CUR OOF(시드 평균).  잠금 날 제외.
지표(g 별): 하루 평균 |dEC|; 지속 계수 = b 하루 평균을 a 하루 평균에 회귀한 기울기·R2;
  최적 당김 w_g = argmin_w sum_h (y_bh − (base_bh + w (A − cm_bh)))^2, A = a 의 23시 EC, cm = base 0..h 누적평균(EA2 식) — 닫힌 해;
  w_g 로 당겼을 때와 w = 1(완전 당김)·0(기본) 의 RMSE.  1차/2차(b 번호 179 기준), 보통/고EC(b) 별.
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import env  # noqa: F401
import numpy as np, pandas as pd
T = pd.read_csv(os.path.join(HERE, "cp2_tracks_v1.csv")); A = pd.read_csv(os.path.join(HERE, "cp2_rows_v1.csv.gz"))
A = A[A.is_train & ~A.lock].sort_values(["rec", "hour"])
Y = {r: g.y.values for r, g in A.groupby("rec")}; B = {r: g.base_oof.values for r, g in A.groupby("rec")}
T = T[T.train & ~T.lock].set_index("rec"); rm = lambda e: float(np.sqrt(np.mean(np.square(e))))
# certain chains: follow day1 links only
nxt1 = {p: r for r, p, lt in zip(T.index, T.prev, T.link_type) if lt == "day1" and isinstance(p, str) and p in T.index}
prv1 = {v: k for k, v in nxt1.items()}
chains = []
for r in T.index:
    if r in prv1: continue
    c = [r]
    while c[-1] in nxt1: c.append(nxt1[c[-1]])
    chains.append(c)
pairs = []
for c in chains:
    for i in range(len(c)):
        for j in range(i + 1, min(len(c), i + 7)):
            a, b = c[i], c[j]; g = int(T.at[b, "day"] - T.at[a, "day"]) if np.isfinite(T.at[b, "day"]) and np.isfinite(T.at[a, "day"]) else j - i
            pairs.append(dict(a=a, b=b, g=g, kind="certain"))
for r, p, lt, gd in zip(T.index, T.prev, T.link_type, T.gap_days):
    if lt == "gap" and isinstance(p, str) and p in T.index: pairs.append(dict(a=p, b=r, g=int(gd), kind="gap-link"))
P = pd.DataFrame(pairs); P = P[P.a.isin(Y) & P.b.isin(Y)]
P["p2"] = P.b.str[4:].astype(int) >= 179; P["hi"] = [np.nanmean(Y[b]) >= 1.0 for b in P.b]
def summarize(G):
    if len(G) < 3: return None
    da = np.array([np.nanmean(Y[a]) for a in G.a]); db = np.array([np.nanmean(Y[b]) for b in G.b])
    sl = np.polyfit(da, db, 1)[0]; r2 = np.corrcoef(da, db)[0, 1] ** 2
    num = den = 0.0; e0, e1, ew = [], [], []
    for a, b in zip(G.a, G.b):
        y, base = Y[b], B[b]; m = np.isfinite(y) & np.isfinite(base)
        cm = pd.Series(base).expanding().mean().values; Aa = Y[a][23] if np.isfinite(Y[a][23]) else np.nanmean(Y[a]); d = (Aa - cm)
        num += np.sum((y - base)[m] * d[m]); den += np.sum(d[m] ** 2)
    w = num / den if den > 0 else np.nan
    for a, b in zip(G.a, G.b):
        y, base = Y[b], B[b]; m = np.isfinite(y) & np.isfinite(base); cm = pd.Series(base).expanding().mean().values; Aa = Y[a][23] if np.isfinite(Y[a][23]) else np.nanmean(Y[a])
        e0.append((base - y)[m]); e1.append((base + (Aa - cm) - y)[m]); ew.append((base + w * (Aa - cm) - y)[m])
    return dict(n=len(G), mean_abs_dEC=float(np.mean(np.abs(db - da))), slope=sl, R2=r2, w_opt=w, rmse_base=rm(np.concatenate(e0)), rmse_full=rm(np.concatenate(e1)), rmse_wopt=rm(np.concatenate(ew)))
rows = []
for (kind, g), G in P.groupby(["kind", "g"]):
    s = summarize(G)
    if s: rows.append(dict(kind=kind, g=g, subset="all", **s))
    for nm, m in (("pass-1", ~G.p2), ("pass-2", G.p2), ("normal", ~G.hi), ("high", G.hi)):
        s = summarize(G[m])
        if s: rows.append(dict(kind=kind, g=g, subset=nm, **s))
S = pd.DataFrame(rows); pd.set_option("display.width", 220)
print(S[S.subset == "all"].round(3).to_string(index=False))
print("\nby subset (certain pairs):"); print(S[(S.kind == "certain") & (S.subset != "all")].pivot_table(index="g", columns="subset", values="w_opt").round(2).to_string())
print(S[(S.kind == "certain") & (S.subset != "all")].pivot_table(index="g", columns="subset", values="n").to_string())
S.to_csv(os.path.join(HERE, "dg1_gap_decay_v1.csv"), index=False)
