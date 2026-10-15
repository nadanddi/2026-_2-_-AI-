# -*- coding: utf-8 -*-
"""AC2 (2026-10-15 집 클로드; 사용자: AC1 후 '그렇게 진행해봐').  사전 고정(커밋 후 실행).  EC.  학습 없음.
AC1(6.9xx) S5 는 2차 모든 판 개선이나 DIAG10 1차 손해(F47_132·F47_097: 고EC 날에 틀린 링크, 앵커 A 가 모델 0시 예측과 크게 어긋남).
안전장치(현장 규칙: '어제 마지막 EC 와 오늘 모델 0시 판단이 너무 다르면 링크를 의심'):  |A − q_0| > τ 이면 그 앵커 안 씀 (q_0 = NH 0시 예측; 0시에 알 수 있음).
  S5G(τ) = S5(p ≥ .5, 비율 p) + 안전장치.  τ 격자 {0.15, 0.25, 0.4, 0.6, 1.0, ∞}.
선택: P2LOO+EL1 2차 dSSE(S5G − K8, 3시드 합) 최소 τ* (같으면 큰 τ).  판정 판 = DIAG10 (단, AC1 에서 손해 기록을 이미 봄 — 사후 성격, 설명자료에 밝힘).
판정(α .025): τ* 에서 (1) P2LOO·EL1 3시드 2차 dSSE < 0, (2) DIAG10 3시드 전체 dSSE < 0 & DIAG10 블록(농장 x 5일) P(악화) < .025.
서술: τ 별 표, 안전장치에 걸린 기록, 각 안 vs NH.
Run:  cd 집/클로드/research && PYTHONPATH="" python -u ac2_anchor_guard_ec_v1.py
"""
import os, sys, pickle
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import env  # noqa: F401
import numpy as np, pandas as pd
src = open(os.path.join(HERE, "ev3l_ec_linked_yesterday_gru_v1.py"), encoding="utf-8").read(); __file__ = os.path.join(HERE, "ev3l_ec_linked_yesterday_gru_v1.py")
exec(src[:src.index("\nO = pd.read_csv")])
__file__ = os.path.join(HERE, "ac2_anchor_guard_ec_v1.py")
FLT = pd.read_csv(os.path.join(env.LOCAL, "fl3_train_links.csv")); SEEDS = (1357, 2468, 3579)
_X = pd.concat([pd.read_csv(os.path.join(env.DATA, f)) for f in ("train_X.csv", "test_X.csv")]); _X = _X[_X.row_id.str[:3].isin(["F13", "F47"])]
_X["rec"] = _X.row_id.str[:7]; _X["h"] = _X.row_id.str[8:].astype(int); TIN = _X.pivot(index="rec", columns="h", values="in_temp")
def anchors(t):
    return {r: (p, q) for r, p, q in zip(FLT.rec, FLT.prev, FLT.p) if q >= t and p in idx and TRAIN[idx[p]] and np.isfinite(EC[idx[p], 23]) and abs(TIN.at[r, 0] - TIN.at[p, 23]) <= 1.5}
ARMS = {"K8": (anchors(.8), False), "S5": (anchors(.5), True), "H5": (anchors(.5), False), "S3": (anchors(.3), True)}
TAUS = (0.15, 0.25, 0.4, 0.6, 1.0, np.inf)
R = pd.read_csv(os.path.join(HERE, "nh1_rows_v1.csv.gz"))
MT = pickle.load(open(os.path.join(env.LOCAL, "nh1_mt1_v1.pkl"), "rb")); mt = np.array([MT[(v, k)][r][h] for v, k, r, h in zip(R.validator, R.fold, R.rec, R.hour)])
FOLDREC = {vk: set(G.rec) for vk, G in R.groupby(["validator", "fold"])}; A8, A5 = anchors(.8), anchors(.5); guarded = {}
GRP = list(R.groupby(["validator", "fold", "rec"]))
for s in SEEDS:
    R[f"NH_{s}"] = (R[f"A1_{s}"] - .1 * mt) / .9
    outs = {"K8": R[f"NH_{s}"].values.copy(), **{t: R[f"NH_{s}"].values.copy() for t in TAUS}}
    for (v, k, r), g in GRP:
        ii = g.sort_values("hour").index.values; x = R.loc[ii, f"NH_{s}"].values; pm = np.cumsum(x) / np.arange(1, len(x) + 1)
        if r in A8 and A8[r][0] not in FOLDREC[(v, k)]: outs["K8"][ii] = x + (EC[idx[A8[r][0]], 23] - pm)
        if r in A5 and A5[r][0] not in FOLDREC[(v, k)]:
            p, q = A5[r]; A = EC[idx[p], 23]
            for t in TAUS:
                if abs(A - x[0]) <= t: outs[t][ii] = x + q * (A - pm)
                elif s == SEEDS[0]: guarded.setdefault(t, set()).add((v, r))
    R[f"K8_{s}"] = outs["K8"]
    for t in TAUS: R[f"G{t}_{s}"] = outs[t]
def dsse(v, c, s, m="p2"):
    g = R[R.validator == v]; y = g.y.values; d = (g[f"{c}_{s}"].values - y) ** 2 - (g[f"K8_{s}"].values - y) ** 2; return d[g.p2.values] if m == "p2" else d
tab = []
for t in TAUS:
    sel = sum(dsse(v, f"G{t}", s).sum() for v in ("P2LOO", "EL1") for s in SEEDS); dg = np.mean([dsse("DIAG10", f"G{t}", s, "all").sum() for s in SEEDS])
    tab.append(dict(tau=t, sel_P2LOO_EL1_dSSE=sel, DIAG10_dSSE_all=dg, n_guarded=len(guarded.get(t, ())))); 
TB = pd.DataFrame(tab); print(TB.round(3).to_string(index=False))
best = TB.sel_P2LOO_EL1_dSSE.min(); ts = TB[TB.sel_P2LOO_EL1_dSSE <= best + 1e-9].tau.max(); print("tau* =", ts, "| guarded at tau*:", sorted(guarded.get(ts, ())))
c1 = all(dsse(v, f"G{ts}", s).sum() < 0 for v in ("P2LOO", "EL1") for s in SEEDS); c2a = all(dsse("DIAG10", f"G{ts}", s, "all").sum() < 0 for s in SEEDS)
g = R[R.validator == "DIAG10"]; key = (g.farm + "_" + (g.day // 5).astype(str)).values
x = pd.concat([pd.Series(dsse("DIAG10", f"G{ts}", s, "all")).groupby(key).mean() for s in SEEDS], axis=1).mean(axis=1).values
pP = float((x[np.random.default_rng(20270113).integers(0, len(x), (20000, len(x)))].mean(1) >= 0).mean())
for v in ("DIAG10", "EL1", "P2LOO"):
    gg = R[R.validator == v]; y = gg.y.values; m = gg.p2.values
    print(v, " ".join(f"s{s}: vsK8 p2 {100 * (rm((gg[f'G{ts}_{s}'].values - y)[m]) / rm((gg[f'K8_{s}'].values - y)[m]) - 1):+.2f}% all {100 * (rm(gg[f'G{ts}_{s}'].values - y) / rm(gg[f'K8_{s}'].values - y) - 1):+.2f}% vsNH all {100 * (rm(gg[f'G{ts}_{s}'].values - y) / rm(gg[f'NH_{s}'].values - y) - 1):+.2f}%" for s in SEEDS))
print(f"AC2 S5G(tau {ts}) vs K8: P2LOO/EL1 cells {'6/6' if c1 else 'not all'} | DIAG10 all seeds better {c2a} | DIAG10 P {pP:.4f} -> {'PASS' if c1 and c2a and pP < .025 else 'FAIL'}")
