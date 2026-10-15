# -*- coding: utf-8 -*-
"""AC1 (2026-10-15 집 클로드; 사용자: '가장 쉬운 부분부터 차근차근').  사전 고정(커밋 후 실행).  EC.  학습 없음(저장 예측만).
문제: 27번 FL 앵커(p ≥ .8·정답 전날·|dT| ≤ 1.5)가 평가 60 기록 중 15 에만 걸림. p ≥ .5 로 넓히면 26.
기본 = NH(.9·pre_SEA 꼴 되살림, FL4 에서 EV3L-FL 기각 → 27번 EC 기본).  링크 = local/fl3_train_links.csv(FL, 학습 표).
  K8(현재) : p ≥ .8 & 정답 전날 & |dT_air| ≤ 1.5 → pred_h = q_h + (A − cm_h(q))
  S5(후보) : p ≥ .5 & 같은 조건 → pred_h = q_h + p·(A − cm_h(q))   (확률만큼만 옮김; 비율 = p 는 미리 고정, 튜닝 없음)
  앵커 전날이 그 폴드 검증 기록이면 사용 안 함(FL4 와 같음).
판정(EC 분야 규칙, 단일 안 α .025): S5 vs K8 — P2LOO·EL1 3시드 모두 2차 dSSE < 0 & (P2LOO 블록 P(악화) < .025 OR 나빠진 2차 기록-시드 0) & DIAG10 평균 dSSE 손해 아님.
서술: H5(p ≥ .5 단단히, 비율 1), S3(p ≥ .3, 비율 p), 새로 앵커 걸린 기록만의 변화, 각 안 vs 앵커 없음.
Run:  cd 집/클로드/research && PYTHONPATH="" python -u ac1_anchor_coverage_ec_v1.py
"""
import os, sys, pickle
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import env  # noqa: F401
import numpy as np, pandas as pd
src = open(os.path.join(HERE, "ev3l_ec_linked_yesterday_gru_v1.py"), encoding="utf-8").read(); __file__ = os.path.join(HERE, "ev3l_ec_linked_yesterday_gru_v1.py")
exec(src[:src.index("\nO = pd.read_csv")])
__file__ = os.path.join(HERE, "ac1_anchor_coverage_ec_v1.py")
FLT = pd.read_csv(os.path.join(env.LOCAL, "fl3_train_links.csv")); SEEDS = (1357, 2468, 3579)
_X = pd.concat([pd.read_csv(os.path.join(env.DATA, f)) for f in ("train_X.csv", "test_X.csv")]); _X = _X[_X.row_id.str[:3].isin(["F13", "F47"])]
_X["rec"] = _X.row_id.str[:7]; _X["h"] = _X.row_id.str[8:].astype(int); TIN = _X.pivot(index="rec", columns="h", values="in_temp")
def anchors(t):
    return {r: (p, q) for r, p, q in zip(FLT.rec, FLT.prev, FLT.p) if q >= t and p in idx and TRAIN[idx[p]] and np.isfinite(EC[idx[p], 23]) and abs(TIN.at[r, 0] - TIN.at[p, 23]) <= 1.5}
ARMS = {"K8": (anchors(.8), False), "S5": (anchors(.5), True), "H5": (anchors(.5), False), "S3": (anchors(.3), True)}
R = pd.read_csv(os.path.join(HERE, "nh1_rows_v1.csv.gz"))
MT = pickle.load(open(os.path.join(env.LOCAL, "nh1_mt1_v1.pkl"), "rb")); mt = np.array([MT[(v, k)][r][h] for v, k, r, h in zip(R.validator, R.fold, R.rec, R.hour)])
FOLDREC = {vk: set(G.rec) for vk, G in R.groupby(["validator", "fold"])}; NEWREC = set(ARMS["S5"][0]) - set(ARMS["K8"][0])
for s in SEEDS:
    R[f"NH_{s}"] = (R[f"A1_{s}"] - .1 * mt) / .9
    for name, (AN, soft) in ARMS.items():
        out = R[f"NH_{s}"].values.copy()
        for (v, k, r), g in R.groupby(["validator", "fold", "rec"]):
            if r not in AN: continue
            p, q = AN[r]
            if p in FOLDREC[(v, k)]: continue
            ii = g.sort_values("hour").index.values; x = R.loc[ii, f"NH_{s}"].values; pm = np.cumsum(x) / np.arange(1, len(x) + 1)
            out[ii] = x + (q if soft else 1.0) * (EC[idx[p], 23] - pm)
        R[f"{name}_{s}"] = out
rows, blocks, recd = [], [], []
for v in ("DIAG10", "EL1", "P2LOO"):
    g = R[R.validator == v]; y = g.y.values; p2 = g.p2.values; key = (g.farm + "_" + (g.day // 5).astype(str)).values; nw = g.rec.isin(NEWREC).values
    for s in SEEDS:
        f = lambda c, ref, m: 100 * (rm((g[c].values - y)[m]) / rm((g[ref].values - y)[m]) - 1) if m.any() else np.nan
        d = (g[f"S5_{s}"].values - y) ** 2 - (g[f"K8_{s}"].values - y) ** 2
        rows.append(dict(val=v, seed=s, dSSE_p2=float(d[p2].sum()), dSSE_all=float(d.sum()), S5_vs_K8_p2=f(f"S5_{s}", f"K8_{s}", p2), S5_vs_K8_newrec=f(f"S5_{s}", f"K8_{s}", nw),
                         H5_vs_K8_p2=f(f"H5_{s}", f"K8_{s}", p2), S3_vs_K8_p2=f(f"S3_{s}", f"K8_{s}", p2), K8_vs_NH_p2=f(f"K8_{s}", f"NH_{s}", p2), S5_vs_NH_p2=f(f"S5_{s}", f"NH_{s}", p2),
                         S5_vs_S_p2=f(f"S5_{s}", f"S_{s}", p2), S5_vs_SH_p2=f(f"S5_{s}", f"SH_{s}", p2)))
        if v == "P2LOO": blocks.append(pd.Series(d[p2]).groupby(key[p2]).mean())
        for r_, x in pd.Series(d[p2]).groupby(g.rec.values[p2]).sum().items(): recd.append(dict(val=v, seed=s, rec=r_, d=x))
S = pd.DataFrame(rows); pd.set_option("display.width", 260); print(S.round(3).to_string(index=False))
print("anchored train records:", {k: len(a) for k, (a, _) in ARMS.items()}, "| new under S5:", len(NEWREC))
main = S[S.val.isin(["P2LOO", "EL1"])]; c1 = bool((main.dSSE_p2 < 0).all()); x = pd.concat(blocks, axis=1).mean(axis=1).values
pP = float((x[np.random.default_rng(20270112).integers(0, len(x), (20000, len(x)))].mean(1) >= 0).mean()); RD = pd.DataFrame(recd)
nh_ = int((RD[RD.val.isin(["P2LOO", "EL1"])].groupby(["val", "seed", "rec"]).d.sum() > 1e-6).sum()); c3 = not (S[S.val == "DIAG10"].dSSE_all.mean() > 0)
print(f"AC1 S5 vs K8: cells {int((main.dSSE_p2 < 0).sum())}/6 | P2LOO P {pP:.4f} | harmed {nh_} | DIAG10 {S[S.val == 'DIAG10'].dSSE_all.mean():+.3f} -> {'PASS' if c1 and (pP < .025 or nh_ == 0) and c3 else 'FAIL'}")
