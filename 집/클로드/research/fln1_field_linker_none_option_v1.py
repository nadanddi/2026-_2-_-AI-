# -*- coding: utf-8 -*-
"""FLN1 (2026-10-15 집 클로드; 사용자: '그렇게 진행해봐' — 추가 실험 1번).  사전 고정(커밋 후 실행).  링커 + EC.
FL(현장형 링커: oj, v0~v7, excl, xf) 에 '전날 없음' 선택지를 더한 FLN.  현장 근거: 어제 기록이 로그에 없을 수 있음.
'없음' 학습(현장형 모의, 대회 배치 흉내 없음): 정답 짝(train_groups7 과 같은 cert 짝·숨김) 마다 두 묶음 —
  (n) 그대로: 정답 = 진짜 전날,  (d) 진짜 전날을 후보에서 지움(removed 에 추가): 정답 = 없음.
없음 줄의 특징(후보 줄은 0): isnone=1, min_oj(후보 최소 외기 비용), min_vs(후보 최소 실내·구동기 거리 합 v0..v7), gap(1·2등 FL 점수 차는 순환이라 안 씀 → 대신 log 후보 수).
  후보 줄의 FL 특징은 없음 줄에서 0.  조건부 로짓(fit_clogit l2 .5, iters 300) — FL 과 같은 설정.
1단계(서술 + 관문, 블록 5겹 CV: 농장 x NUM//20, FL2 acc_cv 와 같은 나눔):
  (n) 묶음: FL·FLN p ≥ .8 / p ≥ .5 포괄(링크 낸 비율)·정밀도(낸 링크 중 정답),  (d) 묶음: p ≥ .8 / .5 헛링크 비율(무엇이든 링크하면 틀림).
  관문(모두): FLN p ≥ .5 정밀도(n) ≥ FL p ≥ .8 정밀도(n) − .02  &  FLN p ≥ .5 헛링크(d) ≤ FL p ≥ .8 헛링크(d)  &  FLN p ≥ .5 포괄(n) > FL p ≥ .8 포괄(n).
  (= 'FLN 을 p ≥ .5 로 넓혀도 지금 K8 만큼 안전하고 더 많이 잇는다')
2단계(관문 통과 시에만): 학습표(fl3 와 같은 규칙: 대상 같은 농장 ±1 정답 숨김, 평가 후보 제외)를 FLN 으로 만들고,
  EC: NH + FLN 앵커(p_link ≥ .5, 이동량 x p, 정답 전날·|dT| ≤ 1.5)  vs  NH + FL K8 — AC1 과 같은 판정(EC 분야 규칙: P2LOO·EL1 3시드 2차 dSSE < 0 &
  (P2LOO 블록 P < .025 OR 손해 기록-칸 0) & DIAG10 전체 dSSE 손해 아님).
Run:  cd 집/클로드/research && PYTHONPATH="" python -u fln1_field_linker_none_option_v1.py
"""
import os, sys, pickle
HERE = os.path.dirname(os.path.abspath(__file__)); sys.argv = ["x"]
sys.path.insert(0, os.path.join(HERE, "linker_v7")); sys.path.insert(0, HERE)
import env  # noqa: F401
from feat7 import *  # noqa: F401,F403
import numpy as np, pandas as pd
FL = ["oj"] + [f"v{j}" for j in range(8)] + ["excl", "xf"]; NX = ["isnone", "min_oj", "min_vs", "logn"]; COLS = FL + NX
def mat(df):
    C = np.zeros((len(df) + 1, len(COLS))); C[:-1, :len(FL)] = df[FL].values.astype(float)
    vs = df[[f"v{j}" for j in range(8)]].values.astype(float).sum(1); C[-1, len(FL):] = [1.0, df.oj.min(), vs.min(), np.log(len(df))]
    return C   # last row = none
groups = []
for b, a in zip(TL7.b, TL7.a):
    lab = TRAIN.copy(); lab[(FARM == FARM[b]) & (np.abs(NUM - NUM[b]) <= 1)] = False
    for mode in ("n", "d"):
        rem = set(TEST_IDX) | ({int(a)} if mode == "d" else set())
        df, _ = features7(b, 0, lab, removed=rem)
        if len(df) == 0: continue
        if mode == "n":
            y = np.where(df.a.values == a)[0]
            if not len(y): continue
            y = int(y[0])
        else: y = len(df)
        groups.append(dict(b=int(b), a=int(a), mode=mode, C=mat(df), y=y, cand=df.a.values.astype(int)))
print("groups", len(groups), "| n", sum(g["mode"] == "n" for g in groups), "d", sum(g["mode"] == "d" for g in groups), flush=True)
blk = np.array([f"{FARM[g['b']]}_{NUM[g['b']] // 20}" for g in groups]); ub = sorted(set(blk)); fold = dict(zip(ub, np.random.default_rng(0).permutation(len(ub)) % 5))
res = []
for k in range(5):
    tr = [g for g, bb in zip(groups, blk) if fold[bb] != k]
    wN = fit_clogit([(g["C"], g["y"]) for g in tr], l2=0.5, iters=300)
    wF = fit_clogit([(g["C"][:-1, :len(FL)], g["y"]) for g in tr if g["mode"] == "n"], l2=0.5, iters=300)
    for g, bb in zip(groups, blk):
        if fold[bb] != k: continue
        s = g["C"] @ wN; p = np.exp(s - s.max()); p /= p.sum(); j = int(np.argmax(p[:-1]))
        sf = g["C"][:-1, :len(FL)] @ wF; pf = np.exp(sf - sf.max()); pf /= pf.sum(); jf = int(np.argmax(pf))
        res.append(dict(mode=g["mode"], fln_p=float(p[j]), fln_ok=(g["mode"] == "n" and j == g["y"]), fln_none=float(p[-1]), fl_p=float(pf[jf]), fl_ok=(g["mode"] == "n" and jf == g["y"])))
R = pd.DataFrame(res); n, d = R[R["mode"] == "n"], R[R["mode"] == "d"]
def st(df, pc, okc, t): m = df[pc] >= t; return float(m.mean()), float(df[okc][m].mean()) if m.any() else np.nan
out = {}
for nm, pc, okc in (("FL", "fl_p", "fl_ok"), ("FLN", "fln_p", "fln_ok")):
    for t in (.8, .5):
        cov, prec = st(n, pc, okc, t); fls = float((d[pc] >= t).mean()); out[(nm, t)] = (cov, prec, fls)
        print(f"{nm:3s} p>={t}: (n) coverage {cov:.3f} precision {prec:.3f} | (d) false-link rate {fls:.3f}", flush=True)
c1 = out[("FLN", .5)][1] >= out[("FL", .8)][1] - .02; c2 = out[("FLN", .5)][2] <= out[("FL", .8)][2]; c3 = out[("FLN", .5)][0] > out[("FL", .8)][0]
print(f"FLN1 gate: precision {c1} | false-link {c2} | coverage {c3} -> {'PASS -> stage 2' if c1 and c2 and c3 else 'FAIL (stop)'}", flush=True)
if not (c1 and c2 and c3): sys.exit(0)
# ---------------- stage 2: training table + EC judgement ----------------
w = fit_clogit([(g["C"], g["y"]) for g in groups], l2=0.5, iters=300); print("FLN weights", dict(zip(COLS, np.round(w, 3))), flush=True)
tab = []
for b in range(N):
    if not TRAIN[b] or FARM[b] not in ("F13", "F47"): continue
    lab = TRAIN.copy(); lab[(FARM == FARM[b]) & (np.abs(NUM - NUM[b]) <= 1)] = False
    df, _ = features7(b, 0, lab, removed=set(TEST_IDX))
    if len(df) == 0: continue
    s = mat(df) @ w; p = np.exp(s - s.max()); p /= p.sum(); j = int(np.argmax(p[:-1])); tab.append(dict(rec=rid(b), prev=rid(int(df.a.values[j])), p=round(float(p[j]), 4), p_none=round(float(p[-1]), 4)))
TB = pd.DataFrame(tab); TB.to_csv(os.path.join(env.LOCAL, "fln1_train_links.csv"), index=False); print("FLN train links", len(TB), "| p>=.5", int((TB.p >= .5).sum()), "| p>=.8", int((TB.p >= .8).sum()), flush=True)
src = open(os.path.join(HERE, "ev3l_ec_linked_yesterday_gru_v1.py"), encoding="utf-8").read(); __file__ = os.path.join(HERE, "ev3l_ec_linked_yesterday_gru_v1.py")
G = {"__file__": __file__, "__name__": "ev3lfln"}; exec(src[:src.index("\nO = pd.read_csv")], G); __file__ = os.path.join(HERE, "fln1_field_linker_none_option_v1.py")
EC_, idx_, TR_, rmm = G["EC"], G["idx"], G["TRAIN"], G["rm"]
FLT = pd.read_csv(os.path.join(env.LOCAL, "fl3_train_links.csv"))
_X = pd.concat([pd.read_csv(os.path.join(env.DATA, f)) for f in ("train_X.csv", "test_X.csv")]); _X = _X[_X.row_id.str[:3].isin(["F13", "F47"])]
_X["rec"] = _X.row_id.str[:7]; _X["h"] = _X.row_id.str[8:].astype(int); TIN = _X.pivot(index="rec", columns="h", values="in_temp")
def anchors(T, t):
    return {r: (p, q) for r, p, q in zip(T.rec, T.prev, T.p) if q >= t and p in idx_ and TR_[idx_[p]] and np.isfinite(EC_[idx_[p], 23]) and abs(TIN.at[r, 0] - TIN.at[p, 23]) <= 1.5}
ARMS = {"K8": (anchors(FLT, .8), False), "N5": (anchors(TB, .5), True)}
R2 = pd.read_csv(os.path.join(HERE, "nh1_rows_v1.csv.gz")); SEEDS = (1357, 2468, 3579)
MT = pickle.load(open(os.path.join(env.LOCAL, "nh1_mt1_v1.pkl"), "rb")); mt = np.array([MT[(v, k)][r][h] for v, k, r, h in zip(R2.validator, R2.fold, R2.rec, R2.hour)])
FOLDREC = {vk: set(Gd.rec) for vk, Gd in R2.groupby(["validator", "fold"])}; GRP = list(R2.groupby(["validator", "fold", "rec"]))
for s in SEEDS:
    R2[f"NH_{s}"] = (R2[f"A1_{s}"] - .1 * mt) / .9
    for name, (AN, soft) in ARMS.items():
        o = R2[f"NH_{s}"].values.copy()
        for (v, k, r), g in GRP:
            if r not in AN or AN[r][0] in FOLDREC[(v, k)]: continue
            p, q = AN[r]; ii = g.sort_values("hour").index.values; x = R2.loc[ii, f"NH_{s}"].values; pm = np.cumsum(x) / np.arange(1, len(x) + 1)
            o[ii] = x + (q if soft else 1.0) * (EC_[idx_[p], 23] - pm)
        R2[f"{name}_{s}"] = o
rows, blocks, recd = [], [], []
for v in ("DIAG10", "EL1", "P2LOO"):
    g = R2[R2.validator == v]; y = g.y.values; p2 = g.p2.values; key = (g.farm + "_" + (g.day // 5).astype(str)).values
    for s in SEEDS:
        dd = (g[f"N5_{s}"].values - y) ** 2 - (g[f"K8_{s}"].values - y) ** 2
        rows.append(dict(val=v, seed=s, dSSE_p2=float(dd[p2].sum()), dSSE_all=float(dd.sum()), N5_vs_K8_p2=100 * (rmm((g[f"N5_{s}"].values - y)[p2]) / rmm((g[f"K8_{s}"].values - y)[p2]) - 1),
                         N5_vs_K8_all=100 * (rmm(g[f"N5_{s}"].values - y) / rmm(g[f"K8_{s}"].values - y) - 1)))
        if v == "P2LOO": blocks.append(pd.Series(dd[p2]).groupby(key[p2]).mean())
        for r_, x in pd.Series(dd[p2]).groupby(g.rec.values[p2]).sum().items(): recd.append(dict(val=v, seed=s, rec=r_, d=x))
S = pd.DataFrame(rows); print(S.round(3).to_string(index=False)); print("anchored train records:", {k: len(a) for k, (a, _) in ARMS.items()})
main = S[S.val.isin(["P2LOO", "EL1"])]; c1 = bool((main.dSSE_p2 < 0).all()); x = pd.concat(blocks, axis=1).mean(axis=1).values
pP = float((x[np.random.default_rng(20270117).integers(0, len(x), (20000, len(x)))].mean(1) >= 0).mean()); RD_ = pd.DataFrame(recd)
nh_ = int((RD_[RD_.val.isin(["P2LOO", "EL1"])].groupby(["val", "seed", "rec"]).d.sum() > 1e-6).sum()); c3 = not (S[S.val == "DIAG10"].dSSE_all.mean() > 0)
print(f"FLN1 stage 2 EC N5 vs K8: cells {int((main.dSSE_p2 < 0).sum())}/6 | P2LOO P {pP:.4f} | harmed {nh_} | DIAG10 {S[S.val == 'DIAG10'].dSSE_all.mean():+.3f} -> {'PASS' if c1 and (pP < .025 or nh_ == 0) and c3 else 'FAIL'}")
