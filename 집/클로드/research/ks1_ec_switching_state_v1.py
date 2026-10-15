# -*- coding: utf-8 -*-
"""KS1 (2026-10-15 집 클로드; 5점모델_설계_v1.md 2층 EC 전환 상태 — 사용자 '기다리는 동안 ec 전환 칼만도 진행해봐').  사전 고정(커밋 후 실행).
가설 H3: EC 하루 수준 = {보통, 고EC 에피소드} 상태 + 상태 안 연속 수준; 상태는 날을 넘어 이어짐(EH0 19/19, FC1B 조상 고EC 14/1).
기준 K8 = NH + EA 앵커(FL p ≥ .8·정답 전날·|dT| ≤ 1.5) — 27번 EC.   KS1 은 K8 앵커가 없는 기록에만 적용.
폴드 (v, k) 마다 '같은 검증기의 다른 폴드 기록'(NH 는 그들의 폴드 밖 예측 → 정당한 학습 자료)으로 적합:
  상태 라벨 H = 그 기록 하루 평균 EC ≥ 1.0.  π0 = 학습 기록 중 H 비율.
  관측 모형 L_h = P(H | c_h, h): 로지스틱(특징 c_h, c_h², h/23; C = 1), c_h = 그날 NH 0..h 누적 평균(인과).
  전이 T(H | 어제 상태): cert 정답 짝(feat7 TL7: 학습 기록끼리 진짜 전날) 중 둘 다 그 폴드 밖인 짝으로 셈(+1 평활).
  잔차 모형: 상태 s 별로 (y_h − NH_h) ~ a_s + b_s·c_h  (최소제곱, 그 폴드 밖 학습 기록 행).
대상 기록 r(K8 앵커 없음):  사전 π_r = P·T(H | s_prev) + (1 − P)·π0 — FLN 1단계 짝 prev 가 그 폴드 밖 정답 기록이고 p ≥ .5 일 때(P = p, s_prev = prev 하루 평균 ≥ 1.0), 아니면 π0.
  사후 logit q_h = logit(L_h) + logit(π_r) − logit(π0);  보정_h = q_h·(a_H + b_H c_h) + (1 − q_h)·(a_N + b_N c_h);  pred_h = NH_h + 보정_h.
판정(EC 분야 규칙, 단일 안 α .025): KS1 vs K8 — P2LOO·EL1 3시드 2차 dSSE < 0 & (P2LOO 블록 P < .025 OR 손해 기록-칸 0) & DIAG10 전체 dSSE 평균 손해 아님.
서술: 사전에 어제 정답 상태를 쓴 기록 수, 고EC/보통 분해, KEA(6.933) 위에 얹은 경우.
Run:  cd 집/클로드/research && PYTHONPATH="" python -u ks1_ec_switching_state_v1.py
"""
import os, sys, pickle
HERE = os.path.dirname(os.path.abspath(__file__)); sys.argv = ["x"]
sys.path.insert(0, os.path.join(HERE, "linker_v7")); sys.path.insert(0, HERE)
import env  # noqa: F401
from feat7 import TL7, rid  # noqa: E402
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
src = open(os.path.join(HERE, "ev3l_ec_linked_yesterday_gru_v1.py"), encoding="utf-8").read()
G = {"__file__": os.path.join(HERE, "ev3l_ec_linked_yesterday_gru_v1.py"), "__name__": "ev3lks"}; exec(src[:src.index("\nO = pd.read_csv")], G)
EC_, idx_, TR_, rmm = G["EC"], G["idx"], G["TRAIN"], G["rm"]
_X = pd.concat([pd.read_csv(os.path.join(env.DATA, f)) for f in ("train_X.csv", "test_X.csv")]); _X = _X[_X.row_id.str[:3].isin(["F13", "F47"])]
_X["rec"] = _X.row_id.str[:7]; _X["h"] = _X.row_id.str[8:].astype(int); TIN = _X.pivot(index="rec", columns="h", values="in_temp")
FLT = pd.read_csv(os.path.join(env.LOCAL, "fl3_train_links.csv"))
K8A = {r: p for r, p, q in zip(FLT.rec, FLT.prev, FLT.p) if q >= .8 and p in idx_ and TR_[idx_[p]] and np.isfinite(EC_[idx_[p], 23]) and abs(TIN.at[r, 0] - TIN.at[p, 23]) <= 1.5}
FN = pd.read_csv(os.path.join(env.LOCAL, "fln1_train_links.csv")); LINK = {r: (p, q) for r, p, q in zip(FN.rec, FN.prev, FN.p)}
DAYM = {rid(i): float(np.nanmean(EC_[i])) for i in range(len(EC_)) if TR_[i] and np.isfinite(EC_[i]).any()}
PAIRS = [(rid(int(a)), rid(int(b))) for a, b in zip(TL7.a, TL7.b)]
R = pd.read_csv(os.path.join(HERE, "nh1_rows_v1.csv.gz")); SEEDS = (1357, 2468, 3579)
MT = pickle.load(open(os.path.join(env.LOCAL, "nh1_mt1_v1.pkl"), "rb")); mt = np.array([MT[(v, k)][r][h] for v, k, r, h in zip(R.validator, R.fold, R.rec, R.hour)])
R = R.sort_values(["validator", "fold", "rec", "hour"]).reset_index(drop=True); mt = None
mt = np.array([MT[(v, k)][r][h] for v, k, r, h in zip(R.validator, R.fold, R.rec, R.hour)])
FOLDREC = {vk: set(g.rec) for vk, g in R.groupby(["validator", "fold"])}
logit = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
used_prior = {}
for s in SEEDS:
    R[f"NH_{s}"] = (R[f"A1_{s}"] - .1 * mt) / .9
    R[f"c_{s}"] = R.groupby(["validator", "fold", "rec"])[f"NH_{s}"].transform(lambda x: x.expanding().mean())
    o8 = R[f"NH_{s}"].values.copy(); oS = o8.copy()
    for (v, k), Gk in R.groupby(["validator", "fold"]):
        fr = FOLDREC[(v, k)]; Tr = R[(R.validator == v) & (R.fold != k) & R.rec.map(lambda r: r in DAYM)]
        Tr = Tr[~Tr.rec.isin(fr)]; yH = Tr.rec.map(lambda r: DAYM[r] >= 1.0).values.astype(int)
        if yH.sum() < 3 or (1 - yH).sum() < 3: continue
        c = Tr[f"c_{s}"].values; hh = Tr.hour.values / 23; Xl = np.c_[c, c ** 2, hh]
        lr = LogisticRegression(C=1.0, max_iter=1000).fit(Xl, yH); pi0 = float(yH[Tr.hour.values == 0].mean())
        res = Tr.y.values - Tr[f"NH_{s}"].values; coef = {}
        for st in (0, 1):
            m = yH == st; A = np.c_[np.ones(m.sum()), c[m]]; coef[st] = np.linalg.lstsq(A, res[m], rcond=None)[0]
        nHH = nHN = nNH = nNN = 1
        for a, b in PAIRS:
            if a in fr or b in fr or a not in DAYM or b not in DAYM: continue
            sa, sb = DAYM[a] >= 1.0, DAYM[b] >= 1.0
            if sa: nHH += sb; nHN += (not sb)
            else: nNH += sb; nNN += (not sb)
        TH = {True: nHH / (nHH + nHN), False: nNH / (nNH + nNN)}
        for r, g in Gk.groupby("rec"):
            ii = g.index.values; x = R.loc[ii, f"NH_{s}"].values; pm = np.cumsum(x) / np.arange(1, len(x) + 1)
            if r in K8A and K8A[r] not in fr:
                o8[ii] = x + (EC_[idx_[K8A[r]], 23] - pm); oS[ii] = o8[ii]; continue
            pr = pi0
            if r in LINK:
                pv, p = LINK[r]
                if p >= .5 and pv in DAYM and pv not in fr: pr = p * TH[DAYM[pv] >= 1.0] + (1 - p) * pi0; used_prior[(v, r)] = DAYM[pv] >= 1.0
            cc = R.loc[ii, f"c_{s}"].values; Lh = lr.predict_proba(np.c_[cc, cc ** 2, R.loc[ii, "hour"].values / 23])[:, 1]
            q = 1 / (1 + np.exp(-(logit(Lh) + logit(pr) - logit(pi0))))
            oS[ii] = x + q * (coef[1][0] + coef[1][1] * cc) + (1 - q) * (coef[0][0] + coef[0][1] * cc)
    R[f"K8_{s}"], R[f"KS_{s}"] = o8, oS
UP = pd.Series(used_prior); print("records with yesterday-state prior (seed 1):", UP.groupby(level=0).size().to_dict(), "| prev high:", int(UP.sum()), flush=True)
rows, blocks, recd = [], [], []
for v in ("DIAG10", "EL1", "P2LOO"):
    g = R[R.validator == v]; y = g.y.values; p2 = g.p2.values; key = (g.farm + "_" + (g.day // 5).astype(str)).values
    hi = g.rec.map(lambda r: DAYM.get(r, 0) >= 1.0).values
    for s in SEEDS:
        dd = (g[f"KS_{s}"].values - y) ** 2 - (g[f"K8_{s}"].values - y) ** 2
        f = lambda m: 100 * (rmm((g[f"KS_{s}"].values - y)[m]) / rmm((g[f"K8_{s}"].values - y)[m]) - 1) if m.any() else np.nan
        rows.append(dict(val=v, seed=s, dSSE_p2=float(dd[p2].sum()), dSSE_all=float(dd.sum()), vsK8_p2=f(p2), vsK8_all=f(np.ones(len(y), bool)), high_p2=f(p2 & hi), normal_p2=f(p2 & ~hi)))
        if v == "P2LOO": blocks.append(pd.Series(dd[p2]).groupby(key[p2]).mean())
        for r_, xx in pd.Series(dd[p2]).groupby(g.rec.values[p2]).sum().items(): recd.append(dict(val=v, seed=s, rec=r_, d=xx))
S = pd.DataFrame(rows); pd.set_option("display.width", 220); print(S.round(3).to_string(index=False))
main = S[S.val.isin(["P2LOO", "EL1"])]; c1 = bool((main.dSSE_p2 < 0).all()); xb = pd.concat(blocks, axis=1).mean(axis=1).values
pP = float((xb[np.random.default_rng(20270120).integers(0, len(xb), (20000, len(xb)))].mean(1) >= 0).mean()); RD_ = pd.DataFrame(recd)
nh_ = int((RD_[RD_.val.isin(["P2LOO", "EL1"])].groupby(["val", "seed", "rec"]).d.sum() > 1e-6).sum()); c3 = not (S[S.val == "DIAG10"].dSSE_all.mean() > 0)
print(f"KS1 vs K8: cells {int((main.dSSE_p2 < 0).sum())}/6 | P2LOO P {pP:.4f} | harmed {nh_} | DIAG10 {S[S.val == 'DIAG10'].dSSE_all.mean():+.3f} -> {'PASS' if c1 and (pP < .025 or nh_ == 0) and c3 else 'FAIL'}")
