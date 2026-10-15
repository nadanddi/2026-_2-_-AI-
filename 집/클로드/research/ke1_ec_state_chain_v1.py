# -*- coding: utf-8 -*-
"""KE1 (2026-10-15 집 클로드; 5점모델_설계_v1.md 2층 — 동 상태 추적, EC).  사전 고정(커밋 후 실행).  학습 없음.
가설 H2·H3: EC 하루 수준은 동 상태로 이어짐 → 1단계 정답 전날이 없는 기록도 2~3단계 FLN 사슬(중간 = 그 폴드 검증 기록)로 정답 조상의 23시 EC 를 전파.
기준 K8 = NH + EA 앵커(FL 학습표 fl3 p ≥ .8·정답 전날·|dT| ≤ 1.5; 27번 EC 와 같음).
사슬(K8 앵커 없는 기록만): r → p1 → p2 → p3 (FLN 학습표 local/fln1_train_links.csv, 최대 3단계, 누적 P = Π p ≥ .5), 끝 = 그 폴드 밖 정답 기록, 중간 = 그 폴드 기록(NH 예측 있음).
  전파(EA 식 그대로, 새 상수 없음): 중간 m 마다 est23 ← NH_m23 + (est23 − mean(NH_m 0..23));  대상: pred_h = NH_h + P · (est23 − cm_h(NH)).
  1단계 정답 전날이지만 p < .8 인 기록은 사슬 길이 1 로 같은 식(AC1 의 S5 와 같은 꼴 — 이미 기각; 여기선 FLN 확률·누적 P 기준).
안(본페로니 k 2 → α .0125):  KEA = 위 사슬 전부,  KEH = 끝 정답 조상 A23 ≥ 1.0(고EC 상태)일 때만 (FC1B 6.823 진단: 조상 고EC 진짜 14 / 가짜 1).
판정(EC 분야 규칙): P2LOO·EL1 3시드 2차 dSSE < 0 & (P2LOO 블록 P < .0125 OR 손해 기록-칸 0) & DIAG10 전체 dSSE 평균 손해 아님.
Run:  cd 집/클로드/research && PYTHONPATH="" python -u ke1_ec_state_chain_v1.py
"""
import os, sys, pickle
HERE = os.path.dirname(os.path.abspath(__file__)); sys.argv = ["x"]; sys.path.insert(0, HERE)
import env  # noqa: F401
import numpy as np, pandas as pd
src = open(os.path.join(HERE, "ev3l_ec_linked_yesterday_gru_v1.py"), encoding="utf-8").read(); __file__ = os.path.join(HERE, "ev3l_ec_linked_yesterday_gru_v1.py")
G = {"__file__": __file__, "__name__": "ev3lfln"}; exec(src[:src.index("\nO = pd.read_csv")], G); __file__ = os.path.join(HERE, "ke1_ec_state_chain_v1.py")
EC_, idx_, TR_, rmm = G["EC"], G["idx"], G["TRAIN"], G["rm"]
FLT = pd.read_csv(os.path.join(env.LOCAL, "fl3_train_links.csv"))
_X = pd.concat([pd.read_csv(os.path.join(env.DATA, f)) for f in ("train_X.csv", "test_X.csv")]); _X = _X[_X.row_id.str[:3].isin(["F13", "F47"])]
_X["rec"] = _X.row_id.str[:7]; _X["h"] = _X.row_id.str[8:].astype(int); TIN = _X.pivot(index="rec", columns="h", values="in_temp")
def anchors(T, t):
    return {r: (p, q) for r, p, q in zip(T.rec, T.prev, T.p) if q >= t and p in idx_ and TR_[idx_[p]] and np.isfinite(EC_[idx_[p], 23]) and abs(TIN.at[r, 0] - TIN.at[p, 23]) <= 1.5}
TB = pd.read_csv(os.path.join(env.LOCAL, "fln1_train_links.csv")); LINK = {r: (p, q) for r, p, q in zip(TB.rec, TB.prev, TB.p)}
K8A = anchors(FLT, .8)
R2 = pd.read_csv(os.path.join(HERE, "nh1_rows_v1.csv.gz")); SEEDS = (1357, 2468, 3579)
MT = pickle.load(open(os.path.join(env.LOCAL, "nh1_mt1_v1.pkl"), "rb")); mt = np.array([MT[(v, k)][r][h] for v, k, r, h in zip(R2.validator, R2.fold, R2.rec, R2.hour)])
FOLDREC = {vk: set(Gd.rec) for vk, Gd in R2.groupby(["validator", "fold"])}; GRP = list(R2.groupby(["validator", "fold", "rec"]))
lab_ok = lambda p, vk: p in idx_ and bool(TR_[idx_[p]]) and np.isfinite(EC_[idx_[p], 23]) and p not in FOLDREC[vk]
fired = {"KEA": [], "KEH": []}
for s in SEEDS:
    R2[f"NH_{s}"] = (R2[f"A1_{s}"] - .1 * mt) / .9
    NHR = {(v, k, r): R2.loc[g.sort_values("hour").index, f"NH_{s}"].values for (v, k, r), g in GRP}
    o8 = R2[f"NH_{s}"].values.copy(); oA = o8.copy(); oH = o8.copy()
    for (v, k, r), g in GRP:
        ii = g.sort_values("hour").index.values; x = R2.loc[ii, f"NH_{s}"].values; pm = np.cumsum(x) / np.arange(1, len(x) + 1)
        if r in K8A and K8A[r][0] not in FOLDREC[(v, k)]:
            o8[ii] = x + (EC_[idx_[K8A[r][0]], 23] - pm); oA[ii] = o8[ii]; oH[ii] = o8[ii]; continue
        chain, P, cur = [], 1.0, r
        for step in range(3):
            if cur not in LINK: chain = None; break
            pv, q = LINK[cur]; P *= q
            if P < .5: chain = None; break
            chain.append(pv)
            if lab_ok(pv, (v, k)): break
            if (v, k, pv) not in NHR: chain = None; break
            cur = pv
        if not chain or not lab_ok(chain[-1], (v, k)): continue
        A = float(EC_[idx_[chain[-1]], 23]); est = A
        for m in reversed(chain[:-1]): nm = NHR[(v, k, m)]; est = nm[-1] + (est - nm.mean())
        new = x + P * (est - pm); oA[ii] = new
        if A >= 1.0: oH[ii] = new
        if s == SEEDS[0]: fired["KEA"].append((v, r, len(chain))); (fired["KEH"].append((v, r, len(chain))) if A >= 1.0 else None)
    R2[f"K8_{s}"], R2[f"KEA_{s}"], R2[f"KEH_{s}"] = o8, oA, oH
for a in ("KEA", "KEH"):
    F = pd.DataFrame(fired[a], columns=["val", "rec", "steps"]); print(a, "fired by validator/steps:", F.groupby(["val", "steps"]).size().to_dict(), flush=True)
    rows, blocks, recd = [], [], []
    for v in ("DIAG10", "EL1", "P2LOO"):
        g = R2[R2.validator == v]; y = g.y.values; p2 = g.p2.values; key = (g.farm + "_" + (g.day // 5).astype(str)).values
        for s in SEEDS:
            dd = (g[f"{a}_{s}"].values - y) ** 2 - (g[f"K8_{s}"].values - y) ** 2
            rows.append(dict(val=v, seed=s, dSSE_p2=float(dd[p2].sum()), dSSE_all=float(dd.sum()), vsK8_p2=100 * (rmm((g[f"{a}_{s}"].values - y)[p2]) / rmm((g[f"K8_{s}"].values - y)[p2]) - 1),
                             vsK8_all=100 * (rmm(g[f"{a}_{s}"].values - y) / rmm(g[f"K8_{s}"].values - y) - 1)))
            if v == "P2LOO": blocks.append(pd.Series(dd[p2]).groupby(key[p2]).mean())
            for r_, xx in pd.Series(dd[p2]).groupby(g.rec.values[p2]).sum().items(): recd.append(dict(val=v, seed=s, rec=r_, d=xx))
    S = pd.DataFrame(rows); print(S.round(3).to_string(index=False))
    main = S[S.val.isin(["P2LOO", "EL1"])]; c1 = bool((main.dSSE_p2 < 0).all()); xb = pd.concat(blocks, axis=1).mean(axis=1).values
    pP = float((xb[np.random.default_rng(20270119).integers(0, len(xb), (20000, len(xb)))].mean(1) >= 0).mean()); RD_ = pd.DataFrame(recd)
    nh_ = int((RD_[RD_.val.isin(["P2LOO", "EL1"])].groupby(["val", "seed", "rec"]).d.sum() > 1e-6).sum()); c3 = not (S[S.val == "DIAG10"].dSSE_all.mean() > 0)
    print(f"KE1 {a} vs K8: cells {int((main.dSSE_p2 < 0).sum())}/6 | P2LOO P {pP:.4f} | harmed {nh_} | DIAG10 {S[S.val == 'DIAG10'].dSSE_all.mean():+.3f} -> {'PASS' if c1 and (pP < .0125 or nh_ == 0) and c3 else 'FAIL'}", flush=True)
