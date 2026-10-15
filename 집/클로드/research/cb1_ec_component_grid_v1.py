# -*- coding: utf-8 -*-
"""CB1-EC (2026-10-15 집 클로드; 사용자 '부품들끼리 결합하거나 나누거나 다른 부품도 붙여보기').  사전 고정(커밋 후 실행).  EC.  학습 없음.
부품 격자 32조합 = 기본 섞기 4 x FL 확신 앵커(K8) 반영 강도 4 x FLN 약한 사슬(KEA) 2:
  기본: B0 NH(27번) | B1 .9·NH + .1·MT1(MTL 구성원) | B2 .95·NH + .05·EV3L-FL | B3 .9·NH + .1·EV3L-FL   (EV3L-FL = fl4_ev3lf, FL 링크 전날 창 GRU)
  K8: FL p ≥ .8·정답 전날(폴드 밖)·|dT| ≤ 1.5 기록에 pred = x + w·(A23 − cm_h(x)),  w ∈ {0, .5, .75, 1}
  KEA: K8 없는 기록에 FLN 사슬(KE1 과 같음: 최대 3단계, 누적 P ≥ .5, 중간은 같은 폴드 기록) on/off
고르기: DIAG10 전체 행 RMSE(3시드 평균) 최소 조합(같으면 격자 순서상 앞).   기준 REF27 = B0·w 1·KEA off.
판정(EC 분야 규칙, 선택 조합 vs REF27, α .025): P2LOO·EL1 3시드 2차 dSSE < 0 & (P2LOO 블록 P < .025 OR 손해 기록-칸 0). 고르기에 쓴 DIAG10 은 서술만.
서술: 32조합의 P2LOO·EL1·DIAG10 2차 상대 RMSE 표(판정 아님).
Run:  cd 집/클로드/research && PYTHONPATH="" python -u cb1_ec_component_grid_v1.py
"""
import os, sys, pickle, itertools
HERE = os.path.dirname(os.path.abspath(__file__)); sys.argv = ["x"]; sys.path.insert(0, HERE)
import env  # noqa: F401
import numpy as np, pandas as pd
src = open(os.path.join(HERE, "ev3l_ec_linked_yesterday_gru_v1.py"), encoding="utf-8").read()
G = {"__file__": os.path.join(HERE, "ev3l_ec_linked_yesterday_gru_v1.py"), "__name__": "ev3lcb"}; exec(src[:src.index("\nO = pd.read_csv")], G)
EC_, idx_, TR_ = G["EC"], G["idx"], G["TRAIN"]; rm = lambda e: float(np.sqrt(np.mean(np.square(e))))
_X = pd.concat([pd.read_csv(os.path.join(env.DATA, f)) for f in ("train_X.csv", "test_X.csv")]); _X = _X[_X.row_id.str[:3].isin(["F13", "F47"])]
_X["rec"] = _X.row_id.str[:7]; _X["h"] = _X.row_id.str[8:].astype(int); TIN = _X.pivot(index="rec", columns="h", values="in_temp")
FLT = pd.read_csv(os.path.join(env.LOCAL, "fl3_train_links.csv"))
K8L = {r: p for r, p, q in zip(FLT.rec, FLT.prev, FLT.p) if q >= .8 and p in idx_ and TR_[idx_[p]] and np.isfinite(EC_[idx_[p], 23]) and abs(TIN.at[r, 0] - TIN.at[p, 23]) <= 1.5}
FN = pd.read_csv(os.path.join(env.LOCAL, "fln1_train_links.csv")); LINK = {r: (p, q) for r, p, q in zip(FN.rec, FN.prev, FN.p)}
R = pd.read_csv(os.path.join(HERE, "nh1_rows_v1.csv.gz")).sort_values(["validator", "fold", "rec", "hour"]).reset_index(drop=True); SEEDS = (1357, 2468, 3579)
MT = pickle.load(open(os.path.join(env.LOCAL, "nh1_mt1_v1.pkl"), "rb")); EVF = pickle.load(open(os.path.join(env.LOCAL, "fl4_ev3lf_v1.pkl"), "rb"))
mt = np.array([MT[(v, k)][r][h] for v, k, r, h in zip(R.validator, R.fold, R.rec, R.hour)]); evf = np.array([EVF[(v, k)][r][h] for v, k, r, h in zip(R.validator, R.fold, R.rec, R.hour)])
FOLDREC = {vk: set(g.rec) for vk, g in R.groupby(["validator", "fold"])}; GRP = [(key, g.index.values) for key, g in R.groupby(["validator", "fold", "rec"])]
lab_ok = lambda p, vk: p in idx_ and bool(TR_[idx_[p]]) and np.isfinite(EC_[idx_[p], 23]) and p not in FOLDREC[vk]
BASES = ("B0", "B1", "B2", "B3"); WS = (0.0, .5, .75, 1.0); KS = (False, True); COMBOS = list(itertools.product(BASES, WS, KS))
P = {}
for s in SEEDS:
    nh = (R[f"A1_{s}"].values - .1 * mt) / .9
    base = {"B0": nh, "B1": .9 * nh + .1 * mt, "B2": .95 * nh + .05 * evf, "B3": .9 * nh + .1 * evf}
    for bn, x0 in base.items():
        XR = {key: x0[ii] for key, ii in GRP}
        addK8 = np.zeros(len(R)); addKEA = np.zeros(len(R)); hasK8 = np.zeros(len(R), bool)
        for (v, k, r), ii in GRP:
            x = XR[(v, k, r)]; pm = np.cumsum(x) / np.arange(1, len(x) + 1)
            if r in K8L and K8L[r] not in FOLDREC[(v, k)]:
                addK8[ii] = EC_[idx_[K8L[r]], 23] - pm; hasK8[ii] = True; continue
            chain, Pc, cur = [], 1.0, r
            for step in range(3):
                if cur not in LINK: chain = None; break
                pv, q = LINK[cur]; Pc *= q
                if Pc < .5: chain = None; break
                chain.append(pv)
                if lab_ok(pv, (v, k)): break
                if (v, k, pv) not in XR: chain = None; break
                cur = pv
            if not chain or not lab_ok(chain[-1], (v, k)): continue
            est = float(EC_[idx_[chain[-1]], 23])
            for m in reversed(chain[:-1]): nm = XR[(v, k, m)]; est = nm[-1] + (est - nm.mean())
            addKEA[ii] = Pc * (est - pm)
        for w in WS:
            for kea in KS: P[(bn, w, kea, s)] = x0 + w * addK8 + (addKEA if kea else 0.0)
def rel(v, c, m="p2"):
    g = R.validator == v; y = R.y.values[g]; sel = R.p2.values[g] if m == "p2" else np.ones(int(g.sum()), bool)
    return 100 * np.mean([rm((P[c + (s,)][g] - y)[sel]) / rm((P[("B0", 1.0, False, s)][g] - y)[sel]) - 1 for s in SEEDS])
gD = (R.validator == "DIAG10").values; yD = R.y.values[gD]
score = {c: np.mean([rm(P[c + (s,)][gD] - yD) for s in SEEDS]) for c in COMBOS}; best = min(COMBOS, key=lambda c: (round(score[c], 6), COMBOS.index(c)))
T = pd.DataFrame([dict(base=c[0], w=c[1], KEA=c[2], DIAG10_all=round(score[c], 5), DIAG10_p2=rel("DIAG10", c), P2LOO_p2=rel("P2LOO", c), EL1_p2=rel("EL1", c)) for c in COMBOS])
pd.set_option("display.width", 200); print(T.round(3).to_string(index=False)); print("\nselected on DIAG10:", best)
rows, blocks, recd = [], [], []
for v in ("P2LOO", "EL1"):
    g = (R.validator == v).values; y = R.y.values[g]; p2 = R.p2.values[g]; key = (R.farm[g] + "_" + (R.day[g] // 5).astype(str)).values
    for s in SEEDS:
        dd = (P[best + (s,)][g] - y) ** 2 - (P[("B0", 1.0, False, s)][g] - y) ** 2; rows.append(dict(val=v, seed=s, dSSE_p2=float(dd[p2].sum())))
        if v == "P2LOO": blocks.append(pd.Series(dd[p2]).groupby(key[p2]).mean())
        for r_, xx in pd.Series(dd[p2]).groupby(R.rec.values[g][p2]).sum().items(): recd.append(dict(val=v, seed=s, rec=r_, d=xx))
S = pd.DataFrame(rows); xb = pd.concat(blocks, axis=1).mean(axis=1).values
pP = float((xb[np.random.default_rng(20270128).integers(0, len(xb), (20000, len(xb)))].mean(1) >= 0).mean()); RD_ = pd.DataFrame(recd)
nh_ = int((RD_.groupby(["val", "seed", "rec"]).d.sum() > 1e-6).sum()); c1 = bool((S.dSSE_p2 < 0).all())
print(S.round(3).to_string(index=False))
print(f"CB1-EC {best} vs REF27: cells {int((S.dSSE_p2 < 0).sum())}/6 | P2LOO P {pP:.4f} | harmed {nh_} -> {'PASS' if (best != ('B0', 1.0, False)) and c1 and (pP < .025 or nh_ == 0) else 'FAIL'}")
