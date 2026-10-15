# -*- coding: utf-8 -*-
"""CB2-EC (2026-10-15 집 클로드; 사용자 '괜찮아 보이는 부품을 이리 깎고 저리 깎기').  사전 고정(커밋 후 실행).  EC.  학습 없음.
CB1-EC(서술 표): FL 확신 앵커 강도 .75 가 1.0 보다 기본 4 x 검증기 3 모두에서 나음 → 앵커 부품을 다듬음.
손잡이(격자 = 11 x 2 x 3 x 3 x 4 = 792):
  강도 w0 ∈ {.5, .55, …, 1.0};  모양: CONST(하루 내내 w0) | DECAY(0시 w0 → 23시 w0/2 선형);  링크 문턱 p ≥ {.7, .8, .9};  |dT_air| ≤ {1.0, 1.5, 2.0};
  기본 = .x·NH + m·MT1, m ∈ {0, .05, .10, .15} (나머지 NH).   KEA 없음(확신 없는 어제 배제 원칙).
  앵커 식: pred_h = x_h + w_h·(A23 − cm_h(x)), 전날 = FL(fl3) p ≥ 문턱·정답 학습 기록·폴드 밖·|dT| ≤ 문턱.
고르기: DIAG10 전체 행 RMSE(3시드 평균) 최소.   기준 REF27 = w 1·CONST·p .8·dT 1.5·m 0.
판정(EC 분야 규칙, 선택 vs REF27, α .025): P2LOO·EL1 3시드 2차 dSSE < 0 & (P2LOO 블록 P < .025 OR 손해 기록-칸 0).
서술: 손잡이별 주효과(P2LOO·EL1·DIAG10), 고르기 상위 10.
Run:  cd 집/클로드/research && PYTHONPATH="" python -u cb2_ec_anchor_shaping_v1.py
"""
import os, sys, pickle, itertools
HERE = os.path.dirname(os.path.abspath(__file__)); sys.argv = ["x"]; sys.path.insert(0, HERE)
import env  # noqa: F401
import numpy as np, pandas as pd
src = open(os.path.join(HERE, "ev3l_ec_linked_yesterday_gru_v1.py"), encoding="utf-8").read()
G = {"__file__": os.path.join(HERE, "ev3l_ec_linked_yesterday_gru_v1.py"), "__name__": "ev3lcb2"}; exec(src[:src.index("\nO = pd.read_csv")], G)
EC_, idx_, TR_ = G["EC"], G["idx"], G["TRAIN"]; rm = lambda e: float(np.sqrt(np.mean(np.square(e))))
_X = pd.concat([pd.read_csv(os.path.join(env.DATA, f)) for f in ("train_X.csv", "test_X.csv")]); _X = _X[_X.row_id.str[:3].isin(["F13", "F47"])]
_X["rec"] = _X.row_id.str[:7]; _X["h"] = _X.row_id.str[8:].astype(int); TIN = _X.pivot(index="rec", columns="h", values="in_temp")
FLT = pd.read_csv(os.path.join(env.LOCAL, "fl3_train_links.csv"))
R = pd.read_csv(os.path.join(HERE, "nh1_rows_v1.csv.gz")).sort_values(["validator", "fold", "rec", "hour"]).reset_index(drop=True); SEEDS = (1357, 2468, 3579)
MT = pickle.load(open(os.path.join(env.LOCAL, "nh1_mt1_v1.pkl"), "rb")); EVF = pickle.load(open(os.path.join(env.LOCAL, "fl4_ev3lf_v1.pkl"), "rb"))
mt = np.array([MT[(v, k)][r][h] for v, k, r, h in zip(R.validator, R.fold, R.rec, R.hour)]); evf = np.array([EVF[(v, k)][r][h] for v, k, r, h in zip(R.validator, R.fold, R.rec, R.hour)])
FOLDREC = {vk: set(g.rec) for vk, g in R.groupby(["validator", "fold"])}; GRP = [(key, g.index.values) for key, g in R.groupby(["validator", "fold", "rec"])]
import itertools
FL3 = {r: (p, q) for r, p, q in zip(FLT.rec, FLT.prev, FLT.p) if p in idx_ and TR_[idx_[p]] and np.isfinite(EC_[idx_[p], 23])}
DT = {r: abs(TIN.at[r, 0] - TIN.at[p, 23]) for r, (p, q) in FL3.items()}
WS = tuple(round(.5 + .05 * i, 2) for i in range(11)); SHAPES = ("CONST", "DECAY"); PTH = (.7, .8, .9); DTH = (1.0, 1.5, 2.0); MS = (0.0, .05, .10, .15)
COMBOS = list(itertools.product(WS, SHAPES, PTH, DTH, MS)); REF = (1.0, "CONST", .8, 1.5, 0.0)
hr = R.hour.values; SH = {"CONST": np.ones(len(R)), "DECAY": 1 - .5 * hr / 23}
P = {}
for s in SEEDS:
    nh = (R[f"A1_{s}"].values - .1 * mt) / .9
    for m in MS:
        x0 = (1 - m) * nh + m * mt; gap = np.zeros(len(R)); prob = np.zeros(len(R)); dtv = np.full(len(R), 9.0)
        for (v, k, r), ii in GRP:
            if r not in FL3 or FL3[r][0] in FOLDREC[(v, k)]: continue
            x = x0[ii]; pm = np.cumsum(x) / np.arange(1, len(x) + 1); gap[ii] = EC_[idx_[FL3[r][0]], 23] - pm; prob[ii] = FL3[r][1]; dtv[ii] = DT[r]
        for w, sh, pt, dt in itertools.product(WS, SHAPES, PTH, DTH):
            on = (prob >= pt) & (dtv <= dt); P[(w, sh, pt, dt, m, s)] = x0 + on * w * SH[sh] * gap
gD = (R.validator == "DIAG10").values; yD = R.y.values[gD]
score = {c: np.mean([rm(P[c + (s,)][gD] - yD) for s in SEEDS]) for c in COMBOS}; best = min(COMBOS, key=lambda c: (round(score[c], 6), COMBOS.index(c)))
def rel(v, c):
    g = (R.validator == v).values; y = R.y.values[g]; p2 = R.p2.values[g]
    return 100 * np.mean([rm((P[c + (s,)][g] - y)[p2]) / rm((P[REF + (s,)][g] - y)[p2]) - 1 for s in SEEDS])
T = pd.DataFrame([dict(w=c[0], shape=c[1], p=c[2], dT=c[3], mt=c[4], DIAG10=score[c]) for c in COMBOS])
pd.set_option("display.width", 200)
top = sorted(COMBOS, key=lambda c: score[c])[:10]
print(pd.DataFrame([dict(w=c[0], shape=c[1], p=c[2], dT=c[3], mt=c[4], DIAG10=round(score[c], 5), P2LOO_p2=round(rel("P2LOO", c), 2), EL1_p2=round(rel("EL1", c), 2)) for c in top]).to_string(index=False))
sub = [c for c in COMBOS if c[2] == .8 and c[3] == 1.5 and c[4] == 0.0]
print("
strength x shape at p .8, dT 1.5, mt 0 (P2LOO / EL1 / DIAG10 p2 %):")
print(pd.DataFrame([dict(w=c[0], shape=c[1], P2LOO=round(rel("P2LOO", c), 2), EL1=round(rel("EL1", c), 2), DIAG10=round(rel("DIAG10", c), 2)) for c in sub]).to_string(index=False))
print("
selected on DIAG10:", best)
rows, blocks, recd = [], [], []
for v in ("P2LOO", "EL1"):
    g = (R.validator == v).values; y = R.y.values[g]; p2 = R.p2.values[g]; key = (R.farm[g] + "_" + (R.day[g] // 5).astype(str)).values
    for s in SEEDS:
        dd = (P[best + (s,)][g] - y) ** 2 - (P[REF + (s,)][g] - y) ** 2; rows.append(dict(val=v, seed=s, dSSE_p2=float(dd[p2].sum()), rel=round(rel(v, best), 2)))
        if v == "P2LOO": blocks.append(pd.Series(dd[p2]).groupby(key[p2]).mean())
        for r_, xx in pd.Series(dd[p2]).groupby(R.rec.values[g][p2]).sum().items(): recd.append(dict(val=v, seed=s, rec=r_, d=xx))
S = pd.DataFrame(rows); xb = pd.concat(blocks, axis=1).mean(axis=1).values
pP = float((xb[np.random.default_rng(20270130).integers(0, len(xb), (20000, len(xb)))].mean(1) >= 0).mean()); RD_ = pd.DataFrame(recd)
nh_ = int((RD_.groupby(["val", "seed", "rec"]).d.sum() > 1e-6).sum()); c1 = bool((S.dSSE_p2 < 0).all())
print(S.round(3).to_string(index=False))
print(f"CB2-EC {best} vs REF27: cells {int((S.dSSE_p2 < 0).sum())}/6 | P2LOO P {pP:.4f} | harmed {nh_} -> {'PASS' if best != REF and c1 and (pP < .025 or nh_ == 0) else 'FAIL'}")
