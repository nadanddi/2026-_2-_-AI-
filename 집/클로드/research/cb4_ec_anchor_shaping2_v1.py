# -*- coding: utf-8 -*-
"""CB4-EC (2026-10-15 집 클로드; 사용자 '둘 다 진행' — EC 앵커 손잡이 더 다듬기).  사전 고정(커밋 후 실행).  EC.  학습 없음.
CB2-EC·AW1: 강도 .65~.80 CONST 가 세 검증기 일관(판정은 P 미달).  이번 손잡이(격자 5 x 3 x 3 x 2 = 90):
  강도 w0 {.6, .7, .75, .8, .9};  하루 모양 CONST | DECAY(0시 1 → 23시 .5) | RISE(0시 .5 → 23시 1);
  고EC 어제(A23 ≥ 1.0) 배율 {.8, 1, 1.2};  링크 확률 비례 OFF | ON(x p).   고정: FL p ≥ .8, |dT| ≤ 1.5, MT 0.
  pred_h = x_h + w0·shape_h·hi·pfac·(A23 − cm_h(x)).
고르기 DIAG10 전체 행 RMSE(3시드 평균).  기준 REF27 = (1, CONST, 1, OFF).
판정(EC 분야 규칙, α .025): P2LOO·EL1 3시드 2차 dSSE < 0 & (P2LOO 블록 P < .025 OR 손해 기록-칸 0).  서술: 손잡이 주효과, AW(.75 CONST) 대비.
Run:  cd 집/클로드/research && PYTHONPATH="" python -u cb4_ec_anchor_shaping2_v1.py
"""
import os, sys, itertools
HERE = os.path.dirname(os.path.abspath(__file__)); sys.argv = ["x"]; sys.path.insert(0, HERE)
src = open(os.path.join(HERE, "cb2_ec_anchor_shaping_v1.py"), encoding="utf-8").read()
exec(src[:src.index("WS = tuple(")])
__file__ = os.path.join(HERE, "cb4_ec_anchor_shaping2_v1.py")
W0 = (.6, .7, .75, .8, .9); SHP = ("CONST", "DECAY", "RISE"); HI = (.8, 1.0, 1.2); PF = (False, True)
COMBOS = list(itertools.product(W0, SHP, HI, PF)); REF = (1.0, "CONST", 1.0, False); EXTRA = [REF, (.75, "CONST", 1.0, False)]
hr = R.hour.values; SHV = {"CONST": np.ones(len(R)), "DECAY": 1 - .5 * hr / 23, "RISE": .5 + .5 * hr / 23}
P = {}
for s in SEEDS:
    x0 = (R[f"A1_{s}"].values - .1 * mt) / .9; gap = np.zeros(len(R)); A = np.zeros(len(R)); pr = np.zeros(len(R))
    for (v, k, r), ii in GRP:
        if r not in FL3 or FL3[r][0] in FOLDREC[(v, k)] or FL3[r][1] < .8 or DT[r] > 1.5: continue
        x = x0[ii]; pm = np.cumsum(x) / np.arange(1, len(x) + 1); a = EC_[idx_[FL3[r][0]], 23]; gap[ii] = a - pm; A[ii] = a; pr[ii] = FL3[r][1]
    for c in COMBOS + EXTRA:
        w0, sh, hi, pf = c; P[c + (s,)] = x0 + w0 * SHV[sh] * np.where(A >= 1.0, hi, 1.0) * (pr if pf else 1.0) * gap
gD = (R.validator == "DIAG10").values; yD = R.y.values[gD]
score = {c: np.mean([rm(P[c + (s,)][gD] - yD) for s in SEEDS]) for c in COMBOS}; best = min(COMBOS, key=lambda c: (round(score[c], 6), COMBOS.index(c)))
def rel(v, c, ref=REF):
    g = (R.validator == v).values; y = R.y.values[g]; p2 = R.p2.values[g]
    return 100 * np.mean([rm((P[c + (s,)][g] - y)[p2]) / rm((P[ref + (s,)][g] - y)[p2]) - 1 for s in SEEDS])
T_ = pd.DataFrame([dict(w0=c[0], shape=c[1], hi=c[2], pfac=c[3], DIAG10=rel("DIAG10", c), P2LOO=rel("P2LOO", c), EL1=rel("EL1", c)) for c in COMBOS])
pd.set_option("display.width", 200)
for col in ("w0", "shape", "hi", "pfac"): print(col, T_.groupby(col)[["DIAG10", "P2LOO", "EL1"]].mean().round(3).to_dict("index"))
print(T_.sort_values("DIAG10").head(8).round(3).to_string(index=False)); print("selected on DIAG10:", best, "| vs AW(.75 CONST): P2LOO %+.2f EL1 %+.2f" % (rel("P2LOO", best, EXTRA[1]), rel("EL1", best, EXTRA[1])))
rows, blocks, recd = [], [], []
for v in ("P2LOO", "EL1"):
    g = (R.validator == v).values; y = R.y.values[g]; p2 = R.p2.values[g]; key = (R.farm[g] + "_" + (R.day[g] // 5).astype(str)).values
    for s in SEEDS:
        dd = (P[best + (s,)][g] - y) ** 2 - (P[REF + (s,)][g] - y) ** 2; rows.append(dict(val=v, seed=s, dSSE_p2=float(dd[p2].sum()), rel=round(rel(v, best), 2)))
        if v == "P2LOO": blocks.append(pd.Series(dd[p2]).groupby(key[p2]).mean())
        for r_, xx in pd.Series(dd[p2]).groupby(R.rec.values[g][p2]).sum().items(): recd.append(dict(val=v, seed=s, rec=r_, d=xx))
S = pd.DataFrame(rows); xb = pd.concat(blocks, axis=1).mean(axis=1).values
pP = float((xb[np.random.default_rng(20270136).integers(0, len(xb), (20000, len(xb)))].mean(1) >= 0).mean()); RD_ = pd.DataFrame(recd)
nh_ = int((RD_.groupby(["val", "seed", "rec"]).d.sum() > 1e-6).sum()); c1 = bool((S.dSSE_p2 < 0).all())
print(S.round(3).to_string(index=False))
print(f"CB4-EC {best} vs REF27: cells {int((S.dSSE_p2 < 0).sum())}/6 | P2LOO P {pP:.4f} | harmed {nh_} -> {'PASS' if best != REF and c1 and (pP < .025 or nh_ == 0) else 'FAIL'}")
