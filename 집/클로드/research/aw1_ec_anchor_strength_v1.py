# -*- coding: utf-8 -*-
"""AW1 (2026-10-15 집 클로드; CB1-EC·CB2-EC 서술 후속).  사전 고정(커밋 후 실행).  EC.  학습 없음.
CB1-EC(DIAG10 로 고른 격자)에서 앵커 강도 .75 가 1.0 보다 기본 4 x 검증기 3 모두에서 나았고, CB2-EC 강도 곡선(p .8·dT 1.5·MT 0)도
  .65~.80 구간이 세 검증기 모두 음(P2LOO −1.9~−2.3, EL1 −1.9~−2.6, DIAG10 −3.7~−5.1).  → 손잡이 하나(강도)만 바꾼 단일 안: w = .75 (CB1 에서 고른 값, 새로 맞추지 않음).
안: 27번 EC(NH + FL K8 앵커) 에서 앵커 반영만 pred = x + .75·(A23 − cm_h(x)).  기준 REF27 = 1.0.
판정(EC 분야 규칙, α .025): P2LOO·EL1 3시드 2차 dSSE < 0 & (P2LOO 블록 P < .025 OR 손해 기록-칸 0) & DIAG10 전체 손해 아님(DIAG10 은 고르기에 씀 → 서술).
Run:  cd 집/클로드/research && PYTHONPATH="" python -u aw1_ec_anchor_strength_v1.py
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.argv = ["x"]; sys.path.insert(0, HERE)
src = open(os.path.join(HERE, "cb2_ec_anchor_shaping_v1.py"), encoding="utf-8").read()
exec(src[:src.index("WS = tuple(")])
__file__ = os.path.join(HERE, "aw1_ec_anchor_strength_v1.py")
P = {}
for s in SEEDS:
    x0 = (R[f"A1_{s}"].values - .1 * mt) / .9; gap = np.zeros(len(R))
    for (v, k, r), ii in GRP:
        if r not in FL3 or FL3[r][0] in FOLDREC[(v, k)] or FL3[r][1] < .8 or DT[r] > 1.5: continue
        x = x0[ii]; pm = np.cumsum(x) / np.arange(1, len(x) + 1); gap[ii] = EC_[idx_[FL3[r][0]], 23] - pm
    P[("REF", s)], P[("AW", s)] = x0 + gap, x0 + .75 * gap
rows, blocks, recd = [], [], []
for v in ("P2LOO", "EL1", "DIAG10"):
    g = (R.validator == v).values; y = R.y.values[g]; p2 = R.p2.values[g]; key = (R.farm[g] + "_" + (R.day[g] // 5).astype(str)).values
    for s in SEEDS:
        dd = (P[("AW", s)][g] - y) ** 2 - (P[("REF", s)][g] - y) ** 2
        rows.append(dict(val=v, seed=s, dSSE_p2=float(dd[p2].sum()), dSSE_all=float(dd.sum()), rel_p2=100 * (rm((P[("AW", s)][g] - y)[p2]) / rm((P[("REF", s)][g] - y)[p2]) - 1)))
        if v == "P2LOO": blocks.append(pd.Series(dd[p2]).groupby(key[p2]).mean())
        if v != "DIAG10":
            for r_, xx in pd.Series(dd[p2]).groupby(R.rec.values[g][p2]).sum().items(): recd.append(dict(val=v, seed=s, rec=r_, d=xx))
S = pd.DataFrame(rows); print(S.round(3).to_string(index=False))
main = S[S.val != "DIAG10"]; xb = pd.concat(blocks, axis=1).mean(axis=1).values
pP = float((xb[np.random.default_rng(20270131).integers(0, len(xb), (20000, len(xb)))].mean(1) >= 0).mean()); RD_ = pd.DataFrame(recd)
harm = RD_.groupby(["val", "seed", "rec"]).d.sum(); nh_ = int((harm > 1e-6).sum())
print("harmed record-cells:", sorted(set(i[2] for i in harm[harm > 1e-6].index)))
c1 = bool((main.dSSE_p2 < 0).all()); c3 = not (S[S.val == "DIAG10"].dSSE_all.mean() > 0)
print(f"AW1 w .75: cells {int((main.dSSE_p2 < 0).sum())}/6 | P2LOO P {pP:.4f} | harmed {nh_} | DIAG10 {S[S.val == 'DIAG10'].dSSE_all.mean():+.3f} -> {'PASS' if c1 and (pP < .025 or nh_ == 0) and c3 else 'FAIL'}")
