# -*- coding: utf-8 -*-
"""ECG1 (2026-10-16 집 클로드; DGX4(6.988) 'K8 EC 앵커는 고EC 행에서 크게 이득, 2차 보통 행은 세 검증기 모두 손해' -> 사용자 'ec 도 이런 식으로 개선 부분').
사전 고정(커밋 후 실행).  EC.  학습 없음.  앵커를 '어제 EC 수준' 으로 거르기 (어제 = 정답 있는 학습 기록 23시 EC = 현장에선 어제 측정값):
  기준 REF27 = NH + K8 앵커(fl3, p >= .8, |dT| <= 1.5, 폴드 밖, pred = x + (A23 − cm_h(x)), 강도 1)  (CB2 의 REF27 과 같음)
  HARD: A23 >= 1.0 일 때만 앵커 (1.0 = 고EC 정의, 하루 평균 >= 1.0; 6.934·nh1 과 같은 미리 정해진 값)
  SOFT: 앵커 x clip((A23 − .8)/.4, 0, 1)  (TX3 의 EC 게이트 상수 .8/.4 그대로)
판정(EC 분야 규칙, 안마다, 본페로니 2 -> alpha .0125): P2LOO·EL1 3시드 2차 dSSE < 0 & (P2LOO 블록 P(악화) < .0125 OR 손해 2차 기록-칸 0) & DIAG10 손해 감시(전체 dSSE > 0 이고 '나음' 비율 < .025 이면 FAIL).
서술: 2차 보통/고EC, 앵커가 꺼진 기록 수, 손해 기록.
Run:  cd 집/클로드/research && PYTHONPATH="" python -u ecg1_ec_anchor_level_gate_v1.py
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.argv = ["x"]; sys.path.insert(0, HERE)
src = open(os.path.join(HERE, "cb2_ec_anchor_shaping_v1.py"), encoding="utf-8").read()
exec(src[:src.index("WS = tuple(")])
__file__ = os.path.join(HERE, "ecg1_ec_anchor_level_gate_v1.py")
y = R.y.values
def anchor(x, gfun):
    out = x.copy(); n_on, n_off = 0, 0
    for (v, k, r), ii in GRP:
        if r not in FL3 or FL3[r][0] in FOLDREC[(v, k)] or FL3[r][1] < .8 or DT[r] > 1.5: continue
        A = EC_[idx_[FL3[r][0]], 23]; wgt = gfun(A); xx = x[ii]; out[ii] = xx + wgt * (A - np.cumsum(xx) / np.arange(1, len(xx) + 1))
        n_on += wgt > 0; n_off += wgt == 0
    return out, n_on, n_off
G = {"REF": lambda A: 1.0, "HARD": lambda A: float(A >= 1.0), "SOFT": lambda A: float(np.clip((A - .8) / .4, 0, 1))}
P = {}
for s in SEEDS:
    nh = (R[f"A1_{s}"].values - .1 * mt) / .9
    for nm, gf in G.items():
        P[(nm, s)], on, off = anchor(nh, gf)
        if s == SEEDS[0]: print(f"{nm}: anchored record-validator groups on {on} / off {off}")
rng = np.random.default_rng(20270158)
for nm in ("HARD", "SOFT"):
    rows, blocks, recd = [], [], []
    for v in ("P2LOO", "EL1", "DIAG10"):
        g = (R.validator == v).values; yy = y[g]; p2 = R.p2.values[g]; hi = R.hi.values[g].astype(bool); key = (R.farm[g] + "_" + (R.day[g] // 5).astype(str)).values
        for s in SEEDS:
            a, b = P[(nm, s)][g], P[("REF", s)][g]; dd = (a - yy) ** 2 - (b - yy) ** 2
            f = lambda m: 100 * (rm((a - yy)[m]) / rm((b - yy)[m]) - 1) if m.any() else np.nan
            rows.append(dict(val=v, seed=s, dSSE_p2=dd[p2].sum(), dSSE_all=dd.sum(), rel_p2=f(p2), p2_norm=f(p2 & ~hi), p2_hi=f(p2 & hi), rel_all=f(np.ones(len(yy), bool))))
            if v == "P2LOO": blocks.append(pd.Series(dd[p2]).groupby(key[p2]).mean())
            if v != "DIAG10":
                for r_, xx in pd.Series(dd[p2]).groupby(R.rec.values[g][p2]).sum().items(): recd.append(dict(val=v, seed=s, rec=r_, d=xx))
    S = pd.DataFrame(rows); pd.set_option("display.width", 200); print(f"\n== {nm}"); print(S.round(4).to_string(index=False))
    xb = pd.concat(blocks, axis=1).mean(axis=1).values; pP = float((xb[rng.integers(0, len(xb), (20000, len(xb)))].mean(1) >= 0).mean())
    RC = pd.DataFrame(recd); harmed = int((RC.groupby(["val", "seed", "rec"]).d.sum() > 1e-6).sum()); helped = int((RC.groupby(["val", "seed", "rec"]).d.sum() < -1e-6).sum())
    D = S[S.val == "DIAG10"]; gD = (R.validator == "DIAG10").values; keyD = (R.farm[gD] + "_" + (R.day[gD] // 5).astype(str)).values
    ddD = np.mean([(P[(nm, s)][gD] - y[gD]) ** 2 - (P[("REF", s)][gD] - y[gD]) ** 2 for s in SEEDS], 0); xd = pd.Series(ddD).groupby(keyD).mean().values
    pbD = float((xd[rng.integers(0, len(xd), (20000, len(xd)))].mean(1) < 0).mean()); guard = not (D.dSSE_all.mean() > 0 and pbD < .025)
    main = S[S.val != "DIAG10"]; c1 = bool((main.dSSE_p2 < 0).all())
    print(f"  changed record-cells helped {helped} harmed {harmed}")
    print(f"ECG1 {nm}: P2LOO/EL1 6 cells p2 better {c1} | P2LOO P {pP:.4f} | harmed {harmed} | DIAG10 dSSE {D.dSSE_all.mean():+.4f} guard ok {guard} -> {'PASS' if c1 and (pP < .0125 or harmed == 0) and guard else 'FAIL'}")
