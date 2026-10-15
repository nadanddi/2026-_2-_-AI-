# -*- coding: utf-8 -*-
"""KF3 (2026-10-15 집 클로드; 5점모델_설계_v1.md 2층 — 동 상태 추적, 온도).  사전 고정(커밋 후 실행).  학습 없음(저장 셀 + 연결표).
가설 H2: 동 차이(0시 배지 − 반응 모델)는 천천히 변하는 상태 → 어제(들) 상태를 칼만식으로 전파해 오늘 0시 차이를 추정.
0층 연결 = FLN(6.932: 현장형 링커 + '전날 없음', local/fln1_train_links.csv; 대상 같은 농장 ±1 정답 숨김·평가 후보 제외로 만든 학습표).
  KF2 와 다른 점: v15R·HP2·DMT3e 달력 날짜 확인(대회 구조) 없음 — 연결 신뢰는 FLN 확률(전날 없음 포함)만.
판정 배치 = 처음 쓰는 N·CK(OS1 os1_base_ckpt, BASE 1616/1717 x PFN A/B = 4셀; 거친 날 제외).
기준 B = 셀 q + K8 앵커(FLN p ≥ .8, 전날이 그 폴드 학습 쪽 정답 기록, |dT_air| ≤ 1.5: q_h + BETA[h]·clip(A23 − .36 − q0, ±2)) + TX3(6.931).
후보 KF3 = B 에서 K8 앵커가 없는 기록에만:  사슬 r → p1 → p2 → p3 (FLN 짝, 최대 3단계, 누적 P = Π p ≥ .5), 끝이 그 폴드 학습 쪽 정답 기록이어야 함;
  중간 기록은 그 폴드 검증 기록(반응 모델 예측 q̄ = 4셀 평균 있음)이어야 함(제외 구역 기록이면 중단).
  끝 정답 A23, 분산 v = L2 .342;  중간 기록마다 est23 ← q̄_m23 + B23·((est23 − .36) − q̄_m0),  v ← B23²·v + H2 .239   (KF1 측정 상수 그대로).
  g0 = (est23 − .36) − q̄_r0;  보정 = clip(S2/(S2 + v) · P · g0, ±2),  S2 .549;  pred_h = q_h + (BETA[h]/BETA[0]) · 보정 + TX3.
판정(단일 안 α .025): N·CK 각각 4셀 모두 B 보다 개선 & 각각 농장 x 5일 블록 P(악화) < .025.   서술: 발동 기록 수(단계별), 추운 행·2차 분해.
Run:  cd 집/클로드/research && PYTHONPATH="" python -u kf3_field_state_kalman_temp_v1.py
"""
import os, sys, pickle, importlib.util
HERE = os.path.dirname(os.path.abspath(__file__)); sys.argv = ["x"]
sys.path.insert(0, os.path.join(HERE, "linker_v7")); sys.path.insert(0, HERE)
import env  # noqa: F401
from feat7 import *  # noqa: F401,F403
import numpy as np, pandas as pd
def load(name, file):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, file)); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
O1 = load("os1", "os1_temp_regime_split_v1.py"); B = load("tc1b", "tc1b_chain_history_common_train_v1.py")
BETA = np.load(os.path.join(env.LOCAL, "linker_v7", "ta8_beta.npy")); SHAPE = BETA / BETA[0]; B23 = float(BETA[23])
S2, L2, H2, BIAS = .549, .342, .239, .36
IDX = {rid(i): i for i in range(N)}
FT = pd.read_csv(os.path.join(env.LOCAL, "fln1_train_links.csv")); LINK = {r: (p, q) for r, p, q in zip(FT.rec, FT.prev, FT.p)}
RD = O1.RD; rm = lambda e: float(np.sqrt(np.mean(np.square(e))))
_R = pd.read_csv(os.path.join(HERE, "nh1_rows_v1.csv.gz")); _R = _R[_R.validator == "DIAG10"]
_MT = pickle.load(open(os.path.join(env.LOCAL, "nh1_mt1_v1.pkl"), "rb")); _mt = np.array([_MT[(v, k)][r][h] for v, k, r, h in zip(_R.validator, _R.fold, _R.rec, _R.hour)])
_R["NH"] = np.mean([(_R[f"A1_{s}"].values - .1 * _mt) / .9 for s in (1357, 2468, 3579)], axis=0)
Eh = _R.groupby(["rec", "hour"]).NH.mean().reset_index().sort_values(["rec", "hour"]); Eh["cm"] = Eh.groupby("rec").NH.transform(lambda s: s.expanding().mean())
CM = {(r, int(h)): c for r, h, c in zip(Eh.rec, Eh.hour, Eh.cm)}
T27 = load("t27", "tt27_chain_corrector_layoutL_v1.py"); W = T27.W; lab = W.worlds()[0]; LF = O1.layout_folds(W, lab)
rng = np.random.default_rng(20270118); ok = True
for L in O1.LAYS:
    M = pd.concat([pd.read_csv(os.path.join(O1.CKB, f)) for f in sorted(os.listdir(O1.CKB)) if f.startswith(L + "_")], ignore_index=True)
    M = M[np.array([(f, d) not in RD for f, d in zip(M.farm, M.day)])].reset_index(drop=True)
    y = M.sub_temp.values; h = M.hour.values; key = (M.farm + "_" + M.day.map("{:03d}".format)).values; recs = sorted(set(key)); pos = {r: np.where(key == r)[0] for r in recs}
    cg = np.nan_to_num(np.clip((10 - M.in_temp.values) / 2, 0, 1), nan=0.0); cm = np.array([CM.get((k, int(hh)), np.nan) for k, hh in zip(key, h)])
    tx3 = .25 * np.clip((np.nan_to_num(cm, nan=0) - .8) / .4, 0, 1) * cg
    TRS, VAL = {}, {}
    for k, fd in enumerate(LF[L]):
        tm, vm = O1.masks(lab, fd); trs = set(lab.farm[tm] + "_" + lab.day[tm].map("{:03d}".format)); vs = set(lab.farm[vm] + "_" + lab.day[vm].map("{:03d}".format))
        for r in vs: TRS[r], VAL[r] = trs, vs
    cells = [B.w40(M, "base_R_%d" % s, "codex_R", fam) for s in O1.SEEDS for fam in "AB"]; qbar = np.mean(cells, 0)
    Q = {r: np.array([qbar[pos[r]][h[pos[r]] == t][0] if (h[pos[r]] == t).any() else np.nan for t in range(24)]) for r in recs}
    K8, KF, log = {}, {}, []
    for r in recs:
        trs, vs = TRS[r], VAL[r]
        if r in LINK:
            pv, p = LINK[r]
            if p >= .8 and pv in trs and np.isfinite(ST[IDX[pv], 23]) and abs(IN_[IDX[r], 0, 0] - IN_[IDX[pv], 23, 0]) <= 1.5:
                K8[r] = float(ST[IDX[pv], 23]); continue
        chain, P, cur = [], 1.0, r
        for step in range(3):
            if cur not in LINK: chain = None; break
            pv, p = LINK[cur]; P *= p
            if P < .5: chain = None; break
            chain.append(pv)
            if pv in trs and np.isfinite(ST[IDX[pv], 23]): break
            if pv not in vs or pv not in Q: chain = None; break
            cur = pv
        if not chain or not (chain[-1] in trs and np.isfinite(ST[IDX[chain[-1]], 23])): continue
        est, v = float(ST[IDX[chain[-1]], 23]), L2
        for m in reversed(chain[:-1]): est = Q[m][23] + B23 * ((est - BIAS) - Q[m][0]); v = B23 ** 2 * v + H2
        if not np.isfinite(est) or not np.isfinite(Q[r][0]): continue
        KF[r] = float(np.clip(S2 / (S2 + v) * P * ((est - BIAS) - Q[r][0]), -2, 2)); log.append(len(chain))
    P_B, P_K = [], []
    for c in cells:
        b = c + tx3; k = b.copy()
        for r, A in K8.items():
            ix = pos[r]; q0 = c[ix][h[ix] == 0]
            if len(q0): b[ix] += BETA[h[ix]] * np.clip(A - BIAS - q0[0], -2, 2); k[ix] = b[ix]
        for r, g in KF.items(): ix = pos[r]; k[ix] = c[ix] + tx3[ix] + SHAPE[h[ix]] * g
        P_B.append(b); P_K.append(k)
    ch = [rm(kk - y) / rm(bb - y) - 1 for kk, bb in zip(P_K, P_B)]; bk = (M.farm + "_" + (M.day // 5).astype(str)).values
    dd = pd.Series(np.mean([(kk - y) ** 2 - (bb - y) ** 2 for kk, bb in zip(P_K, P_B)], 0)).groupby(bk).mean().values
    mb = dd[rng.integers(0, len(dd), (20000, len(dd)))].mean(1); pw = float((mb >= 0).mean())
    cold = (M.in_temp <= 8).values; p2 = M.day.values >= 179
    sub = lambda m: 100 * np.mean([rm((kk - y)[m]) / rm((bb - y)[m]) - 1 for kk, bb in zip(P_K, P_B)]) if m.any() else np.nan
    c_ok = all(x < 0 for x in ch) and pw < .025; ok &= c_ok
    print(f"{L:3s} KF3 vs B {' '.join('%+.3f' % (100 * x) for x in ch)} | P(worse) {pw:.4f} -> {'ok' if c_ok else 'FAIL'} | cold {sub(cold):+.2f}% p2 {sub(p2):+.2f}% | "
          f"K8 anchors {len(K8)} | KF fired {len(KF)} by steps {pd.Series(log).value_counts().sort_index().to_dict()} (records {len(recs)})", flush=True)
print("KF3:", "ADOPT" if ok else "FAIL")
