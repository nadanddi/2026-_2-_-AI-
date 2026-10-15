# -*- coding: utf-8 -*-
"""CMA1 (2026-10-15 집 클로드; 사용자 '확신 없는 어제의 오차를 줄이는 방향').  사전 고정(커밋 후 실행).  온도 + EC.  학습 없음(저장 예측 + 링커 후보표).
배경: 16 → 27 차이 분해(6.942, 서술) — 두 열 모두 '현장형 링커가 확신한 어제만' 쓰는 쪽으로 바뀐 기록에 변화가 몰림. 확신 없는 어제를 하나 골라 쓰면 틀릴 때 손해가 큼(AC1·AT1).
  AG6(6.709, v7 시절): 후보 소프트 가중 평균 '보조 층' 4칸 모두 개선·2차 −1.6% 이나 P .05~.07.
안: K8 앵커(FL p ≥ .8 정답 전날)가 없는 기록에만 — FL 링커(fl3_w.npy, 학습표와 같은 숨김: 대상 같은 농장 ±1 정답 숨김·평가 후보 제외)의 0시 후보 확률 p_i(전체 softmax) 중
  '그 폴드에서 정답 있는 학습 기록이고 |dT_air| ≤ 1.5' 인 후보 L 만 사용:  m = Σ_L p_i,  Ā = Σ p_i A_i / m,  V = Σ p_i (A_i − Ā)² / m.
  온도: A = 23시 배지, c = .342 / (.342 + V) (자정 전달 잡음 L2, KF1 측정),  pred_h = q_h + BETA[h] · m · c · clip(Ā − .36 − q0, ±2).
  EC  : A = 23시 EC,  c = 1 / (1 + V / Ā²)(상대 퍼짐),  pred_h = q_h + m · c · (Ā − cm_h(q)).
  m < .2 이면 적용 안 함(미리 고정).  새로 맞춘 상수 없음.
판정(각 열 단일 안 α .025):
  온도: 배치 N·CK2(OS1 v2 저장 셀; 기준 = 셀 + FL K8 + TX3, ASM1 의 R27 과 같음) — 8셀 모두 개선 & 두 배치 합친 블록 P(악화) < .025.
  EC : 기준 = NH + FL K8(27번 EC) — EC 분야 규칙(P2LOO·EL1 3시드 2차 dSSE < 0 & (P2LOO 블록 P < .025 OR 손해 기록-칸 0) & DIAG10 전체 손해 아님).
서술: 발동 기록 수, 평균 m·c, 2차·추운 행.
Run:  cd 집/클로드/research && PYTHONPATH="" python -u cma1_candidate_mixture_anchor_v1.py
"""
import os, sys, pickle
HERE = os.path.dirname(os.path.abspath(__file__)); sys.argv = ["x"]
sys.path.insert(0, os.path.join(HERE, "linker_v7")); sys.path.insert(0, HERE)
import env  # noqa: F401
from feat7 import *  # noqa: F401,F403
import numpy as np, pandas as pd
FLC = ["oj"] + [f"v{j}" for j in range(8)] + ["excl", "xf"]; WFL = np.load(os.path.join(env.LOCAL, "fl3_w.npy"))
IDX = {rid(i): i for i in range(N)}; rm = lambda e: float(np.sqrt(np.mean(np.square(e))))
CPATH = os.path.join(env.LOCAL, "cma1_candidates.pkl")
if os.path.exists(CPATH): CAND = pickle.load(open(CPATH, "rb"))
else:
    CAND = {}
    for b in range(N):
        if not TRAIN[b] or FARM[b] not in ("F13", "F47"): continue
        lab = TRAIN.copy(); lab[(FARM == FARM[b]) & (np.abs(NUM - NUM[b]) <= 1)] = False
        df, _ = features7(b, 0, lab, removed=set(TEST_IDX))
        if len(df) == 0: continue
        s = df[FLC].values.astype(float) @ WFL; p = np.exp(s - s.max()); p /= p.sum()
        CAND[rid(b)] = [(rid(int(a)), float(q)) for a, q in zip(df.a.values, p) if q >= .01]
    pickle.dump(CAND, open(CPATH, "wb"))
print("candidate tables", len(CAND), flush=True)
FLT = pd.read_csv(os.path.join(env.LOCAL, "fl3_train_links.csv")); FLK = {r: (p, q) for r, p, q in zip(FLT.rec, FLT.prev, FLT.p)}
dTok = lambda r, a: abs(IN_[IDX[r], 0, 0] - IN_[IDX[a], 23, 0]) <= 1.5
def mix(r, usable, val):
    L = [(a, q) for a, q in CAND.get(r, []) if usable(a) and dTok(r, a) and np.isfinite(val(a))]
    if not L: return None
    m = sum(q for a, q in L)
    if m < .2: return None
    A = np.array([val(a) for a, q in L]); P = np.array([q for a, q in L]); Ab = float((P * A).sum() / m); V = float((P * (A - Ab) ** 2).sum() / m)
    return m, Ab, V
# ---------------- temperature (N, CK2) ----------------
src = open(os.path.join(HERE, "kf3_field_state_kalman_temp_v2.py"), encoding="utf-8").read()
exec(src[:src.index("rng = np.random.default_rng(20270118)")]); __file__ = os.path.join(HERE, "cma1_candidate_mixture_anchor_v1.py")
rng = np.random.default_rng(20270124); okT = True; diffs = []
for L in O1.LAYS:
    M = pd.concat([pd.read_csv(os.path.join(O1.CKB, f)) for f in sorted(os.listdir(O1.CKB)) if f.startswith(L + "_")], ignore_index=True)
    M = M[np.array([(f, d) not in RD for f, d in zip(M.farm, M.day)])].reset_index(drop=True)
    y = M.sub_temp.values; h = M.hour.values; key = (M.farm + "_" + M.day.map("{:03d}".format)).values; pos = {r: np.where(key == r)[0] for r in set(key)}
    cg = np.nan_to_num(np.clip((10 - M.in_temp.values) / 2, 0, 1), nan=0.0); cm = np.array([CM.get((k, int(hh)), np.nan) for k, hh in zip(key, h)])
    tx3 = .25 * np.clip((np.nan_to_num(cm, nan=0) - .8) / .4, 0, 1) * cg
    TRS = {}
    for k, fd in enumerate(LF[L]):
        tm, vm = O1.masks(lab, fd); trs = set(lab.farm[tm] + "_" + lab.day[tm].map("{:03d}".format))
        for r in set(lab.farm[vm] + "_" + lab.day[vm].map("{:03d}".format)): TRS[r] = trs
    K8 = {r: float(ST[IDX[FLK[r][0]], 23]) for r in pos if r in FLK and FLK[r][1] >= .8 and FLK[r][0] in TRS[r] and np.isfinite(ST[IDX[FLK[r][0]], 23]) and dTok(r, FLK[r][0])}
    MX = {}
    for r in pos:
        if r in K8: continue
        o = mix(r, lambda a, r=r: a in TRS[r], lambda a: ST[IDX[a], 23])
        if o: MX[r] = o
    PB, PC = [], []
    for s in O1.SEEDS:
        for fam in "AB":
            c = B.w40(M, "base_R_%d" % s, "codex_R", fam); b = c + tx3; n = b.copy()
            for r, A in K8.items():
                ix = pos[r]; q0 = c[ix][h[ix] == 0]
                if len(q0): b[ix] += BETA[h[ix]] * np.clip(A - .36 - q0[0], -2, 2); n[ix] = b[ix]
            for r, (m, Ab, V) in MX.items():
                ix = pos[r]; q0 = c[ix][h[ix] == 0]
                if len(q0): n[ix] = b[ix] + BETA[h[ix]] * m * (.342 / (.342 + V)) * np.clip(Ab - .36 - q0[0], -2, 2)
            PB.append(b); PC.append(n)
    ch = [rm(n - y) / rm(b - y) - 1 for n, b in zip(PC, PB)]; okT &= all(x < 0 for x in ch)
    kb = (L + "_" + M.farm + "_" + (M.day // 5).astype(str)).values; diffs.append(pd.Series(np.mean([(n - y) ** 2 - (b - y) ** 2 for n, b in zip(PC, PB)], 0)).groupby(kb).mean())
    cold = (M.in_temp <= 8).values; p2 = M.day.values >= 179
    sub = lambda mk: 100 * np.mean([rm((n - y)[mk]) / rm((b - y)[mk]) - 1 for n, b in zip(PC, PB)]) if mk.any() else np.nan
    print(f"[T] {L:3s} CMA vs R27 {' '.join('%+.3f' % (100 * x) for x in ch)} | p2 {sub(p2):+.2f}% cold {sub(cold):+.2f}% | K8 {len(K8)} | mixed {len(MX)} "
          f"(mean m {np.mean([v[0] for v in MX.values()]):.2f}, mean c {np.mean([.342 / (.342 + v[2]) for v in MX.values()]):.2f}) of {len(pos)}", flush=True)
S = pd.concat(diffs).values; PT = float((S[rng.integers(0, len(S), (20000, len(S)))].mean(1) >= 0).mean())
print(f"[T] pooled P(worse) {PT:.4f} -> CMA1 temp: {'ADOPT' if okT and PT < .025 else 'FAIL'}", flush=True)
# ---------------- EC ----------------
src = open(os.path.join(HERE, "ev3l_ec_linked_yesterday_gru_v1.py"), encoding="utf-8").read()
G = {"__file__": os.path.join(HERE, "ev3l_ec_linked_yesterday_gru_v1.py"), "__name__": "ev3lcma"}; exec(src[:src.index("\nO = pd.read_csv")], G)
EC_, idx_, TR_ = G["EC"], G["idx"], G["TRAIN"]
R2 = pd.read_csv(os.path.join(HERE, "nh1_rows_v1.csv.gz")); SEEDS = (1357, 2468, 3579)
MT = pickle.load(open(os.path.join(env.LOCAL, "nh1_mt1_v1.pkl"), "rb")); mt = np.array([MT[(v, k)][r][hh] for v, k, r, hh in zip(R2.validator, R2.fold, R2.rec, R2.hour)])
FOLDREC = {vk: set(Gd.rec) for vk, Gd in R2.groupby(["validator", "fold"])}; GRP = list(R2.groupby(["validator", "fold", "rec"]))
fired = {}
for s in SEEDS:
    R2[f"NH_{s}"] = (R2[f"A1_{s}"] - .1 * mt) / .9; o8 = R2[f"NH_{s}"].values.copy(); oC = o8.copy()
    for (v, k, r), gg in GRP:
        ii = gg.sort_values("hour").index.values; x = R2.loc[ii, f"NH_{s}"].values; pm = np.cumsum(x) / np.arange(1, len(x) + 1); fr = FOLDREC[(v, k)]
        if r in FLK and FLK[r][1] >= .8 and FLK[r][0] in idx_ and TR_[idx_[FLK[r][0]]] and FLK[r][0] not in fr and np.isfinite(EC_[idx_[FLK[r][0]], 23]) and dTok(r, FLK[r][0]):
            o8[ii] = x + (EC_[idx_[FLK[r][0]], 23] - pm); oC[ii] = o8[ii]; continue
        o = mix(r, lambda a, fr=fr: a in idx_ and bool(TR_[idx_[a]]) and a not in fr, lambda a: EC_[idx_[a], 23] if a in idx_ else np.nan)
        if not o: continue
        m, Ab, V = o; oC[ii] = x + m * (1 / (1 + V / Ab ** 2)) * (Ab - pm); fired[(v, r)] = (m, 1 / (1 + V / Ab ** 2))
    R2[f"K8_{s}"], R2[f"CM_{s}"] = o8, oC
print("[EC] mixed records by validator:", pd.Series([v for v, r in fired]).value_counts().to_dict(), flush=True)
rows, blocks, recd = [], [], []
for v in ("DIAG10", "EL1", "P2LOO"):
    gq = R2[R2.validator == v]; y = gq.y.values; p2 = gq.p2.values; key = (gq.farm + "_" + (gq.day // 5).astype(str)).values
    for s in SEEDS:
        dd = (gq[f"CM_{s}"].values - y) ** 2 - (gq[f"K8_{s}"].values - y) ** 2
        rows.append(dict(val=v, seed=s, dSSE_p2=float(dd[p2].sum()), dSSE_all=float(dd.sum()), vsK8_p2=100 * (rm((gq[f"CM_{s}"].values - y)[p2]) / rm((gq[f"K8_{s}"].values - y)[p2]) - 1),
                         vsK8_all=100 * (rm(gq[f"CM_{s}"].values - y) / rm(gq[f"K8_{s}"].values - y) - 1)))
        if v == "P2LOO": blocks.append(pd.Series(dd[p2]).groupby(key[p2]).mean())
        for r_, xx in pd.Series(dd[p2]).groupby(gq.rec.values[p2]).sum().items(): recd.append(dict(val=v, seed=s, rec=r_, d=xx))
SX = pd.DataFrame(rows); print(SX.round(3).to_string(index=False))
main = SX[SX.val.isin(["P2LOO", "EL1"])]; c1 = bool((main.dSSE_p2 < 0).all()); xb = pd.concat(blocks, axis=1).mean(axis=1).values
pP = float((xb[np.random.default_rng(20270125).integers(0, len(xb), (20000, len(xb)))].mean(1) >= 0).mean()); RD_ = pd.DataFrame(recd)
nh_ = int((RD_[RD_.val.isin(["P2LOO", "EL1"])].groupby(["val", "seed", "rec"]).d.sum() > 1e-6).sum()); c3 = not (SX[SX.val == "DIAG10"].dSSE_all.mean() > 0)
print(f"[EC] CMA vs K8: cells {int((main.dSSE_p2 < 0).sum())}/6 | P2LOO P {pP:.4f} | harmed {nh_} | DIAG10 {SX[SX.val == 'DIAG10'].dSSE_all.mean():+.3f} -> CMA1 EC: {'ADOPT' if c1 and (pP < .025 or nh_ == 0) and c3 else 'FAIL'}")
