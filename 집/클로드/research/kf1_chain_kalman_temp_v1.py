# -*- coding: utf-8 -*-
"""KF1 (2026-10-13 집 클로드; 사용자: 'RNN 류 장점(상태 전달·단서 누적·불확실성·빠진 관측 처리)을 살려서, 온도부터').  사전 고정(커밋 후 실행).  온도.
같은 동 사슬 칼만 필터(1차원 상태 = 오늘 0시 배지 차이 g0 = 진짜 0시 − 기준 q0; 13회차 앵커가 쓰는 양):
  사슬 = v15R(거울 G0) 1순위 짝을 최대 3단계(누적 짝 확신 P = Π p0 ≥ .3 일 때만 계속).
  정답 보이는 짝(학습 기록 & 시험 폴드에서 숨김 아님) 만나면: 그 날 23시 배지 A → 다음 날 진짜 0시 추정 = A − .36(13회차 편향), 분산 σ_L².
  정답 없는 짝(같은 폴드 숨김 기록 = 평가의 같은 ID 앞 평가 기록 흉내): 그 날 0시 차이 추정 ĝ, 분산 v 에서 그 날 23시 추정 = q23 + β23·ĝ(β23 = .2355, 측정 감쇠),
     분산 β23²·v + σ_h² → 다음 날 0시 추정 = (그 23시 추정) − .36.  (사슬 시작점에 정답이 없으면 ĝ = 0, v = σ_s².)
  오늘: ĝ0 = (전날 23시 추정 − .36) − q0, 분산 v0;  축소 K = σ_s²/(σ_s² + v0) × P(누적 짝 확신);  pred_h = q_h + BETA_h·clip(K·ĝ0, ±2).
  매개변수 3개(1차 DIAG10 OOF·깨끗한 연결에서 측정, 맞추지 않음): σ_s² = var(진짜 0시 − q0), σ_L² = var(A − .36 − 진짜 0시)(정답 1단계 연결), σ_h² = var(진짜 23시 − q23).
안(2개, α .0125):  KFa = 14회차형(v7 + TNA8 앵커) 그대로 + 앵커 없는 기록에 KF 추정;   KFb = 모든 기록 KF(앵커 대체, 짝 확신 가중).
기준 = 14회차형.  판정(온도 범위 ALL): 주 DIAG10·EXT10 모두 4칸 개선 & 농장×5일 블록 부트스트랩 P(악화) < .0125; 나머지 배치는 칸 평균 악화 & '후보가 나음' 비율 < .0125 면 FAIL.
서술: 단계 수별 발동 기록 수, 측정한 σ 들.
"""
import os, sys, importlib.util
import env  # noqa: F401
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.argv = ["x"]
src = open(os.path.join(HERE, "cr1_chain_gru_temp_v1.py"), encoding="utf-8").read()
head = src[:src.index("V = [\"in_temp\", \"in_hum\"")].replace("import env_extra_gpu  # noqa: F401\n", "").replace("import torch, torch.nn as nn\n", "").replace('DEV = "cuda" if torch.cuda.is_available() else "cpu"', "DEV = None")
exec(head)   # A7(BETA), LC (v7+TNA8), PM (v15R mirror), ST, idx, TRAIN, FARM, NUM, rm, L7f, v7set
rng = np.random.default_rng(20261125); B23 = float(A7.BETA[23]); BIAS = .36
TLx = pd.read_pickle(os.path.join(_ns["S3"], "truth.pkl")) if "_ns" in dir() and "S3" in _ns else None
FR = A7.A2.A1.frames()
def layout_maps(M, cells):
    key = (M.farm + "_" + M.day.map("{:03d}".format)).values; h = M.hour.values; y = M.sub_temp.values
    kb, Ab = A7.build(M, LC); anc = [A7.correct(M, c, kb, Ab, .8, "const", "meas", 2, 1.5) for c in cells]
    recs = sorted(set(key)); pos = {r: np.where(key == r)[0] for r in recs}
    def per_rec(arr): return {r: np.array([arr[pos[r]][h[pos[r]] == t][0] if (h[pos[r]] == t).any() else np.nan for t in range(24)]) for r in recs}
    return key, h, y, anc, recs, pos, per_rec(np.mean(cells, 0)), per_rec(y)
# ---- measure the 3 variances on DIAG10 pass-1 (cell-mean OOF q, clean labelled 1-step links from v15R picks that are labelled & p0 >= .8) ----
key, h, y, anc, recs, pos, Q, Y = layout_maps(*FR["DIAG10"])
p1 = [r for r in recs if NUM[idx[r]] < 179]
g0 = np.array([Y[r][0] - Q[r][0] for r in p1 if np.isfinite(Y[r][0]) and np.isfinite(Q[r][0])]); S2 = float(np.var(g0))
e23 = np.array([Y[r][23] - Q[r][23] for r in p1 if np.isfinite(Y[r][23]) and np.isfinite(Q[r][23])]); H2 = float(np.var(e23))
la = [ST[idx[PM.at[r, "prev0"]], 23] - BIAS - Y[r][0] for r in p1 if r in PM.index and isinstance(PM.at[r, "prev0"], str) and bool(PM.at[r, "lab0"]) and PM.at[r, "p0"] >= .8 and np.isfinite(Y[r][0])]
L2 = float(np.var(la)); print(f"measured: sigma_s^2 {S2:.3f}  sigma_L^2 {L2:.3f} (n {len(la)})  sigma_h^2 {H2:.3f}  beta23 {B23:.3f}", flush=True)
def kf_gap(r, Qm, hidden, inlayout):
    """returns (K*g0_hat, steps) or (None, 0)"""
    chain = []; cur = r; P = 1.0
    for k in range(3):
        if cur not in PM.index or not isinstance(PM.at[cur, "prev0"], str): break
        pv = PM.at[cur, "prev0"]; P *= float(PM.at[cur, "p0"])
        if P < .3: break
        vis = TRAIN[idx[pv]] and pv not in hidden and np.isfinite(ST[idx[pv], 23])
        chain.append((pv, vis))
        if vis: break
        if pv not in inlayout: break
        cur = pv
    if not chain or not chain[-1][1]: return None, 0          # require the chain to end on a visible label (information source)
    pv, _ = chain[-1]; est23 = ST[idx[pv], 23]; v = L2      # estimate of the next day's true 0h = est23 - BIAS, variance L2
    for pv2, _ in reversed(chain[:-1]):                     # unlabelled hops (oldest first after the labelled one)
        g = (est23 - BIAS) - Qm[pv2][0]; est23 = Qm[pv2][23] + B23 * g; v = B23 ** 2 * v + H2
    g0h = (est23 - BIAS) - Qm[r][0]; K = S2 / (S2 + v) * P
    return float(np.clip(K * g0h, -2, 2)), len(chain)
res, blocks, steps_log = [], {}, []
for L, (M, cells) in FR.items():
    key, h, y, anc, recs, pos, Q, Y = layout_maps(M, cells)
    fold_of = {r: M.fold.values[pos[r][0]] for r in recs}; hidden_of = {}
    for r in recs: hidden_of.setdefault(fold_of[r], set()).add(r)
    anchored = set(LC.rec); inl = set(recs)
    KFG = {}
    for r in recs:
        g, k = kf_gap(r, Q, hidden_of[fold_of[r]], inl)
        if g is not None: KFG[r] = (g, k); steps_log.append(dict(layout=L, rec=r, steps=k, anchored=r in anchored))
    for ci, c in enumerate(cells):
        base = anc[ci]; ka, kb_ = base.copy(), c.copy()
        for r, (g, k) in KFG.items():
            ix = pos[r]; add = A7.BETA[h[ix]] * g
            if r not in anchored: ka[ix] = base[ix] + add
            kb_[ix] = c[ix] + add
        for a, new in (("KFa", ka), ("KFb", kb_)):
            p2 = M.day.values >= 179
            res.append(dict(layout=L, cell=ci, arm=a, pct=100 * (rm(new - y) / rm(base - y) - 1), p2=100 * (rm((new - y)[p2]) / rm((base - y)[p2]) - 1)))
            if ci == 0:
                bk = M.farm + "_" + (M.day // 5).astype(str); blocks[(L, a)] = pd.Series((new - y) ** 2 - (base - y) ** 2).groupby(bk.values).mean()
    print("done", L, flush=True)
SL = pd.DataFrame(steps_log); pd.set_option("display.width", 220)
print("KF fired records by steps (DIAG10):", SL[SL.layout == "DIAG10"].groupby(["steps", "anchored"]).size().to_dict())
Sx = pd.DataFrame(res); print(Sx.pivot_table(index=["arm", "layout"], columns="cell", values="pct").round(3).to_string())
print(Sx.groupby(["arm", "layout"]).p2.mean().round(3).unstack(0).to_string())
def bootP(L, a):
    v = blocks[(L, a)].values; ix = rng.integers(0, len(v), (20000, len(v))); m = v[ix].mean(1); return float((m >= 0).mean()), float((m < 0).mean())
for a in ("KFa", "KFb"):
    ok = True; msg = []
    for L in sorted(set(Sx.layout)):
        c_ = Sx[(Sx.arm == a) & (Sx.layout == L)].pct; pw, pb = bootP(L, a)
        if L in ("DIAG10", "EXT10"):
            c = bool((c_ < 0).all()) and pw < .0125; ok &= c; msg.append(f"{L}[primary] cells {int((c_ < 0).sum())}/4 P(worse) {pw:.4f} -> {'ok' if c else 'FAIL'}")
        else:
            bad = c_.mean() > 0 and pb < .0125; ok &= not bad; msg.append(f"{L} mean {c_.mean():+.3f}% P(better) {pb:.3f}{' SIGNIF WORSE' if bad else ''}")
    print(f"\n== KF1 {a}:"); print("\n".join("   " + m for m in msg)); print(f"KF1 {a}:", "ADOPT" if ok else "FAIL")
