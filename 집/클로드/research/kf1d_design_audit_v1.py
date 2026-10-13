# -*- coding: utf-8 -*-
"""KF1D (2026-10-13 집 클로드; 사용자: '칼만 필터 설계가 제대로 됐는지 검사'). 서술·감사.
KF1(6.810)·KF1E(6.811) 의 KFa 발동 기록별로: 0시 차이 보정 크기 |K·ĝ0|, 짝의 날짜 일치(학습 전용 달력 DMT3e: 짝 날짜 = 대상 날짜 − 1 인가),
  사슬 단계, 짝 확신, 진짜 연결(깨끗한 진실)과 일치 여부, 그리고 그 기록의 ΔSSE(셀 평균) — DIAG10 과 EV 에서.
또 이중 축소 점검: 1단계 정답 짝에서 '진짜 0시 차이 vs 추정 ĝ0' 회귀 기울기(= 이상적 축소) 와 KF 의 K·β0 비교.
"""
import os, sys
import env  # noqa: F401
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.argv = ["x"]
src = open(os.path.join(HERE, "kf1_chain_kalman_temp_v1.py"), encoding="utf-8").read(); __file__ = os.path.join(HERE, "kf1_chain_kalman_temp_v1.py")
exec(src[:src.index("res, blocks, steps_log = [], {}, []")])   # layout_maps, kf_gap, PM, S2, L2, H2, LC, FR, A7, ST, idx, TRAIN, FARM, NUM
C = pd.read_csv(os.path.join(env.LOCAL, "dmt3e_E0_days.csv")); DAYT = dict(zip(C.rec, C.day)); SEGT = dict(zip(C.rec, C.seg))
from feat4 import X as OUTX  # noqa: E402
def tdate(r):
    i = idx[r]; vi = np.where(TRAIN)[0]; vi = vi[vi != i]; m = vi[np.all(np.abs(OUTX[vi, 0] - OUTX[i, 0]) < 1e-6, axis=1)]
    ds = [(DAYT.get(rid(j)), SEGT.get(rid(j))) for j in m if rid(j) in DAYT]
    return pd.Series(ds).mode().iloc[0] if ds else None
TL = pd.read_pickle(os.path.join(_ns["S3"], "truth.pkl")); CAL = pd.read_csv(os.path.join(HERE, "ec0h", "rise", "dm11_calendar_v1.csv")); D11 = dict(zip(CAL.rec, CAL.day))
CLEAN = {rid(int(b)): rid(int(a)) for a, b in zip(TL.a1, TL.b) if D11.get(rid(int(b)), np.nan) - D11.get(rid(int(a)), np.nan) == 1}
rows = []
M, cells = FR["DIAG10"]; key, h, y, anc, recs, pos, Q, Y = layout_maps(M, cells)
fold_of = {r: M.fold.values[pos[r][0]] for r in recs}; hidden_of = {}
for r in recs: hidden_of.setdefault(fold_of[r], set()).add(r)
anchored = set(LC.rec); base = np.mean(anc, 0)
for r in recs:
    if r in anchored: continue
    g, k = kf_gap(r, Q, hidden_of[fold_of[r]], set(recs))
    if g is None: continue
    ix = pos[r]; new = base[ix] + A7.BETA[h[ix]] * g; dsse = float(np.sum((new - y[ix]) ** 2) - np.sum((base[ix] - y[ix]) ** 2))
    pv = PM.at[r, "prev0"]; td = tdate(r); dok = None
    if td is not None and pv in DAYT: dok = (SEGT[pv] == td[1]) and (DAYT[pv] == td[0] - 1)
    true_g0 = Y[r][0] - Q[r][0]
    rows.append(dict(rec=r, steps=k, p0=PM.at[r, "p0"], move=g, absmove=abs(g), date_ok=dok, true_prev=(CLEAN.get(r) == pv) if r in CLEAN else None, dsse=dsse, true_g0=true_g0, p2=NUM[idx[r]] >= 179))
D = pd.DataFrame(rows); pd.set_option("display.width", 220)
print(f"DIAG10 KFa fired {len(D)} | total dSSE {D.dsse.sum():+.2f} (neg = better)")
print("by date consistency:", D.groupby(D.date_ok.astype(str)).dsse.agg(["size", "sum", "mean"]).round(3).to_dict("index"))
print("by first link truth:", D.groupby(D.true_prev.astype(str)).dsse.agg(["size", "sum", "mean"]).round(3).to_dict("index"))
D["mbin"] = pd.cut(D.absmove, [0, .25, .5, 1, 1.5, 2.01]); print("by |move|:", D.groupby("mbin").dsse.agg(["size", "sum", "mean"]).round(3).to_dict("index"))
print("by steps:", D.groupby("steps").dsse.agg(["size", "sum", "mean"]).round(3).to_dict("index"))
print("sign agreement move vs true 0h gap:", round(float((np.sign(D.move) == np.sign(D.true_g0)).mean()), 3), "| slope of true_g0 on move:", round(float(np.polyfit(D.move, D.true_g0, 1)[0]), 3))
print(D.sort_values("dsse").tail(6)[["rec", "steps", "p0", "move", "date_ok", "true_prev", "dsse"]].round(3).to_string(index=False))
# double shrink check on 1-step labelled confident links (pass-1)
p1 = [r for r in recs if NUM[idx[r]] < 179 and r in PM.index and isinstance(PM.at[r, "prev0"], str) and bool(PM.at[r, "lab0"]) and PM.at[r, "p0"] >= .8]
gh = np.array([ST[idx[PM.at[r, "prev0"]], 23] - .36 - Q[r][0] for r in p1]); gt = np.array([Y[r][0] - Q[r][0] for r in p1]); m = np.isfinite(gh) & np.isfinite(gt)
print(f"\ndouble-shrink check (pass-1 confident 1-step, n {m.sum()}): ideal slope true_g0~g_hat {np.polyfit(gh[m], gt[m], 1)[0]:.3f} | KF K(p=1) = {S2 / (S2 + L2):.3f} | K*BETA0 = {S2 / (S2 + L2) * A7.BETA[0]:.3f} | BETA0 alone {A7.BETA[0]:.3f}")
