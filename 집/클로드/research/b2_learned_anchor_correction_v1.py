# -*- coding: utf-8 -*-
"""B2 (2026-10-13 집 클로드; 사용자: '리스트 전부 다 해봐' — B2: 고정 감쇠 β 대신 앵커 보정량을 작은 모델이 배우게).  사전 고정(커밋 후 실행).  온도.
지금(13회차): pred_h = q_h + BETA_h · clip(A − .36 − q0, ±2)  (BETA 는 1차 구간에서 시각별로 잰 기울기).
새: 같은 앵커 기록 집합에서 보정량 Δ_h = f(gap, h, drift_in_h, drift_q_h, in_temp_h) 를 학습 — gap = clip(A − .36 − q0, ±2),
  drift_in_h = 대상 실내온도_h − 실내온도_0 (그 시각까지 입력, 인과), drift_q_h = q_h − q0.  목표 = y_h − q_h.
  학습 = DIAG10 계열 1차(번호 < 179) 앵커 기록 행(4칸 합침, v7 거울 CV 앵커: p ≥ .8·|dT| ≤ 1.5·1단계·정답) — 2차 행 학습에 안 씀.
안(2개 → α .0125): LIN = 시각 묶음 4개(0–3, 4–7, 8–15, 16–23)마다 Δ = gap·(b0 + b1·drift_in + b2·drift_q + b3·|gap|) 릿지(λ 1);
  GBM = HistGradientBoosting(깊이 3, 반복 200, 학습률 .05, L2 1, 시드 0)으로 Δ 직접.
판정(기준 = 13회차 공식, 같은 앵커 집합): DIAG10 4칸·EV 4칸 모두 2차 행 개선 AND ((두 검증기 2차 기록 부트스트랩 P < .0125) 또는 (두 검증기 2차 손해 기록 0)).
  EV 앵커 기록 = v22 ≠ v18 인 기록, gap = (v22_0 − v18_0)/BETA_0 로 복원(자르기 경계 ±2 그대로).  서술: 전체 행, 1차 적합도, 계수.
"""
import os, sys, importlib.util
import env  # noqa: F401
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
HERE = os.path.dirname(os.path.abspath(__file__)); sys.argv = ["x"]
s_ = importlib.util.spec_from_file_location("ta7", os.path.join(HERE, "ta7_anchor_sweep_v1.py")); A7 = importlib.util.module_from_spec(s_); s_.loader.exec_module(A7)
BETA = np.load(os.path.join(env.LOCAL, "linker_v7", "ta8_beta.npy"))
rm = lambda e: float(np.sqrt(np.mean(np.square(e)))); rng = np.random.default_rng(20261117)
L7 = pd.read_csv(os.path.join(env.LOCAL, "linker_v7", "cv_all400_choice_v7m.csv"), encoding="utf-8-sig")
L7f = L7[L7.prev_is_labelled.astype(bool) & L7.anchor_st23.notna() & (L7.steps_to_anchor == 1) & (L7.p >= .8) & (L7.dT_air.abs() <= 1.5)].set_index("rec")
BLK = np.array([0] * 4 + [1] * 4 + [2] * 8 + [3] * 8)
def rows_for(key, h, q, inT, y, gaps):
    """gaps: dict rec -> gap.  returns DataFrame of anchored rows with features."""
    q0 = pd.Series(q[h == 0], index=key[h == 0]); i0 = pd.Series(inT[h == 0], index=key[h == 0]); out = []
    m = np.isin(key, list(gaps)); ix = np.where(m)[0]
    D = pd.DataFrame(dict(i=ix, rec=key[ix], h=h[ix], q=q[ix], y=y[ix], inT=inT[ix]))
    D["gap"] = D.rec.map(gaps); D["drift_in"] = D.inT - D.rec.map(i0); D["drift_q"] = D.q - D.rec.map(q0); D["base"] = D.gap * BETA[D.h.values]
    return D
def Xlin(D):
    g = D.gap.values; cols = []
    for b in range(4):
        s = (BLK[D.h.values] == b).astype(float)
        cols += [s * g, s * g * D.drift_in.values, s * g * D.drift_q.values, s * g * np.abs(g)]
    return np.nan_to_num(np.column_stack(cols))
FG = ["gap", "h", "drift_in", "drift_q", "inT"]
R = A7.A2.A1.frames(); M, cells = R["DIAG10"]; key = (M.farm + "_" + M.day.map("{:03d}".format)).values; h = M.hour.values; yv = M.sub_temp.values; inT = M.in_temp.values
p1r = M.day.values < 179; p2r = ~p1r
Dcells = []
for q in cells:
    q0 = pd.Series(q[h == 0], index=key[h == 0]); gaps = {r: float(np.clip(L7f.at[r, "anchor_st23"] - .36 - q0[r], -2, 2)) for r in L7f.index.intersection(q0.index)}
    Dcells.append(rows_for(key, h, q, inT, yv, gaps))
TR = pd.concat([D[D.rec.map(lambda r: int(r[4:7]) < 179)] for D in Dcells]); TR["res"] = TR.y - TR.q
Xl = Xlin(TR); lam = 1.0; wl = np.linalg.solve(Xl.T @ Xl + lam * np.eye(Xl.shape[1]), Xl.T @ TR.res.values)
gbm = HistGradientBoostingRegressor(max_depth=3, max_iter=200, learning_rate=.05, l2_regularization=1.0, random_state=0).fit(np.nan_to_num(TR[FG].values), TR.res.values)
print(f"train rows (pass-1 anchored, 4 cells) {len(TR)} records {TR.rec.nunique()} | pass-1 fit RMSE: none {rm(TR.res):.3f} base-beta {rm(TR.res - TR.base):.3f} LIN {rm(TR.res - Xl @ wl):.3f} GBM {rm(TR.res - gbm.predict(np.nan_to_num(TR[FG].values))):.3f}")
print("LIN coefs per block [g, g*drift_in, g*drift_q, g*|g|]:", np.round(wl.reshape(4, 4), 3).tolist(), flush=True)
pred = {"LIN": lambda D: Xlin(D) @ wl, "GBM": lambda D: gbm.predict(np.nan_to_num(D[FG].values))}
res, dsets = [], {}
def score(tag, yy, p2, recs, base, news, cellname):
    for a, nw in news.items():
        res.append(dict(val=tag, cell=cellname, arm=a, p2_pct=100 * (rm((nw - yy)[p2]) / rm((base - yy)[p2]) - 1), all_pct=100 * (rm(nw - yy) / rm(base - yy) - 1)))
        dsets.setdefault((tag, a), []).append(pd.Series(((nw - yy) ** 2 - (base - yy) ** 2)[p2]).groupby(recs[p2]).sum())
for ci, (q, D) in enumerate(zip(cells, Dcells)):
    base = q.copy(); base[D.i.values] = q[D.i.values] + D.base.values
    news = {}
    for a, f in pred.items():
        nw = q.copy(); nw[D.i.values] = q[D.i.values] + f(D); news[a] = nw
    score("DIAG10", yv, p2r, key, base, news, ci)
DD = pd.read_pickle(os.path.join(HERE, "ev_validator", "temp", "local_ckpt", "ev_all.pkl")); DD["rec"] = DD.row_id.str[:7]; DD = DD.reset_index(drop=True)
ke = DD.rec.values; he = DD.hour.values; ye = DD.sub_temp.values; ie = DD.in_temp.values; p2e = DD.day.values >= 179
for c in ["7A", "7B", "101A", "101B"]:
    v18 = DD[f"v18_a_{c}"].values; v22 = DD[f"v22_a_{c}"].values
    d0 = pd.Series((v22 - v18)[he == 0], index=ke[he == 0]); gaps = {r: float(np.clip(v / BETA[0], -2, 2)) for r, v in d0.items() if abs(v) > 1e-9}
    D = rows_for(ke, he, v18, ie, ye, gaps); news = {}
    for a, f in pred.items():
        nw = v22.copy(); nw[D.i.values] = v18[D.i.values] + f(D); news[a] = nw
    score("EV", ye, p2e, ke, v22, news, c)
S = pd.DataFrame(res); pd.set_option("display.width", 200)
print(S.pivot_table(index="arm", columns=["val", "cell"], values="p2_pct").round(2).to_string())
print(S.pivot_table(index="arm", columns="val", values="all_pct", aggfunc="mean").round(2).to_string())
boot = lambda d: float((d[rng.integers(0, len(d), (20000, len(d)))].sum(1) >= 0).mean())
for a in pred:
    c1 = bool((S[S.arm == a].p2_pct < 0).all()); Ps, harm = [], []
    for v in ("DIAG10", "EV"):
        Dm = pd.concat(dsets[(v, a)], axis=1); Ps.append(boot(Dm.mean(axis=1).values)); harm.append(int((Dm > 1e-6).any(axis=1).sum()))
    ok = c1 and (all(p < .0125 for p in Ps) or all(x == 0 for x in harm))
    g = pd.concat(dsets[("DIAG10", a)], axis=1).mean(axis=1)
    print(f"\n{a}: all 8 cells better {c1} | P DIAG10 {Ps[0]:.4f} EV {Ps[1]:.4f} | harmed DIAG10 {harm[0]} EV {harm[1]} -> {'ADOPT' if ok else 'FAIL'}")
    print("   DIAG10 top gains:", (-g).sort_values(ascending=False).head(5).round(2).to_dict(), "| worst:", (-g).sort_values().head(4).round(2).to_dict())
