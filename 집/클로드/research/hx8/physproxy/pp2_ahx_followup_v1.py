# -*- coding: utf-8 -*-
"""PP2 (2026-10-13 집 클로드, hx8/physproxy).  PP1 에서 경계선이던 AHx(실내−외기 절대습도 차) 후속 점검. 서술.
(1) 고예측 표본에서 AHx 가 '진짜 고EC 크기'를 설명하는지, 아니면 '가짜 고EC(예측 높고 실제 보통)' 를 가르는지 분해.
(2) 문턱(예측 ≥ .8/.9/1.0/1.2) 민감도, 검증기 EL1·P2LOO(2차 기록) 방향.
(3) 인과판: AHx(0..h), h = 0, 5, 11, 17, 23 — 그 시각까지 입력만.
(4) 6.697 단순 요약(in_hum, out_hum, in_temp, out_temp 평균)과의 상관: AHx 가 단순 요약의 재포장인지.
(5) 다변수 릿지(파생 21종, λ 3단계 고정) LOEO.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pp1_physproxy_v1 import *   # noqa

F23 = pd.read_csv(os.path.join(here, "pp1_train_feats_full_v1.csv"), index_col=0)
Ftab = {h: (F23 if h == 23 else table(A_tr, h)) for h in [0, 5, 11, 17, 23]}

def sample(val, thr, sd="avg"):
    Ov = O[O.validator == val]; Os = Ov if sd == "avg" else Ov[Ov.seed == sd]
    q = Os.groupby("rec").sg.mean(); recs = [r for r in q.index if r in yday.index]
    q = q[recs]; y = yday[recs]; k = q >= thr
    recs = list(q.index[k.values]); return recs, y[recs].values, q[recs].values

print("(1) S_pred(.9) DIAG10 avg: records sorted by AHx")
recs, y, q = sample("DIAG10", .9)
D = pd.DataFrame(dict(y=y, q=q, r=y - q, AHx=F23.loc[recs, "AHx"].values, AHx_night=F23.loc[recs, "AHx_night"].values,
                      fan=[np.nanmean(A_tr[r]["act_circfan"]) for r in recs], vent=[np.nanmean(A_tr[r]["act_vent"]) for r in recs]), index=recs)
ep = episodes(recs); D["ep"] = [ep[r] for r in recs]
print(D.sort_values("AHx").round(2).to_string())
D["true_high"] = D.y >= 1.0
print("\n   AHx by true_high:", D.groupby("true_high").AHx.describe().round(2).to_string())
print("   rho(AHx, r) among true-high only %.2f (n %d); among false-high only %.2f (n %d)" % (
    spear(D.AHx[D.true_high].values, D.r[D.true_high].values), D.true_high.sum(),
    spear(D.AHx[~D.true_high].values, D.r[~D.true_high].values), (~D.true_high).sum()))

print("\n(2) threshold / validator sensitivity of AHx (full day): LOEO RMSE q, q+icpt, q+icpt+b*AHx")
for val in ["DIAG10", "EL1", "P2LOO"]:
    for thr in [.8, .9, 1.0, 1.2]:
        for sd in ["avg", 47, 1414, 6464]:
            recs, y, q = sample(val, thr, sd)
            if len(recs) < 6: print(f"   {val} thr {thr} seed {sd}: n={len(recs)} too small"); break
            ep = episodes(recs); g = np.array([ep[r] for r in recs])
            if len(np.unique(g)) < 3: print(f"   {val} thr {thr}: episodes {len(np.unique(g))} too few"); break
            f = F23.loc[recs, "AHx"].values
            p0, p1 = loeo(y, q, f, g)
            print(f"   {val} thr {thr} seed {sd}: n {len(y)} ep {len(np.unique(g))} | q {rmse(q,y):.4f}  icpt {rmse(p0,y):.4f}  AHx {rmse(p1,y):.4f}  gain {rmse(p0,y)-rmse(p1,y):+.4f}  better {np.mean(np.abs(p1-y)<np.abs(p0-y)):.2f}  rho_r {spear(f,y-q):+.2f}")

print("\n(3) causal AHx(0..h): DIAG10 S_pred(.9), seed avg")
recs, y, q = sample("DIAG10", .9); ep = episodes(recs); g = np.array([ep[r] for r in recs])
for h in [0, 5, 11, 17, 23]:
    for c in ["AHx", "AHx_night", "AH"]:
        f = Ftab[h].loc[recs, c].values.astype(float)
        p0, p1 = loeo(y, q, f, g)
        print(f"   h {h:2d} {c:10s}: gain vs icpt {rmse(p0,y)-rmse(p1,y):+.4f} (vs q {rmse(q,y)-rmse(p1,y):+.4f})  rho_r {spear(f,y-q):+.2f}  ok {np.isfinite(f).sum()}")

print("\n(4) AHx vs simple summaries (all 400 training records, full day)")
S = pd.DataFrame({r: {v: np.nanmean(A_tr[r][v]) for v in ["in_hum", "out_hum", "in_temp", "out_temp", "act_vent", "act_circfan"]} for r in A_tr}).T
S["AHx"] = F23.AHx; print(S.corr(method="spearman").AHx.round(2).to_string())
Ss = S.loc[recs]; print("   within S_pred:"); print(Ss.corr(method="spearman").AHx.round(2).to_string())
# partial: residualise AHx on simple summaries (within S_pred, LOEO not needed for a descriptive check)
from numpy.linalg import lstsq
Z = np.c_[np.ones(len(recs)), Ss[["in_hum", "out_hum", "in_temp", "out_temp"]].values]
ahx_res = Ss.AHx.values - Z @ lstsq(Z, Ss.AHx.values, rcond=None)[0]
print("   rho(AHx residualised on 4 simple means, model residual) = %.2f" % spear(ahx_res, y - q))
for c in ["in_hum", "out_hum", "in_temp", "out_temp"]:
    print(f"   rho({c}, residual) = {spear(Ss[c].values, y - q):+.2f}")

print("\n(5) ridge on all 21 derived features (standardised in fold, NaN->fold median), LOEO, DIAG10 S_pred(.9)")
def ridge_loeo(y, q, Fm, g, lam):
    r = y - q; p = np.zeros_like(y)
    for gg in np.unique(g):
        tr = g != gg; te = ~tr
        med = np.nanmedian(Fm[tr], 0); Fi = np.where(np.isfinite(Fm), Fm, med)
        mu, sdv = Fi[tr].mean(0), Fi[tr].std(0) + 1e-9; Z = (Fi - mu) / sdv
        Zt = Z[tr]; rt = r[tr] - r[tr].mean()
        w = np.linalg.solve(Zt.T @ Zt + lam * np.eye(Z.shape[1]), Zt.T @ rt)
        p[te] = q[te] + r[tr].mean() + Z[te] @ w
    return p
for sel_thr in [.9, 1.0]:
    for sd in ["avg", 47, 1414, 6464]:
        recs, y, q = sample("DIAG10", sel_thr, sd); ep = episodes(recs); g = np.array([ep[r] for r in recs])
        Fm = F23.loc[recs, FEATS].values.astype(float)
        p0, _ = loeo(y, q, Fm[:, 0], g)
        s = f"   thr {sel_thr} seed {sd}: n {len(y)} q {rmse(q,y):.4f} icpt {rmse(p0,y):.4f}"
        for lam in [10, 30, 100]:
            p = ridge_loeo(y, q, Fm, g, lam); s += f" | ridge{lam} {rmse(p,y):.4f}"
        print(s)
