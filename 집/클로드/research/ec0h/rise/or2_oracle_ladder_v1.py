# -*- coding: utf-8 -*-
"""OR2 (2026-10-12 집 클로드; 사용자: "'하루 수준' → '0시 수준'처럼 더 세밀하게, 리더보드 최상위로 갈 핵심 병목을 조사").
오라클 사다리(정답에서 정보를 덜어낸 가짜 정보를 기존 예측에 주고 점수 변화 측정) — 서술, 채택 판정 아님.
기준 예측: EC = EV 검증기 EA2(3시드 평균, 평가 배치), 온도 = EV v22(a/b × 7/101 × A/B 평균).  보조: EC DIAG10 CUR sg.
보정 방식: 0시 오차 e0 를 알면 pred_h = q_h + c_h * e0_hat (c_h = 1차에서 잰 e_h 의 e0 회귀 기울기; 오라클이라 같은 자료 사용 허용).
사다리:
 L1 0시 정확.   L2 0시를 잡음 σ 로 앎(베이즈 가중 w = v/(v+σ²)).   L3 0시를 몇 칸(분위수 2/3/5/10칸)으로만 앎 → q0 를 그 칸 안으로 자름.
 L4 일부 기록만 0시 정확(고EC / 오차 상위 k% / F13·F47 / 앵커 있음·없음).   L5 0시 대신 다른 한 시각(6·12·23시)·하루 평균.
 L6 0시 다음 병목: 0시+23시 정확(직선 보간 변화) / 0시+12시+23시 / 0시+하루 평균.
 L7 지금 가진 정보원의 실제 0시 오차 σ(모델 q0, EA2 앵커 기록, 소프트 앵커 대상) → L2 곡선 위 위치.
리더보드 환산 = 현재 공개 점수 × (오라클 RMSE / 기준 RMSE), 2차(평가 배치) 기준 — 가정(2차 상대 변화가 평가로 옮겨감)에 기댄 대략값.
"""
import sys, os
sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env
import numpy as np, pandas as pd
rng = np.random.default_rng(20261216)
H_ = os.path.join(env.ROOT, u"집", u"클로드", "research")
LB = {"EC": .1271, "TEMP": .5012}
rm = lambda e: float(np.sqrt(np.nanmean(np.square(e))))
# ---- load ----
R = pd.read_csv(os.path.join(H_, "ev_validator", "ec", "out_v1", "v2_ev_ec_rows.csv")); R = R[(R.arm == "BASE") & (R.seed.astype(str) != "avg")]
QE = R.groupby(["rec", "hour"]).EA2.mean().unstack(); YE = R.groupby(["rec", "hour"]).y.first().unstack(); ANC = R.groupby("rec").anchored.first()
T = pd.read_pickle(os.path.join(H_, "ev_validator", "temp", "local_ckpt", "ev_all.pkl")); T["rec"] = T.row_id.str[:7]
vc = [c for c in T.columns if c.startswith("v22_")]; T["q"] = T[vc].mean(1)
QT = T.pivot(index="rec", columns="hour", values="q"); YT = T.pivot(index="rec", columns="hour", values="sub_temp")
O = pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "ec_rv1_preds_v1.csv")); O = O[(O.tag == "CUR") & (O.validator == "DIAG10")]
O["rec"] = O.row_id.str[:7]; QD = O.groupby(["rec", "hour"]).sg.mean().unstack(); YD = O.groupby(["rec", "hour"]).sub_ec.first().unstack()
p2 = lambda idx: np.array([int(r[4:7]) >= 179 for r in idx])
def run(tag, Q, Y, anc=None):
    E = (Y - Q); P2 = p2(Q.index); e1 = E[~P2].values
    c = np.array([np.nansum(e1[:, 0] * e1[:, h]) / np.nansum(e1[:, 0] ** 2) for h in range(24)])
    print(f"\n#################### {tag} | records pass-1 {int((~P2).sum())}, pass-2 {int(P2.sum())} | c_h 0,6,12,18,23 = {np.round(c[[0,6,12,18,23]],2).tolist()}")
    def sc(newE, part):
        s = P2 if part == 2 else ~P2; return rm(newE[s])
    base = {k: sc(E.values, k) for k in (1, 2)}
    lbk = "EC" if "EC" in tag else "TEMP"
    def line(nm, newE):
        r1, r2 = sc(newE, 1), sc(newE, 2)
        print(f"  {nm:46s} pass-1 {r1:.4f} ({100*(r1/base[1]-1):+5.1f}%) | pass-2 {r2:.4f} ({100*(r2/base[2]-1):+5.1f}%) | LB~{LB[lbk]*r2/base[2]:.3f}")
    line("BASE", E.values)
    e0 = E[0].values[:, None]
    line("L1 0h exact, full shift", E.values - e0)
    line("L1 0h exact, c_h carry", E.values - c[None, :] * e0)
    v = np.nanvar(E[0].values[~P2])
    print("  -- L2 0h known with noise sigma (Bayes weight) --")
    for s in (.01, .02, .03, .05, .08, .1, .15, .2, .3, .5, 1.0):
        w = v / (v + s * s); noise = rng.normal(0, s, len(e0))[:, None]
        line(f"L2 sigma {s:.2f} (w {w:.2f})", E.values - c[None, :] * w * (e0 + noise))
    print("  -- L3 0h known only as a bin (quantile bins of true 0h, pass-1 edges) --")
    y0 = Y[0].values; q0 = Q[0].values
    for k in (2, 3, 5, 10, 20):
        edges = np.quantile(y0[~P2], np.linspace(0, 1, k + 1)); edges[0], edges[-1] = -np.inf, np.inf
        b = np.clip(np.searchsorted(edges, y0, side="right") - 1, 0, k - 1); lo, hi = edges[b], edges[b + 1]
        q0n = np.clip(q0, lo, hi); line(f"L3 {k} bins", E.values - c[None, :] * (q0n - q0)[:, None])
    print("  -- L4 exact 0h for a subset only --")
    ae = np.abs(E[0].values)
    subsets = {"high level (true 0h >= 1.0 EC / sub_temp<=12 TEMP)": (Y[0].values >= 1.0) if lbk == "EC" else (Y[0].values <= 12),
               "F47 only": np.array([r[:3] == "F47" for r in Q.index]), "F13 only": np.array([r[:3] == "F13" for r in Q.index])}
    for q in (.1, .2, .3, .5):
        thr = np.r_[np.quantile(ae[~P2], 1 - q)]; subsets[f"top {int(q*100)}% |0h error| (pass-1 threshold)"] = ae >= thr[0]
    if anc is not None:
        a = anc.reindex(Q.index).fillna(False).values.astype(bool); subsets["records WITH EA2 anchor"] = a; subsets["records WITHOUT anchor"] = ~a
    for nm, m in subsets.items():
        newE = E.values.copy(); newE[m] = newE[m] - c[None, :] * e0[m]
        line(f"L4 {nm} (share p2 {m[P2].mean():.2f})", newE)
    print("  -- L5 one other hour exact instead of 0h (carry from that hour, symmetric regression) --")
    for hh in (3, 6, 12, 18, 23):
        ch = np.array([np.nansum(e1[:, hh] * e1[:, h]) / np.nansum(e1[:, hh] ** 2) for h in range(24)])
        line(f"L5 {hh}h exact", E.values - ch[None, :] * E[hh].values[:, None])
    dm = E.mean(1).values[:, None]; line("L5 daily mean exact (shift)", E.values - dm)
    print("  -- L6 after 0h: what is the next bottleneck --")
    lin = np.array([E[0].values + (E[23].values - E[0].values) * h / 23 for h in range(24)]).T
    line("L6 0h+23h exact (linear between)", E.values - lin)
    pts = [0, 12, 23]; L6b = np.array([np.interp(range(24), pts, E.loc[r, pts].values) for r in E.index]); line("L6 0h+12h+23h exact", E.values - L6b)
    pts = [0, 6, 12, 18, 23]; L6c = np.array([np.interp(range(24), pts, E.loc[r, pts].values) for r in E.index]); line("L6 every 6h exact", E.values - L6c)
    # residual decomposition after L1
    R1 = E.values - e0
    for part in (1, 2):
        s = (P2 if part == 2 else ~P2)
        tot = np.nansum(E.values[s] ** 2); dl = np.nansum((np.nanmean(E.values[s], 1) ** 2)) * 24
        print(f"  pass-{part}: day-level share of SSE {dl/tot:.2f} | SSE left after 0h-exact shift {np.nansum(R1[s]**2)/tot:.2f}")
    print("  -- L7 where our current 0h estimates sit (RMSE of 0h error) --")
    for part in (1, 2):
        s = (P2 if part == 2 else ~P2)
        print(f"   pass-{part}: model 0h RMSE {rm(E[0].values[s]):.3f} | 0h error sd {np.nanstd(E[0].values[s]):.3f}", end="")
        if anc is not None:
            a = anc.reindex(Q.index).fillna(False).values.astype(bool)
            print(f" | anchored records 0h RMSE {rm(E[0].values[s & a]):.3f} (n {int((s&a).sum())}) | unanchored {rm(E[0].values[s & ~a]):.3f} (n {int((s&~a).sum())})", end="")
        print()
    return E, c
run("EC  EV (EA2)", QE, YE.reindex(QE.index), ANC)
run("EC  DIAG10 (CUR sg)", QD, YD.reindex(QD.index))
run("TEMP EV (v22)", QT, YT.reindex(QT.index))
