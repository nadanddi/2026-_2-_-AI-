# -*- coding: utf-8 -*-
"""ECW1 (2026-10-13 집 클로드; 사용자: '최고의 확신 크기를 찾는 모델 — 뭐라도 해봐').  사전 고정(이 파일 커밋 후 실행, G1 링커 출력 확인 전 작성).  EC 만.
'확신만큼만 따라가기': pred_h = clip(sg_h + w·(A − sg 0..h 누적평균)), A = 1순위 실제 후보(보이는 정답 기록) 23시 EC, w = sigmoid(a + b·logit(p0)),
  p0 = v15(없음 선택지, PI .15) 의 '없음 포함' 확률.  숫자 2개(a, b)만 맞춤.
맞추기(판정과 분리): EV 기하(v15pi15_ev × ev_validator EC 행 SG2, 시드 평균 예측·같은 y), 숨김 짝 제외, 전체 행 SSE 최소(Nelder-Mead, 시작 a=0,b=1).
판정(처음 쓰는 기하): G1 거울 링커 짝(v15g1_mirror: 구간 시작 +5, 다른 ID 이동 ±4) × rv1 DIAG10·P2LOO·EL1 x 시드 47/1414/6464(EV 는 판정에 안 씀).
안(2개 → 본페로니 α .0125): SW = 모든 기록 부드러운 앵커(v7 안 씀); SWV = v7 앵커 기록은 v7 그대로(w=1), 나머지 부드러운 앵커.
기준: NONE(앵커 없음 ≈ 12회차)·V7(v7 거울 짝, 13회차) 각각에 대해 모두:
  (1) 9칸 모두 2차 행 RMSE 개선; (2) DIAG10 시드별 2차 기록 부트스트랩 P(악화) < .0125; (3) DIAG10(시드 평균) 2차 이득 상위 3기록 제외 ΔSSE 합 < 0;
  (4) DIAG10(시드 평균) 2차에서 바뀐 기록(|ΔSSE| > 1e-6) ≥ 5.   모두 만족하면 ADOPT.
서술: 맞춘 a·b 와 p0 별 w 표, 전체 행, 바뀐 기록 수, 이득·손해 상위.
"""
import os, sys
import env  # noqa: F401
import numpy as np, pandas as pd
from scipy.optimize import minimize
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "linker_v4")); from feat4 import EC, FARM, NUM, N, TRAIN  # noqa: E402
rid = lambda i: f"{FARM[i]}_{NUM[i]:03d}"; idx = {rid(i): i for i in range(N)}
rm = lambda e: float(np.sqrt(np.mean(np.square(e)))); rng = np.random.default_rng(20261031)
lo_, hi_ = np.nanmin(EC[TRAIN]), np.nanmax(EC[TRAIN])
wfun = lambda p, a, b: 1 / (1 + np.exp(-(a + b * np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4))))))
# ---------- fit on EV ----------
FO = pd.read_csv(os.path.join(HERE, "ev_validator", "ev_folds_v1.csv")); hid = {k: set(g.rec[g.role.isin(["val", "del"])]) for k, g in FO.groupby("fold")}
R = pd.read_csv(os.path.join(HERE, "ev_validator", "ec", "out_v1", "v2_ev_ec_rows.csv")); R = R[(R.arm == "BASE") & (R.seed.astype(str) != "avg")]
G = R.groupby(["rec", "hour", "fold", "day"], as_index=False).agg(y=("y", "first"), sg=("SG2", "mean")).sort_values(["rec", "hour"]).reset_index(drop=True)
G["cm"] = G.groupby("rec").sg.transform(lambda s: s.expanding().mean())
PE = pd.read_csv(os.path.join(env.LOCAL, "linker_v10", "v15pi15_ev.csv")).set_index("rec")
ok = np.array([r in PE.index and bool(PE.lab0.get(r)) and PE.prev0.get(r) not in hid[f] for r, f in zip(G.rec, G.fold)])
F = G[ok].copy(); F["A"] = [EC[idx[PE.prev0[r]], 23] for r in F.rec]; F["p0"] = [PE.p0[r] for r in F.rec]
def loss(th):
    w = wfun(F.p0.values, *th); pr = np.clip(F.sg.values + w * (F.A.values - F.cm.values), lo_, hi_); return float(np.sum((pr - F.y.values) ** 2))
fit = minimize(loss, [0.0, 1.0], method="Nelder-Mead"); a_, b_ = fit.x
print(f"fit on EV ({F.rec.nunique()} records with a visible labelled top pick): a {a_:.3f} b {b_:.3f} | SSE {fit.fun:.3f} vs w=0 {loss([-50, 0]):.3f} vs w=1 {loss([50, 0]):.3f}")
print("w by p0:", {p: round(float(wfun(p, a_, b_)), 3) for p in (.1, .3, .5, .7, .8, .9, .95, .99)}, flush=True)
# ---------- judge on rv1 with G1 picks ----------
L7 = pd.read_csv(os.path.join(env.LOCAL, "linker_v7", "cv_all400_choice_v7m.csv"), encoding="utf-8-sig")
V7 = {r: p for r, p, q, l, d in zip(L7.rec, L7.prev, L7.p, L7.prev_is_labelled, L7.dT_air) if q >= .8 and l and abs(d) <= 1.5}
PG = pd.read_csv(os.path.join(env.LOCAL, "linker_v10", "v15g1_mirror.csv")).set_index("rec")
O = pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "ec_rv1_preds_v1.csv")); O = O[O.tag == "CUR"].copy(); O["rec"] = O.row_id.str[:7]
res = []; dsse = {}
for (v, sd), Gv in O.groupby(["validator", "seed"]):
    Gv = Gv.sort_values(["rec", "hour"]).reset_index(drop=True); y = Gv.sub_ec.values; sg = Gv.sg.values; p2 = Gv.day.values >= 179
    cm = Gv.groupby("rec").sg.transform(lambda s: s.expanding().mean()).values
    hidden = Gv.groupby("fold").rec.apply(set).to_dict(); fold_of = Gv.groupby("rec").fold.first().to_dict(); hs = lambda r: hidden.get(fold_of[r], set())
    rec = {r: g.index.values for r, g in Gv.groupby("rec")}
    v7 = np.zeros(len(Gv)); v7w = np.zeros(len(Gv)); sw = np.zeros(len(Gv)); sww = np.zeros(len(Gv))
    for r, ix in rec.items():
        if r in V7 and V7[r] not in hs(r): v7[ix] = EC[idx[V7[r]], 23]; v7w[ix] = 1.0
        if r in PG.index and bool(PG.lab0[r]) and PG.prev0[r] not in hs(r) and np.isfinite(EC[idx[PG.prev0[r]], 23]):
            sw[ix] = EC[idx[PG.prev0[r]], 23]; sww[ix] = wfun(PG.p0[r], a_, b_)
    mk = lambda Av, Wv: np.clip(sg + Wv * (Av - cm), lo_, hi_)
    preds = {"NONE": sg, "V7": mk(v7, v7w), "SW": mk(sw, sww), "SWV": mk(np.where(v7w > 0, v7, sw), np.where(v7w > 0, 1.0, sww))}
    for arm in ("SW", "SWV"):
        for bn in ("NONE", "V7"):
            b, n = preds[bn], preds[arm]
            res.append(dict(val=v, seed=sd, arm=arm, base=bn, all_pct=100 * (rm(n - y) / rm(b - y) - 1), p2_pct=100 * (rm((n - y)[p2]) / rm((b - y)[p2]) - 1)))
            if v == "DIAG10": dsse[(arm, bn, sd)] = pd.Series(((n - y) ** 2 - (b - y) ** 2)[p2]).groupby(Gv.rec.values[p2]).sum()
S = pd.DataFrame(res); pd.set_option("display.width", 240)
print("\npass-2 RMSE % change:"); print(S.pivot_table(index=["arm", "base"], columns=["val", "seed"], values="p2_pct").round(2).to_string())
print("\nall-rows % change (seed mean):"); print(S.pivot_table(index=["arm", "base"], columns="val", values="all_pct", aggfunc="mean").round(2).to_string())
boot = lambda d: float((d[rng.integers(0, len(d), (20000, len(d)))].sum(1) >= 0).mean())
for arm in ("SW", "SWV"):
    verdict = True
    for bn in ("NONE", "V7"):
        g = S[(S.arm == arm) & (S.base == bn)]; c1 = bool((g.p2_pct < 0).all())
        Pd = [boot(dsse[(arm, bn, sd)].values) for sd in (47, 1414, 6464)]
        Dd = pd.concat([dsse[(arm, bn, sd)] for sd in (47, 1414, 6464)], axis=1).mean(axis=1)
        t3 = Dd.sort_values().iloc[3:].sum(); nch = int((Dd.abs() > 1e-6).sum())
        okk = c1 and all(p < .0125 for p in Pd) and t3 < 0 and nch >= 5; verdict &= okk
        print(f"\n{arm} vs {bn}: all p2 better {c1} | DIAG10 P {np.round(Pd, 4).tolist()} | drop-top3 {t3:+.3f} | changed p2 recs {nch} -> {'pass' if okk else 'fail'}")
        print("   top gains:", (-Dd).sort_values(ascending=False).head(5).round(3).to_dict(), "| worst:", (-Dd).sort_values().head(3).round(3).to_dict())
    print(f"ECW1 {arm}:", "ADOPT" if verdict else "FAIL")
