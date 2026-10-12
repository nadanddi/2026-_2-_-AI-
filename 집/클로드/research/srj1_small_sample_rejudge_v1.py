# -*- coding: utf-8 -*-
"""SRJ1 (2026-10-13 집 클로드).  사용자 결정: '모든 칸이 같은 방향이고 손해 기록이 없으면 작은 표본이라도 인정하자 일단'.  사전 고정(이 파일 커밋 후 실행).  EC 만.
새 '작은 표본 경로' 판정 — 이미 옛 규칙(P<.0125)으로 FAIL 난 앵커 부품들을 다시 판정(사후 재판정임을 밝혀 둠; 부품·기하·짝은 앞 판정과 동일, 새로 맞춘 것 없음).
안(앞 판정과 같은 정의):
  V15P8 = v7 → v15(PI .15) 1순위 p0 ≥ .8 전 시각 앵커 → sg            (6.770 계열)
  BIA   = v7 → 앞뒤 합의(v15 p0 ≥ .5 의 A, NXL1 다음날 p0 ≥ .8 의 B_h, ±.15 → (A+B_h)/2) → sg   (6.775)
  NXL   = v7 → NXL1 다음날 p0 ≥ .8 → β .9·B_h → sg                       (6.773)
  CAS / CAS+ = CAS1 그대로                                               (6.776)
  SWV   = v7(w=1) → 나머지 부드러운 앵커 w = sigmoid(a + b·logit p0), a·b 는 ECW1 과 같이 EV 에서 맞춤      (6.769; EV 칸은 맞춘 곳이라 참고로만 보고, 판정은 rv1 9칸)
기준(각 안 × 기준선 NONE·V7 각각):  (1) 판정 칸 모두 2차 행 RMSE 개선 (rv1 DIAG10·P2LOO·EL1 × 시드 47/1414/6464 + EV × 시드 7/101/2024 = 12칸; SWV 는 rv1 9칸);
  (2) 판정 칸 어느 하나에서도 2차 기록 ΔSSE > 1e-6 인 기록 없음(손해 기록 0);  (3) 바뀐 2차 기록 ≥ 1.   모두 → PASS(작은 표본 경로).  P 값은 참고로만 출력.
"""
import os, sys
import env  # noqa: F401
import numpy as np, pandas as pd
from scipy.optimize import minimize
HERE = os.path.dirname(os.path.abspath(__file__)); __file__ = os.path.join(HERE, "cas1_cascade_v1.py")
src = open(os.path.join(HERE, "cas1_cascade_v1.py"), encoding="utf-8").read(); exec(src[:src.index("res = []; dsse = {}")])
FWD = {"RV": pd.read_csv(os.path.join(env.LOCAL, "linker_v10", "v15g1_mirror.csv")).set_index("rec"), "EV": pd.read_csv(os.path.join(env.LOCAL, "linker_v10", "v15pi15_ev.csv")).set_index("rec")}
rng = np.random.default_rng(20261113)
wfun = lambda p, a, b: 1 / (1 + np.exp(-(a + b * np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4))))))
# ---- ECW1 fit (same as ecw1) ----
FO = pd.read_csv(os.path.join(HERE, "ev_validator", "ev_folds_v1.csv")); hid = {k: set(g.rec[g.role.isin(["val", "del"])]) for k, g in FO.groupby("fold")}
R = pd.read_csv(os.path.join(HERE, "ev_validator", "ec", "out_v1", "v2_ev_ec_rows.csv")); R = R[(R.arm == "BASE") & (R.seed.astype(str) != "avg")]
G = R.groupby(["rec", "hour", "fold", "day"], as_index=False).agg(y=("y", "first"), sg=("SG2", "mean")).sort_values(["rec", "hour"]).reset_index(drop=True)
G["cm"] = G.groupby("rec").sg.transform(lambda s: s.expanding().mean())
PE = pd.read_csv(os.path.join(env.LOCAL, "linker_v10", "v15pi15_ev.csv")).set_index("rec")
okm = np.array([r in PE.index and bool(PE.lab0.get(r)) and PE.prev0.get(r) not in hid[f] for r, f in zip(G.rec, G.fold)])
F = G[okm].copy(); F["A"] = [EC[idx[PE.prev0[r]], 23] for r in F.rec]; F["p0"] = [PE.p0[r] for r in F.rec]
fit = minimize(lambda th: float(np.sum((np.clip(F.sg.values + wfun(F.p0.values, *th) * (F.A.values - F.cm.values), lo_, hi_) - F.y.values) ** 2)), [0.0, 1.0], method="Nelder-Mead"); wa, wb = fit.x
print(f"SWV weights a {wa:.3f} b {wb:.3f}", flush=True)
ARMS = ["V15P8", "BIA", "NXL", "CAS", "CAS+", "SWV"]
res = []; rows = []
def run(Gv, y, sg, p2, hours, cm, recrows, v7val, labf, wf, tag, sd):
    base_v7 = sg.copy(); P = {a: sg.copy() for a in ARMS}
    FP = FWD["EV" if tag == "EV" else "RV"]
    for r, ix in recrows.items():
        if r in v7val:
            base_v7[ix] = np.clip(sg[ix] + (v7val[r] - cm[ix]), lo_, hi_)
            for a in ARMS: P[a][ix] = base_v7[ix]
            continue
        A_ = None; p0 = 0.0
        if r in FP.index:
            fr = FP.loc[r]; pv = fr.get("prev0")
            if isinstance(pv, str) and bool(fr["lab0"]) and pv not in HIDSET(r) and np.isfinite(EC[idx[pv], 23]): A_ = EC[idx[pv], 23]; p0 = fr["p0"]
        if A_ is not None:
            P["SWV"][ix] = np.clip(sg[ix] + wfun(p0, wa, wb) * (A_ - cm[ix]), lo_, hi_)
        Bh, pk = next_anchor(r, labf(r), wf(r)); Bv = Bh[hours[ix]]; fin = np.isfinite(Bv)
        agree = fin & (A_ is not None and p0 >= .5) & (np.abs(Bv - (A_ if A_ is not None else 0)) <= .15)
        AB = np.clip(sg[ix] + (((A_ if A_ is not None else 0) + np.nan_to_num(Bv)) / 2 - cm[ix]), lo_, hi_)
        NX = np.clip(sg[ix] + .9 * (np.nan_to_num(Bv) - cm[ix]), lo_, hi_)
        P["BIA"][ix[agree]] = AB[agree]; P["NXL"][ix[fin]] = NX[fin]
        if A_ is not None and p0 >= .8:
            fa = np.clip(sg[ix] + (A_ - cm[ix]), lo_, hi_)
            for a in ("V15P8", "CAS", "CAS+"): P[a][ix] = fa
            continue
        for a in ("CAS", "CAS+"): P[a][ix[agree]] = AB[agree]
        rest = fin & ~agree; P["CAS+"][ix[rest]] = NX[rest]
    B = {"NONE": sg, "V7": base_v7}
    for a in ARMS:
        for bn in B:
            b_, n_ = B[bn], P[a]
            res.append(dict(val=tag, seed=str(sd), arm=a, base=bn, all_pct=100 * (rm(n_ - y) / rm(b_ - y) - 1), p2_pct=100 * (rm((n_ - y)[p2]) / rm((b_ - y)[p2]) - 1)))
            ds = pd.Series(((n_ - y) ** 2 - (b_ - y) ** 2)[p2]).groupby(Gv.rec.values[p2]).sum()
            for r, d in ds.items(): rows.append(dict(val=tag, seed=str(sd), arm=a, base=bn, rec=r, dsse=d))
O = pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "ec_rv1_preds_v1.csv")); O = O[O.tag == "CUR"].copy(); O["rec"] = O.row_id.str[:7]
for (v, sd), Gv in O.groupby(["validator", "seed"]):
    Gv = Gv.sort_values(["rec", "hour"]).reset_index(drop=True); y = Gv.sub_ec.values; sg = Gv.sg.values; p2 = Gv.day.values >= 179
    cm = Gv.groupby("rec").sg.transform(lambda s: s.expanding().mean()).values
    hidden = Gv.groupby("fold").rec.apply(set).to_dict(); fold_of = Gv.groupby("rec").fold.first().to_dict()
    HM = {k: np.array([rid(i) in vv for i in range(N)]) for k, vv in hidden.items()}
    labf = lambda r: TRAIN & ~HM[fold_of[r]]; wf = lambda r: weights((v, fold_of[r]), HM[fold_of[r]]); HIDSET = lambda r: hidden[fold_of[r]]
    recrows = {r: g.index.values for r, g in Gv.groupby("rec")}
    v7val = {r: EC[idx[p], 23] for r, p in V7.items() if r in recrows and p not in hidden[fold_of[r]]}
    run(Gv, y, sg, p2, Gv.hour.values, cm, recrows, v7val, labf, wf, v, sd); print("done", v, sd, flush=True)
HME = {k: np.array([rid(i) in vv for i in range(N)]) for k, vv in hid.items()}
for sd, Gv in R.groupby(R.seed.astype(str)):
    Gv = Gv.sort_values(["rec", "hour"]).reset_index(drop=True); y = Gv.y.values; sg = Gv.SG2.values; ea = Gv.EA2.values; p2 = Gv.day.values >= 179
    cm = Gv.groupby("rec").SG2.transform(lambda s: s.expanding().mean()).values; recrows = {r: g.index.values for r, g in Gv.groupby("rec")}
    fo = {r: int(Gv.fold.values[ix[0]]) for r, ix in recrows.items()}; labf = lambda r: TRAIN & ~HME[fo[r]]; wf = lambda r: weights(("EV", fo[r]), HME[fo[r]]); HIDSET = lambda r: hid[fo[r]]
    v7val = {r: float(ea[recrows[r][0]]) for r in set(Gv.rec[Gv.anchored.astype(bool)])}
    run(Gv, y, sg, p2, Gv.hour.values, cm, recrows, v7val, labf, wf, "EV", sd); print("done EV", sd, flush=True)
S = pd.DataFrame(res); D = pd.DataFrame(rows); pd.set_option("display.width", 260)
os.makedirs(os.path.join(env.LOCAL, "srj1"), exist_ok=True); D.to_csv(os.path.join(env.LOCAL, "srj1", "srj1_record_dsse.csv"), index=False)
print("pass-2 RMSE % change:"); print(S.pivot_table(index=["arm", "base"], columns=["val", "seed"], values="p2_pct").round(2).to_string())
print("all-rows % change (seed mean):"); print(S.pivot_table(index=["arm", "base"], columns="val", values="all_pct", aggfunc="mean").round(2).to_string())
boot = lambda d: float((d[rng.integers(0, len(d), (20000, len(d)))].sum(1) >= 0).mean())
for a in ARMS:
    verdict = True
    for bn in ("NONE", "V7"):
        cells = S[(S.arm == a) & (S.base == bn)]; dd = D[(D.arm == a) & (D.base == bn)]
        if a == "SWV": cells = cells[cells.val != "EV"]; dd = dd[dd.val != "EV"]
        c1 = bool((cells.p2_pct < 0).all()); nbetter = int((cells.p2_pct < 0).sum())
        harm = dd[dd.dsse > 1e-6].groupby("rec").agg(cells=("dsse", "size"), max_dsse=("dsse", "max")).sort_values("max_dsse", ascending=False)
        ch = dd[dd.dsse.abs() > 1e-6].rec.unique(); ok = c1 and len(harm) == 0 and len(ch) >= 1; verdict &= ok
        Dd = dd[dd.val == "DIAG10"].groupby("rec").dsse.mean(); Pd = boot(Dd.values)
        print(f"\n{a} vs {bn}: cells better {nbetter}/{len(cells)} | harmed records {len(harm)} | changed p2 records {len(ch)} | (ref) DIAG10 seed-mean P {Pd:.4f} -> {'pass' if ok else 'fail'}")
        if len(harm): print("   harmed:", {r: (int(c), round(m, 4)) for r, (c, m) in harm.head(8).iterrows()})
        gain = dd.groupby("rec").dsse.mean(); print("   changed (mean dSSE over cells):", gain[gain.abs() > 1e-6].sort_values().round(4).to_dict())
    print(f"SRJ1 {a}:", "PASS" if verdict else "FAIL")
