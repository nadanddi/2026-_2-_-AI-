# -*- coding: utf-8 -*-
"""V15P5 (2026-10-13 집 클로드; 사용자: 'EC 0.1 아래로 — 과거 정보·규칙성 의존까지만').  사전 고정(이 파일 커밋 후 실행).  EC.
규칙: v7 앵커 기록은 v7 그대로 → 나머지는 v15(없음 선택지, PI .15) 1순위가 보이는 정답 기록 & p0 ≥ T 이면 전 시각 EA2 앵커(A = 짝 23시 EC) → 아니면 sg.
  (과거 정보만: 전날 짝의 정답.  다음날 정보 안 씀.)
안: T .5(V15P5), T .65(V15P65)  — 참고로 T .8(V15P8, 6.779 와 같음).  F47_233(평가, v15 p0 .68)이 동기임을 밝혀 둠(평가 배열을 보고 정한 문턱 후보 → 경고).
판정(기준선 V7 = 13회차 구성, NONE = 12회차 구성 각각): rv1 DIAG10·P2LOO·EL1 × 47/1414/6464 + EV × 7/101/2024 = 12칸 모두 2차 행 RMSE 개선 AND
  ((a) DIAG10·EV 2차 기록 부트스트랩 P(악화) < .0125 (안 2개 본페로니)  또는  (b) 어느 칸에서도 2차 기록 ΔSSE > 1e-6 없음).  → ADOPT.
서술: 바뀐 기록·손해 기록, 문턱 구간 [.5,.8) 짝 정확도(깨끗한 진실 대비), 평가 적용 시 바뀌는 기록.
"""
import os, sys
import env  # noqa: F401
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "linker_v4")); from feat4 import EC, FARM, NUM, N, TRAIN, S3  # noqa: E402
rid = lambda i: f"{FARM[i]}_{NUM[i]:03d}"; idx = {rid(i): i for i in range(N)}
rm = lambda e: float(np.sqrt(np.mean(np.square(e)))); rng = np.random.default_rng(20261114)
lo_, hi_ = np.nanmin(EC[TRAIN]), np.nanmax(EC[TRAIN])
TL = pd.read_pickle(os.path.join(S3, "truth.pkl")); CAL = pd.read_csv(os.path.join(HERE, "ec0h", "rise", "dm11_calendar_v1.csv")); D11 = dict(zip(CAL.rec, CAL.day))
CLEAN = {rid(int(b)): rid(int(a)) for a, b in zip(TL.a1, TL.b) if D11.get(rid(int(b)), np.nan) - D11.get(rid(int(a)), np.nan) == 1}
L7 = pd.read_csv(os.path.join(env.LOCAL, "linker_v7", "cv_all400_choice_v7m.csv"), encoding="utf-8-sig")
V7 = {r: p for r, p, q, l, d in zip(L7.rec, L7.prev, L7.p, L7.prev_is_labelled, L7.dT_air) if q >= .8 and l and abs(d) <= 1.5}
FWD = {"RV": pd.read_csv(os.path.join(env.LOCAL, "linker_v10", "v15g1_mirror.csv")).set_index("rec"), "EV": pd.read_csv(os.path.join(env.LOCAL, "linker_v10", "v15pi15_ev.csv")).set_index("rec")}
ARMS = {"V15P5": .5, "V15P65": .65, "V15P8": .8}
res, rows, accs = [], [], []
def run(Gv, y, sg, p2, cm, recrows, v7val, hsf, tag, sd):
    base_v7 = sg.copy(); P = {a: sg.copy() for a in ARMS}; FP = FWD["EV" if tag == "EV" else "RV"]
    for r, ix in recrows.items():
        if r in v7val:
            base_v7[ix] = np.clip(sg[ix] + (v7val[r] - cm[ix]), lo_, hi_)
            for a in ARMS: P[a][ix] = base_v7[ix]
            continue
        if r not in FP.index: continue
        fr = FP.loc[r]; pv = fr.get("prev0")
        if not (isinstance(pv, str) and bool(fr["lab0"]) and pv not in hsf(r) and np.isfinite(EC[idx[pv], 23])): continue
        if NUM[idx[r]] >= 179 and .5 <= fr["p0"] < .8: accs.append(dict(val=tag, seed=str(sd), rec=r, p0=fr["p0"], ok=(CLEAN.get(r) == pv) if r in CLEAN else None))
        for a, T in ARMS.items():
            if fr["p0"] >= T: P[a][ix] = np.clip(sg[ix] + (EC[idx[pv], 23] - cm[ix]), lo_, hi_)
    B = {"NONE": sg, "V7": base_v7}
    for a in ARMS:
        for bn, b_ in B.items():
            n_ = P[a]; res.append(dict(val=tag, seed=str(sd), arm=a, base=bn, all_pct=100 * (rm(n_ - y) / rm(b_ - y) - 1), p2_pct=100 * (rm((n_ - y)[p2]) / rm((b_ - y)[p2]) - 1)))
            ds = pd.Series(((n_ - y) ** 2 - (b_ - y) ** 2)[p2]).groupby(Gv.rec.values[p2]).sum()
            for r_, d in ds.items(): rows.append(dict(val=tag, seed=str(sd), arm=a, base=bn, rec=r_, dsse=d))
O = pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "ec_rv1_preds_v1.csv")); O = O[O.tag == "CUR"].copy(); O["rec"] = O.row_id.str[:7]
for (v, sd), Gv in O.groupby(["validator", "seed"]):
    Gv = Gv.sort_values(["rec", "hour"]).reset_index(drop=True); y = Gv.sub_ec.values; sg = Gv.sg.values; p2 = Gv.day.values >= 179
    cm = Gv.groupby("rec").sg.transform(lambda s: s.expanding().mean()).values
    hidden = Gv.groupby("fold").rec.apply(set).to_dict(); fold_of = Gv.groupby("rec").fold.first().to_dict(); hsf = lambda r: hidden.get(fold_of[r], set())
    recrows = {r: g.index.values for r, g in Gv.groupby("rec")}
    v7val = {r: EC[idx[p], 23] for r, p in V7.items() if r in recrows and p not in hsf(r)}
    run(Gv, y, sg, p2, cm, recrows, v7val, hsf, v, sd)
FO = pd.read_csv(os.path.join(HERE, "ev_validator", "ev_folds_v1.csv")); hid = {k: set(g.rec[g.role.isin(["val", "del"])]) for k, g in FO.groupby("fold")}
R = pd.read_csv(os.path.join(HERE, "ev_validator", "ec", "out_v1", "v2_ev_ec_rows.csv")); R = R[(R.arm == "BASE") & (R.seed.astype(str) != "avg")]
for sd, Gv in R.groupby(R.seed.astype(str)):
    Gv = Gv.sort_values(["rec", "hour"]).reset_index(drop=True); y = Gv.y.values; sg = Gv.SG2.values; ea = Gv.EA2.values; p2 = Gv.day.values >= 179
    cm = Gv.groupby("rec").SG2.transform(lambda s: s.expanding().mean()).values; recrows = {r: g.index.values for r, g in Gv.groupby("rec")}
    fo = {r: int(Gv.fold.values[ix[0]]) for r, ix in recrows.items()}
    v7val = {r: float(ea[recrows[r][0]]) for r in set(Gv.rec[Gv.anchored.astype(bool)])}
    run(Gv, y, sg, p2, cm, recrows, v7val, lambda r: hid[fo[r]], "EV", sd)
S = pd.DataFrame(res); D = pd.DataFrame(rows); A = pd.DataFrame(accs); pd.set_option("display.width", 240)
print("pass-2 RMSE % change:"); print(S.pivot_table(index=["arm", "base"], columns=["val", "seed"], values="p2_pct").round(2).to_string())
print("all-rows % change (seed mean):"); print(S.pivot_table(index=["arm", "base"], columns="val", values="all_pct", aggfunc="mean").round(2).to_string())
if len(A): print("\n[.5,.8) pass-2 picks vs clean truth:", A.groupby("val").ok.apply(lambda s: f"{s.dropna().astype(bool).mean():.2f} (n {s.notna().sum()})").to_dict())
boot = lambda d: float((d[rng.integers(0, len(d), (20000, len(d)))].sum(1) >= 0).mean())
for a in ARMS:
    verdict = True
    for bn in ("NONE", "V7"):
        cells = S[(S.arm == a) & (S.base == bn)]; dd = D[(D.arm == a) & (D.base == bn)]; c1 = bool((cells.p2_pct < 0).all())
        Pd = boot(dd[dd.val == "DIAG10"].groupby("rec").dsse.mean().values); Pe = boot(dd[dd.val == "EV"].groupby("rec").dsse.mean().values)
        harm = dd[dd.dsse > 1e-6].groupby("rec").dsse.max(); g = dd.groupby("rec").dsse.mean()
        ok = c1 and ((Pd < .0125 and Pe < .0125) or len(harm) == 0); verdict &= ok
        print(f"\n{a} vs {bn}: cells better {int((cells.p2_pct < 0).sum())}/12 | DIAG10 P {Pd:.4f} EV P {Pe:.4f} | harmed {len(harm)} {harm.sort_values(ascending=False).head(5).round(3).to_dict()} | changed {int((g.abs() > 1e-6).sum())} -> {'pass' if ok else 'fail'}")
        print("   top gains:", (-g).sort_values(ascending=False).head(6).round(3).to_dict())
    print(f"{a}:", "ADOPT" if verdict and a != "V15P8" else ("(reference) " if a == "V15P8" else "") + ("PASS" if verdict else "FAIL"))
# eval application (descriptive): which eval records change on top of 12th / 13th
E15 = pd.read_csv(os.path.join(env.LOCAL, "linker_v10", "v15pi15_eval.csv")).set_index("rec")
E7 = pd.read_csv(os.path.join(env.LOCAL, "linker_v7", "eval60_predecessor_v7.csv"), encoding="utf-8-sig")
v7e = set(E7[(E7.p_h0 >= .8) & E7.prev_is_labelled.astype(bool) & (E7.dT_air.abs() <= 1.5)].row)
print("\neval records with v15 labelled top pick, not v7-anchored:")
print(E15[(~E15.index.isin(v7e)) & E15.lab0.astype(bool)][["prev0", "p0"]].sort_values("p0", ascending=False).round(3).to_string())
