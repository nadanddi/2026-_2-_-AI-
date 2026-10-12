# -*- coding: utf-8 -*-
"""V15H (2026-10-13 집 클로드; 사용자: 'EC 0.1 아래로').  사전 고정(이 파일 커밋 후 실행).  EC.  과거 정보만.
동기: V15P5(6.787 예정) — 확신 .5~.8 짝 정확도 ~.5 라 실패.  두 독립 방법 일치로 정밀도 확보(ECL2 6.757: HP2 일치 요구 EC 변형 정확 .987, 2차 1.000).
규칙: v7 앵커 기록은 v7 그대로 → 나머지는 v15(PI .15) 1순위 pv 가 보이는 정답 & p0 ≥ .5 & HP2 짝 == pv 이면 전 시각 EA2 앵커(A = pv 23시) → 아니면 sg.
  HP2 = 대상 0시 외기 4칸과 정확히 같은 보이는 정답 기록들의 학습 전용 달력 DMT2 날짜 d·구간 s → 같은 구간 d−1 의 보이는 정답 기록 중
        '다음날이 이미 확실 연결로 보이는 정답 기록'인 것 제외 → 0시 운영 지문(학습 기록으로만 표준화) 최근접.  (TD8 6.755 와 같은 정의)
안 1개.  판정(기준선 NONE·V7 각각): 12칸 모두 2차 행 RMSE 개선 AND ((a) DIAG10·EV 2차 기록 부트스트랩 P(악화) < .025 또는 (b) 어느 칸에서도 2차 기록 ΔSSE > 1e-6 없음) → ADOPT.
서술: 추가된 짝 정확도(깨끗한 진실), 바뀐·손해 기록, 평가 적용 시 추가 기록.
"""
import os, sys
import env  # noqa: F401
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "linker_v4")); from feat4 import EC, X as OUTX, FARM, NUM, N, TRAIN, S3  # noqa: E402
rid = lambda i: f"{FARM[i]}_{NUM[i]:03d}"; idx = {rid(i): i for i in range(N)}
rm = lambda e: float(np.sqrt(np.mean(np.square(e)))); rng = np.random.default_rng(20261115)
lo_, hi_ = np.nanmin(EC[TRAIN]), np.nanmax(EC[TRAIN])
DT = pd.read_csv(os.path.join(env.LOCAL, "dmt2_days.csv")); DAYT = np.array([dict(zip(DT.rec, DT.day)).get(rid(i), np.nan) for i in range(N)]); SEGT = np.array([dict(zip(DT.rec, DT.seg)).get(rid(i), np.nan) for i in range(N)])
TL = pd.read_pickle(os.path.join(S3, "truth.pkl")); TLc = TL[TL.cert]; SUCC = {}
for a_, b_ in zip(TLc.a1, TLc.b): SUCC.setdefault(int(a_), []).append(int(b_))
CAL = pd.read_csv(os.path.join(HERE, "ec0h", "rise", "dm11_calendar_v1.csv")); D11 = dict(zip(CAL.rec, CAL.day))
CLEAN = {rid(int(b)): rid(int(a)) for a, b in zip(TL.a1, TL.b) if D11.get(rid(int(b)), np.nan) - D11.get(rid(int(a)), np.nan) == 1}
_V = ["act_heating", "act_thermal", "act_vent", "act_shade", "act_circfan", "act_co2", "act_fog", "in_co2"]
_X = pd.concat([pd.read_csv(os.path.join(env.DATA, f)) for f in ("train_X.csv", "test_X.csv")]); _X = _X[_X.row_id.str[:3].isin(["F13", "F47"])]
_X["rec"] = _X.row_id.str[:7]; _X["h"] = _X.row_id.str[8:].astype(int)
FPA = np.stack([_X.pivot(index="rec", columns="h", values=v).reindex(index=[rid(i) for i in range(N)], columns=range(24)).values for v in _V], 2)
FPA = (FPA - np.nanmean(FPA[TRAIN], (0, 1))) / (np.nanstd(FPA[TRAIN], (0, 1)) + 1e-9)
def hp2(b, lab):
    vi = np.where(lab)[0]; vi = vi[vi != b]; eq = np.all(np.abs(OUTX[vi, 0] - OUTX[b, 0]) < 1e-6, axis=1); m = vi[eq]; m = m[np.isfinite(DAYT[m])]
    if len(m) == 0: return None
    d = pd.Series(DAYT[m]).mode().iloc[0]; sg = pd.Series(SEGT[m][DAYT[m] == d]).mode().iloc[0]
    cand = [a for a in vi if DAYT[a] == d - 1 and SEGT[a] == sg and not any(lab[s] and DAYT[s] == d for s in SUCC.get(a, []))]
    if not cand: return None
    return rid(min(cand, key=lambda a: float(np.nanmean(np.minimum(np.abs(FPA[a, 0] - FPA[b, 0]), 4)))))
L7 = pd.read_csv(os.path.join(env.LOCAL, "linker_v7", "cv_all400_choice_v7m.csv"), encoding="utf-8-sig")
V7 = {r: p for r, p, q, l, d in zip(L7.rec, L7.prev, L7.p, L7.prev_is_labelled, L7.dT_air) if q >= .8 and l and abs(d) <= 1.5}
FWD = {"RV": pd.read_csv(os.path.join(env.LOCAL, "linker_v10", "v15g1_mirror.csv")).set_index("rec"), "EV": pd.read_csv(os.path.join(env.LOCAL, "linker_v10", "v15pi15_ev.csv")).set_index("rec")}
res, rows, adds = [], [], []
def run(Gv, y, sg, p2, cm, recrows, v7val, hsf, tag, sd):
    base_v7 = sg.copy(); P = sg.copy(); FP = FWD["EV" if tag == "EV" else "RV"]
    for r, ix in recrows.items():
        if r in v7val:
            base_v7[ix] = np.clip(sg[ix] + (v7val[r] - cm[ix]), lo_, hi_); P[ix] = base_v7[ix]; continue
        if r not in FP.index: continue
        fr = FP.loc[r]; pv = fr.get("prev0")
        if not (isinstance(pv, str) and bool(fr["lab0"]) and pv not in hsf(r) and np.isfinite(EC[idx[pv], 23]) and fr["p0"] >= .5): continue
        hs = hsf(r); lab = TRAIN & ~np.array([rid(i) in hs for i in range(N)])
        if hp2(idx[r], lab) != pv: continue
        P[ix] = np.clip(sg[ix] + (EC[idx[pv], 23] - cm[ix]), lo_, hi_)
        adds.append(dict(val=tag, seed=str(sd), rec=r, p0=fr["p0"], p2=NUM[idx[r]] >= 179, ok=(CLEAN.get(r) == pv) if r in CLEAN else None))
    for bn, b_ in (("NONE", sg), ("V7", base_v7)):
        res.append(dict(val=tag, seed=str(sd), base=bn, all_pct=100 * (rm(P - y) / rm(b_ - y) - 1), p2_pct=100 * (rm((P - y)[p2]) / rm((b_ - y)[p2]) - 1)))
        ds = pd.Series(((P - y) ** 2 - (b_ - y) ** 2)[p2]).groupby(Gv.rec.values[p2]).sum()
        for r_, d in ds.items(): rows.append(dict(val=tag, seed=str(sd), base=bn, rec=r_, dsse=d))
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
S = pd.DataFrame(res); D = pd.DataFrame(rows); A = pd.DataFrame(adds); pd.set_option("display.width", 240)
print("pass-2 RMSE % change:"); print(S.pivot_table(index="base", columns=["val", "seed"], values="p2_pct").round(2).to_string())
print("all-rows % change (seed mean):"); print(S.pivot_table(index="base", columns="val", values="all_pct", aggfunc="mean").round(2).to_string())
if len(A):
    print("\nadded anchors per cell (mean):", A.groupby(["val", "seed"]).size().groupby("val").mean().round(1).to_dict(), "| pass-2:", A[A.p2].groupby(["val", "seed"]).size().groupby("val").mean().round(1).to_dict())
    print("added-pick accuracy vs clean truth (all / pass-2):", round(float(A.ok.dropna().astype(bool).mean()), 3), int(A.ok.notna().sum()), "/", round(float(A[A.p2].ok.dropna().astype(bool).mean()), 3) if A[A.p2].ok.notna().any() else None, int(A[A.p2].ok.notna().sum()))
boot = lambda d: float((d[rng.integers(0, len(d), (20000, len(d)))].sum(1) >= 0).mean())
verdict = True
for bn in ("NONE", "V7"):
    cells = S[S.base == bn]; dd = D[D.base == bn]; c1 = bool((cells.p2_pct < 0).all())
    Pd = boot(dd[dd.val == "DIAG10"].groupby("rec").dsse.mean().values); Pe = boot(dd[dd.val == "EV"].groupby("rec").dsse.mean().values)
    harm = dd[dd.dsse > 1e-6].groupby("rec").dsse.max(); g = dd.groupby("rec").dsse.mean()
    ok = c1 and ((Pd < .025 and Pe < .025) or len(harm) == 0); verdict &= ok
    print(f"\nV15H vs {bn}: cells better {int((cells.p2_pct < 0).sum())}/12 | DIAG10 P {Pd:.4f} EV P {Pe:.4f} | harmed {len(harm)} {harm.sort_values(ascending=False).head(5).round(3).to_dict()} | changed {int((g.abs() > 1e-6).sum())} -> {'pass' if ok else 'fail'}")
    print("   top gains:", (-g).sort_values(ascending=False).head(6).round(3).to_dict())
print("V15H:", "ADOPT" if verdict else "FAIL")
E15 = pd.read_csv(os.path.join(env.LOCAL, "linker_v10", "v15pi15_eval.csv")).set_index("rec")
E7 = pd.read_csv(os.path.join(env.LOCAL, "linker_v7", "eval60_predecessor_v7.csv"), encoding="utf-8-sig")
v7e = set(E7[(E7.p_h0 >= .8) & E7.prev_is_labelled.astype(bool) & (E7.dT_air.abs() <= 1.5)].row)
print("\neval: v15 p0 >= .5 labelled non-v7 records and HP2 agreement:")
for r, fr in E15[(~E15.index.isin(v7e)) & E15.lab0.astype(bool) & (E15.p0 >= .5)].iterrows():
    h = hp2(idx[r], TRAIN.copy()); print(f"   {r} <- {fr.prev0} p0 {fr.p0:.3f} | HP2 {h} -> {'ANCHOR' if h == fr.prev0 else '-'} (A {EC[idx[fr.prev0], 23]:.3f})")
