# -*- coding: utf-8 -*-
"""NXA1 (2026-10-13 집 클로드; 사용자: '뭐라도 찾아봐').  사전 고정(이 파일 커밋 후 실행).  EC 만.
근거: NX1(6.771) 상한 — 다음날 같은 동 학습 기록 0시 EC 로 거꾸로 앵커, 식별 완벽 시 2차 −4~−8%(β .9 > 1). 규정: 학습 정답은 뒤 날짜도 사용 가능(10-12),
  짝 고르기는 대상 0시 외기(날짜)·대상 0..h 입력(지문)·학습 입력/확실 연결만.  (β .9 는 NX1 을 보고 정함 — 밝혀 둠.)
다음날 짝(행 h 마다): 대상 날짜 (d, s) = 0시 외기 4칸이 똑같은 '보이는 정답 기록'들의 DMT3e 날짜(최빈)·구간; 후보 = 보이는 정답 기록 중 날짜 d+1·같은 구간,
  단 '이미 다른 보이는 정답 기록이 확실 연결(truth cert)의 전날인' 후보는 제외(소거, VF5 방식).
  남은 후보 1개 → 전 시각 사용;  2개 이상 → h ≥ 9 행에서만, 0..h 지문(8변수, |차| 4 자름) 최근접이고 2등 거리 ≥ 1.5 x 1등일 때.
식: pred_h = clip(sg_h + .9·(B − sg 0..h 누적평균)), B = 그 짝 0시 EC.  v7 앞 앵커가 있는 기록은 v7 그대로(앞 앵커 우선).
안(1개, α .025): NXA = V7 앞 앵커 ∪ 나머지 기록 다음날 앵커.  기준 NONE·V7 각각 모두:
  (1) rv1 DIAG10·P2LOO·EL1 x 시드 3 + EV x 시드 3, 12칸 2차 행 RMSE 개선; (2) DIAG10 시드별·EV(시드 평균) 2차 기록 부트스트랩 P(악화) < .025;
  (3) DIAG10(시드 평균)·EV 2차 이득 상위 3 제외 ΔSSE < 0; (4) DIAG10 시드 평균 2차 바뀐 기록 ≥ 5.  → ADOPT.
보이는 정답 = 학습 기록 중 그 검증기 폴드 숨김(rv1: 같은 fold, EV: val∪del) 아닌 것.  서술: 다음날 앵커 수, 정확(진단 연결 다음날과 같은지) 비율.
"""
import os, sys
import env  # noqa: F401
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "linker_v4")); from feat4 import EC, X, FARM, NUM, N, TRAIN, S3  # noqa: E402
rid = lambda i: f"{FARM[i]}_{NUM[i]:03d}"; idx = {rid(i): i for i in range(N)}
rm = lambda e: float(np.sqrt(np.mean(np.square(e)))); rng = np.random.default_rng(20261102)
lo_, hi_ = np.nanmin(EC[TRAIN]), np.nanmax(EC[TRAIN])
C = pd.read_csv(os.path.join(env.LOCAL, "dmt3e_E0_days.csv")); DAYT = np.array([dict(zip(C.rec, C.day)).get(rid(i), np.nan) for i in range(N)]); SEGT = np.array([dict(zip(C.rec, C.seg)).get(rid(i), np.nan) for i in range(N)])
TL = pd.read_pickle(os.path.join(S3, "truth.pkl")); TLc = TL[TL.cert]; PRED = {}
for a_, b_ in zip(TLc.a1, TLc.b): PRED.setdefault(int(b_), []).append(int(a_))
CAL = pd.read_csv(os.path.join(HERE, "ec0h", "rise", "dm11_calendar_v1.csv")); D11 = dict(zip(CAL.rec, CAL.day))
NXT_TRUE = {}
for a_, b_ in zip(TL.a1, TL.b):
    if D11.get(rid(int(b_)), np.nan) - D11.get(rid(int(a_)), np.nan) == 1: NXT_TRUE.setdefault(rid(int(a_)), set()).add(rid(int(b_)))
_V = ["act_heating", "act_thermal", "act_vent", "act_shade", "act_circfan", "act_co2", "act_fog", "in_co2"]
_X = pd.concat([pd.read_csv(os.path.join(env.DATA, f)) for f in ("train_X.csv", "test_X.csv")])
_X = _X[_X.row_id.str[:3].isin(["F13", "F47"])]; _X["rec"] = _X.row_id.str[:7]; _X["h"] = _X.row_id.str[8:].astype(int)
FPA = np.stack([_X.pivot(index="rec", columns="h", values=v).reindex(index=[rid(i) for i in range(N)], columns=range(24)).values for v in _V], 2)
FPA = (FPA - np.nanmean(FPA, (0, 1))) / (np.nanstd(FPA, (0, 1)) + 1e-9)
dh = lambda b, a, h: float(np.nanmean(np.minimum(np.abs(FPA[a, :h + 1] - FPA[b, :h + 1]), 4)))
def next_anchor(r, lab):
    """returns per-hour B (nan where not anchored) and the chosen candidate per hour"""
    b = idx[r]; out = np.full(24, np.nan); pick = [None] * 24
    vi = np.where(lab)[0]; vi = vi[vi != b]; eq = np.all(np.abs(X[vi, 0] - X[b, 0]) < 1e-6, axis=1); m = vi[eq]; m = m[np.isfinite(DAYT[m])]
    if len(m) == 0: return out, pick
    d = pd.Series(DAYT[m]).mode().iloc[0]; s = pd.Series(SEGT[m][DAYT[m] == d]).mode().iloc[0]
    cand = [c for c in np.where(lab & (DAYT == d + 1) & (SEGT == s))[0] if c != b and np.isfinite(EC[c, 0])
            and not any(lab[p] and p != b for p in PRED.get(int(c), []))]
    if len(cand) == 1:
        out[:] = EC[cand[0], 0]; pick = [rid(cand[0])] * 24; return out, pick
    if len(cand) >= 2:
        for h in range(9, 24):
            dd = sorted((dh(b, c, h), c) for c in cand)
            if dd[1][0] >= 1.5 * dd[0][0]: out[h] = EC[dd[0][1], 0]; pick[h] = rid(dd[0][1])
    return out, pick
L7 = pd.read_csv(os.path.join(env.LOCAL, "linker_v7", "cv_all400_choice_v7m.csv"), encoding="utf-8-sig")
V7 = {r: p for r, p, q, l, d in zip(L7.rec, L7.prev, L7.p, L7.prev_is_labelled, L7.dT_air) if q >= .8 and l and abs(d) <= 1.5}
res = []; dsse = {}; evds = {}; acc = []
def run(Gv, y, sg, p2, hours, cm, recrows, v7val, labf, tag, sd):
    base_v7 = sg.copy(); arm = sg.copy()
    for r, ix in recrows.items():
        if r in v7val:
            base_v7[ix] = np.clip(sg[ix] + (v7val[r] - cm[ix]), lo_, hi_); arm[ix] = base_v7[ix]; continue
        Bh, pk = next_anchor(r, labf(r)); Bv = Bh[hours[ix]]; mk = np.isfinite(Bv)
        if mk.any():
            arm[ix[mk]] = np.clip(sg[ix[mk]] + .9 * (Bv[mk] - cm[ix[mk]]), lo_, hi_)
            if tag == "DIAG10" and sd == 47:
                chosen = {p for p in pk if p}; acc.append(dict(rec=r, picks=",".join(sorted(chosen)), true_next=",".join(sorted(NXT_TRUE.get(r, []))),
                                                                ok=bool(chosen & NXT_TRUE.get(r, set())), n_hours=int(mk.sum()), p2=int(r[4:7]) >= 179))
    preds = {"NONE": sg, "V7": base_v7, "NXA": arm}
    for bn in ("NONE", "V7"):
        b_, n_ = preds[bn], preds["NXA"]
        res.append(dict(val=tag, seed=str(sd), base=bn, all_pct=100 * (rm(n_ - y) / rm(b_ - y) - 1), p2_pct=100 * (rm((n_ - y)[p2]) / rm((b_ - y)[p2]) - 1)))
        ds = pd.Series(((n_ - y) ** 2 - (b_ - y) ** 2)[p2]).groupby(Gv.rec.values[p2]).sum()
        if tag == "DIAG10": dsse[(bn, sd)] = ds
        if tag == "EV": evds.setdefault(bn, []).append(ds)
O = pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "ec_rv1_preds_v1.csv")); O = O[O.tag == "CUR"].copy(); O["rec"] = O.row_id.str[:7]
for (v, sd), Gv in O.groupby(["validator", "seed"]):
    Gv = Gv.sort_values(["rec", "hour"]).reset_index(drop=True); y = Gv.sub_ec.values; sg = Gv.sg.values; p2 = Gv.day.values >= 179
    cm = Gv.groupby("rec").sg.transform(lambda s: s.expanding().mean()).values
    hidden = Gv.groupby("fold").rec.apply(set).to_dict(); fold_of = Gv.groupby("rec").fold.first().to_dict()
    HM = {k: np.array([rid(i) in vv for i in range(N)]) for k, vv in hidden.items()}
    labf = lambda r: TRAIN & ~HM[fold_of[r]]
    recrows = {r: g.index.values for r, g in Gv.groupby("rec")}
    v7val = {r: EC[idx[p], 23] for r, p in V7.items() if r in recrows and p not in hidden[fold_of[r]]}
    run(Gv, y, sg, p2, Gv.hour.values, cm, recrows, v7val, labf, v, sd); print("done", v, sd, flush=True)
FO = pd.read_csv(os.path.join(HERE, "ev_validator", "ev_folds_v1.csv")); hid = {k: set(g.rec[g.role.isin(["val", "del"])]) for k, g in FO.groupby("fold")}
HME = {k: np.array([rid(i) in vv for i in range(N)]) for k, vv in hid.items()}
R = pd.read_csv(os.path.join(HERE, "ev_validator", "ec", "out_v1", "v2_ev_ec_rows.csv")); R = R[(R.arm == "BASE") & (R.seed.astype(str) != "avg")]
for sd, Gv in R.groupby(R.seed.astype(str)):
    Gv = Gv.sort_values(["rec", "hour"]).reset_index(drop=True); y = Gv.y.values; sg = Gv.SG2.values; ea = Gv.EA2.values; p2 = Gv.day.values >= 179
    cm = Gv.groupby("rec").SG2.transform(lambda s: s.expanding().mean()).values; recrows = {r: g.index.values for r, g in Gv.groupby("rec")}
    fo = {r: int(Gv.fold.values[ix[0]]) for r, ix in recrows.items()}; labf = lambda r: TRAIN & ~HME[fo[r]]
    v7val = {r: float(ea[recrows[r][0]]) for r in set(Gv.rec[Gv.anchored.astype(bool)])}
    run(Gv, y, sg, p2, Gv.hour.values, cm, recrows, v7val, labf, "EV", sd); print("done EV", sd, flush=True)
A = pd.DataFrame(acc); pd.set_option("display.width", 240)
if len(A): print(f"\nDIAG10 s47 next-day anchors: {len(A)} records (pass-2 {A.p2.sum()}) | pick = true next {A.ok.mean():.3f} (records with known true next {int((A.true_next != '').sum())})"); print(A.to_string(index=False))
S = pd.DataFrame(res)
print("\npass-2 RMSE % change:"); print(S.pivot_table(index="base", columns=["val", "seed"], values="p2_pct").round(2).to_string())
print("\nall-rows % change (seed mean):"); print(S.pivot_table(index="base", columns="val", values="all_pct", aggfunc="mean").round(2).to_string())
boot = lambda d: float((d[rng.integers(0, len(d), (20000, len(d)))].sum(1) >= 0).mean())
verdict = True
for bn in ("NONE", "V7"):
    g = S[S.base == bn]; c1 = bool((g.p2_pct < 0).all())
    Pd = [boot(dsse[(bn, sd)].values) for sd in (47, 1414, 6464)]; Dd = pd.concat([dsse[(bn, sd)] for sd in (47, 1414, 6464)], axis=1).mean(axis=1)
    De = pd.concat(evds[bn], axis=1).mean(axis=1); Pe = boot(De.values)
    t3d, t3e = Dd.sort_values().iloc[3:].sum(), De.sort_values().iloc[3:].sum(); nch = int((Dd.abs() > 1e-6).sum())
    ok = c1 and all(p < .025 for p in Pd) and Pe < .025 and t3d < 0 and t3e < 0 and nch >= 5; verdict &= ok
    print(f"\nNXA vs {bn}: all p2 better {c1} | DIAG10 P {np.round(Pd, 4).tolist()} | EV P {Pe:.4f} | drop-top3 DIAG10 {t3d:+.3f} EV {t3e:+.3f} | changed p2 DIAG10 {nch} EV {int((De.abs() > 1e-6).sum())} -> {'pass' if ok else 'fail'}")
    print("   top gains DIAG10:", (-Dd).sort_values(ascending=False).head(6).round(3).to_dict(), "| worst:", (-Dd).sort_values().head(3).round(3).to_dict())
    print("   top gains EV:", (-De).sort_values(ascending=False).head(6).round(3).to_dict(), "| worst:", (-De).sort_values().head(3).round(3).to_dict())
print("NXA1:", "ADOPT" if verdict else "FAIL")
