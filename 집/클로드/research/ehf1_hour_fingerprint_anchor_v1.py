# -*- coding: utf-8 -*-
"""EHF1 (2026-10-13 집 클로드; 사용자: '있는데 못 찾은 것부터 해결').  사전 고정(이 파일 커밋 후 실행).  EC 만.
근거(HF1, 거울 G0 서술): v15 1순위가 맞는지 대상 0..h시 운영 지문으로 확인 가능 — 1순위가 지문 최근접일 비율 맞음 .54~.55 vs 틀림 .18~.36(h ≥ 9), AUC .72~.76;
  2차 '있는데 못 찾음' 8 중 5 가 h≥9~12 에서 지문 최근접, 오연결 F13_214→F47_120 은 내내 4~5등.  (9시·문턱은 G0 를 보고 정함 → 판정은 G1·EV 에서만.)
규칙(h 시 행, 규정: 대상 자기 0..h시 입력): v7 앵커 기록은 v7 그대로(전 시각). 나머지 기록은 h ≥ 9 인 행에서만 —
  v15 1순위가 보이는 정답 기록(폴드 숨김 아님) & p0 ≥ T & 0..h시 지문 거리(8변수, |차| 4 에서 자름)로 보이는 정답 후보(상위 6 중) 최근접 이면
  pred_h = clip(sg_h + (A − sg 0..h 누적평균)), A = 그 짝 23시 EC.  아니면 sg.
안(2개 → 본페로니 α .0125): HF5 = T .5, HF3 = T .3.
판정 기하: rv1 DIAG10·P2LOO·EL1 x 시드 47/1414/6464 에는 G1 거울 짝(v15g1_mirror), EV x 시드 7/101/2024 에는 v15pi15_ev(G0 거울 미사용).
기준(NONE·V7 각각 모두): (1) 12칸 2차 행 RMSE 개선; (2) DIAG10 시드별·EV(시드 평균) 2차 기록 부트스트랩 P(악화) < .0125;
  (3) DIAG10(시드 평균)·EV 2차 이득 상위 3기록 제외 ΔSSE 합 < 0; (4) DIAG10 시드 평균 2차 바뀐 기록 ≥ 5.   모두 → ADOPT.
"""
import os, sys
import env  # noqa: F401
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "linker_v4")); from feat4 import EC, FARM, NUM, N, TRAIN  # noqa: E402
rid = lambda i: f"{FARM[i]}_{NUM[i]:03d}"; idx = {rid(i): i for i in range(N)}
rm = lambda e: float(np.sqrt(np.mean(np.square(e)))); rng = np.random.default_rng(20261101)
lo_, hi_ = np.nanmin(EC[TRAIN]), np.nanmax(EC[TRAIN])
_V = ["act_heating", "act_thermal", "act_vent", "act_shade", "act_circfan", "act_co2", "act_fog", "in_co2"]
_X = pd.concat([pd.read_csv(os.path.join(env.DATA, f)) for f in ("train_X.csv", "test_X.csv")])
_X = _X[_X.row_id.str[:3].isin(["F13", "F47"])]; _X["rec"] = _X.row_id.str[:7]; _X["h"] = _X.row_id.str[8:].astype(int)
FPA = np.stack([_X.pivot(index="rec", columns="h", values=v).reindex(index=[rid(i) for i in range(N)], columns=range(24)).values for v in _V], 2)
FPA = (FPA - np.nanmean(FPA, (0, 1))) / (np.nanstd(FPA, (0, 1)) + 1e-9)
dh = lambda b, a, h: float(np.nanmean(np.minimum(np.abs(FPA[a, :h + 1] - FPA[b, :h + 1]), 4)))
def hour_anchor(row, r, hs, T):
    """returns array over hours: A value where anchored else nan"""
    out = np.full(24, np.nan)
    pv = row.get("prev0")
    if not isinstance(pv, str) or not bool(row["lab0"]) or pv in hs or row["p0"] < T or not np.isfinite(EC[idx[pv], 23]): return out
    vis = [row[f"prev{k}"] for k in range(6) if isinstance(row.get(f"prev{k}"), str) and bool(row[f"lab{k}"]) and row[f"prev{k}"] not in hs]
    b = idx[r]
    for h in range(9, 24):
        d = {c: dh(b, idx[c], h) for c in vis}
        if min(d, key=d.get) == pv: out[h] = EC[idx[pv], 23]
    return out
L7 = pd.read_csv(os.path.join(env.LOCAL, "linker_v7", "cv_all400_choice_v7m.csv"), encoding="utf-8-sig")
V7 = {r: p for r, p, q, l, d in zip(L7.rec, L7.prev, L7.p, L7.prev_is_labelled, L7.dT_air) if q >= .8 and l and abs(d) <= 1.5}
PG = pd.read_csv(os.path.join(env.LOCAL, "linker_v10", "v15g1_mirror.csv")).set_index("rec")
PE = pd.read_csv(os.path.join(env.LOCAL, "linker_v10", "v15pi15_ev.csv")).set_index("rec")
ARMS = {"HF5": .5, "HF3": .3}
res = []; dsse = {}; evds = {}
def run(Gv, y, sg, p2, hours, cm, recrows, v7val, picks, hsf, tag, sd):
    base_v7 = sg.copy(); arms = {a: sg.copy() for a in ARMS}
    for r, ix in recrows.items():
        if r in v7val:
            base_v7[ix] = np.clip(sg[ix] + (v7val[r] - cm[ix]), lo_, hi_)
            for a in ARMS: arms[a][ix] = base_v7[ix]
            continue
        if r not in picks.index: continue
        for a, T in ARMS.items():
            Ah = hour_anchor(picks.loc[r], r, hsf(r), T)[hours[ix]]; m = np.isfinite(Ah)
            if m.any(): arms[a][ix[m]] = np.clip(sg[ix[m]] + (Ah[m] - cm[ix[m]]), lo_, hi_)
    preds = {"NONE": sg, "V7": base_v7, **arms}
    for a in ARMS:
        for bn in ("NONE", "V7"):
            b, n = preds[bn], preds[a]
            res.append(dict(val=tag, seed=str(sd), arm=a, base=bn, all_pct=100 * (rm(n - y) / rm(b - y) - 1), p2_pct=100 * (rm((n - y)[p2]) / rm((b - y)[p2]) - 1)))
            ds = pd.Series(((n - y) ** 2 - (b - y) ** 2)[p2]).groupby(Gv.rec.values[p2]).sum()
            if tag == "DIAG10": dsse[(a, bn, sd)] = ds
            if tag == "EV": evds.setdefault((a, bn), []).append(ds)
O = pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "ec_rv1_preds_v1.csv")); O = O[O.tag == "CUR"].copy(); O["rec"] = O.row_id.str[:7]
for (v, sd), Gv in O.groupby(["validator", "seed"]):
    Gv = Gv.sort_values(["rec", "hour"]).reset_index(drop=True); y = Gv.sub_ec.values; sg = Gv.sg.values; p2 = Gv.day.values >= 179
    cm = Gv.groupby("rec").sg.transform(lambda s: s.expanding().mean()).values
    hidden = Gv.groupby("fold").rec.apply(set).to_dict(); fold_of = Gv.groupby("rec").fold.first().to_dict(); hsf = lambda r: hidden.get(fold_of[r], set())
    recrows = {r: g.index.values for r, g in Gv.groupby("rec")}
    v7val = {r: EC[idx[p], 23] for r, p in V7.items() if r in recrows and p not in hsf(r)}
    run(Gv, y, sg, p2, Gv.hour.values, cm, recrows, v7val, PG, hsf, v, sd)
FO = pd.read_csv(os.path.join(HERE, "ev_validator", "ev_folds_v1.csv")); hid = {k: set(g.rec[g.role.isin(["val", "del"])]) for k, g in FO.groupby("fold")}
R = pd.read_csv(os.path.join(HERE, "ev_validator", "ec", "out_v1", "v2_ev_ec_rows.csv")); R = R[(R.arm == "BASE") & (R.seed.astype(str) != "avg")]
for sd, Gv in R.groupby(R.seed.astype(str)):
    Gv = Gv.sort_values(["rec", "hour"]).reset_index(drop=True); y = Gv.y.values; sg = Gv.SG2.values; ea = Gv.EA2.values; p2 = Gv.day.values >= 179
    cm = Gv.groupby("rec").SG2.transform(lambda s: s.expanding().mean()).values; recrows = {r: g.index.values for r, g in Gv.groupby("rec")}
    fo = {r: int(Gv.fold.values[ix[0]]) for r, ix in recrows.items()}; hsf = lambda r: hid[fo[r]]
    anch = set(Gv.rec[Gv.anchored.astype(bool)])
    # V7 on EV = stored EA2: recover per-record anchor value from EA2 at hour 0 (EA2_0 = A when anchored)
    v7val = {r: float(ea[recrows[r][0]]) for r in anch}
    run(Gv, y, sg, p2, Gv.hour.values, cm, recrows, v7val, PE, hsf, "EV", sd)
S = pd.DataFrame(res); pd.set_option("display.width", 240)
print("pass-2 RMSE % change:"); print(S.pivot_table(index=["arm", "base"], columns=["val", "seed"], values="p2_pct").round(2).to_string())
print("\nall-rows % change (seed mean):"); print(S.pivot_table(index=["arm", "base"], columns="val", values="all_pct", aggfunc="mean").round(2).to_string())
boot = lambda d: float((d[rng.integers(0, len(d), (20000, len(d)))].sum(1) >= 0).mean())
for a in ARMS:
    verdict = True
    for bn in ("NONE", "V7"):
        g = S[(S.arm == a) & (S.base == bn)]; c1 = bool((g.p2_pct < 0).all())
        Pd = [boot(dsse[(a, bn, sd)].values) for sd in (47, 1414, 6464)]; Dd = pd.concat([dsse[(a, bn, sd)] for sd in (47, 1414, 6464)], axis=1).mean(axis=1)
        De = pd.concat(evds[(a, bn)], axis=1).mean(axis=1); Pe = boot(De.values)
        t3d, t3e = Dd.sort_values().iloc[3:].sum(), De.sort_values().iloc[3:].sum(); nch = int((Dd.abs() > 1e-6).sum())
        ok = c1 and all(p < .0125 for p in Pd) and Pe < .0125 and t3d < 0 and t3e < 0 and nch >= 5; verdict &= ok
        print(f"\n{a} vs {bn}: all p2 better {c1} | DIAG10 P {np.round(Pd, 4).tolist()} | EV P {Pe:.4f} | drop-top3 DIAG10 {t3d:+.3f} EV {t3e:+.3f} | changed p2 DIAG10 {nch} EV {int((De.abs() > 1e-6).sum())} -> {'pass' if ok else 'fail'}")
        print("   top gains DIAG10:", (-Dd).sort_values(ascending=False).head(5).round(3).to_dict(), "| worst:", (-Dd).sort_values().head(3).round(3).to_dict())
        print("   top gains EV:", (-De).sort_values(ascending=False).head(5).round(3).to_dict(), "| worst:", (-De).sort_values().head(3).round(3).to_dict())
    print(f"EHF1 {a}:", "ADOPT" if verdict else "FAIL")
