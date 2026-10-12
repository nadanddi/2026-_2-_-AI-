# -*- coding: utf-8 -*-
"""ECJ1 (2026-10-13 집 클로드; 사용자: '일단 후보 2개로 검증해보자').  사전 고정(이 파일 커밋 후 실행).  EC 만.
후보(6.757·6.758):
  ECV = EC 변형 = v7 앵커 ∪ v12(p ≥ .6 & 정답 기록 & date_status ≠ mismatch & v7 앵커 없는 기록) 중 HP2 가 같은 짝을 고른 것.
  V12 = v12 단독 = v12(p ≥ .6 & 정답 & ≠ mismatch), 실내 차 차단·HP2 없음 (v7 미포함).
기준 2개: NONE = 앵커 없음(12회차 EC ≈ sg/SG2), V7 = v7 앵커(13회차 EC ≈ EA2).
식(EA2): pred_h = clip(sg_h + (A − sg 0..h 누적평균), 학습 EC 범위), A = 짝의 23시 EC 정답. 짝이 그 폴드의 숨김 기록이면 버림.
검증기:
  rv1(연구실/클로드/results/ec_rv1_preds_v1.csv, tag CUR, sg): DIAG10·P2LOO·EL1 x 시드 47/1414/6464, 짝 = 거울 CV 기하(v7 cv_all400_choice_v7m, v12_picks_mirror, HP2 는 거울 숨김).
  EV(ev_validator/ec/out_v1/v2_ev_ec_rows.csv, BASE, 시드 7/101/2024): V7 = 저장 EA2 그대로, ECV = EA2 + 새 앵커(v12_picks_ev, HP2 는 EV 폴드 숨김), V12 = SG2 + v12_picks_ev 앵커.
판정(본페로니 2안 → α .0125), 후보마다 기준 NONE·V7 각각에 대해 모두 만족해야 ADOPT:
  (1) rv1 3검증기 x 3시드 + EV 3시드, 모두 2차(번호 ≥ 179) 행 RMSE 개선;
  (2) DIAG10(시드별)·EV(시드 평균 ΔSSE) 2차 기록만 부트스트랩(20000) P(악화) < .0125;
  (3) DIAG10(시드 평균)·EV 2차에서 이득 상위 3기록 제외 후 ΔSSE 합 < 0.
서술: 전체 행 %, 앵커 수, 2차 바뀐 기록 수, 이득 상위 기록.
"""
import os, sys
import env  # noqa: F401
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.argv = ["x"]
src = open(os.path.join(HERE, "td8_rule_clean_strict_v1.py"), encoding="utf-8").read()
exec(src[:src.index("rm = lambda e:")])  # hp2, idx, rid, TRAIN, FARM, NUM, N
from feat4 import EC  # noqa: E402
rm = lambda e: float(np.sqrt(np.mean(np.square(e)))); rng = np.random.default_rng(20261027)
def mirror_lab(r):
    b = idx[r]; f = FARM[b]; g = "F47" if f == "F13" else "F13"; sh = -2 if f == "F13" else 2; lo = NUM[FARM == f].min()
    s = lo + ((NUM[b] - lo) // 10) * 10; m = s + sh
    touch = ((FARM == f) & (NUM >= s - 1) & (NUM <= s + 10)); touch[(FARM == g) & (NUM >= m - 1) & (NUM <= m + 10)] = True
    return TRAIN & ~touch
L7 = pd.read_csv(os.path.join(env.LOCAL, "linker_v7", "cv_all400_choice_v7m.csv"), encoding="utf-8-sig")
V7 = {r: p for r, p, q, l, d in zip(L7.rec, L7.prev, L7.p, L7.prev_is_labelled, L7.dT_air) if q >= .8 and l and abs(d) <= 1.5}
def cands(P, labfun, v7set):
    ecv_new, v12 = {}, {}
    for r, pv, p, ds, lb in zip(P.rec, P.prev, P.p, P.date_status, P.labelled):
        if not lb or p < .6 or ds == "mismatch": continue
        v12[r] = pv
        if r not in v7set and hp2(idx[r], labfun(r)) == pv: ecv_new[r] = pv
    return ecv_new, v12
PM = pd.read_csv(os.path.join(env.LOCAL, "linker_v10", "v12_picks_mirror.csv"))
ECVn_m, V12_m = cands(PM, mirror_lab, set(V7))
ARMS_M = {"V7": V7, "ECV": {**ECVn_m, **V7}, "V12": V12_m}
print("mirror anchors:", {k: len(v) for k, v in ARMS_M.items()}, "| ECV new", len(ECVn_m), flush=True)
lo_, hi_ = np.nanmin(EC[TRAIN]), np.nanmax(EC[TRAIN])
def apply(rec, sg, cm, A, hidden_set):
    new = sg.copy(); nch = 0
    for r, ix in rec.items():
        if r not in A or A[r] in hidden_set(r): continue
        new[ix] = np.clip(sg[ix] + (EC[idx[A[r]], 23] - cm[ix]), lo_, hi_); nch += 1
    return new, nch
res = []; dsse = {}
# ---------- rv1 ----------
O = pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "ec_rv1_preds_v1.csv")); O = O[O.tag == "CUR"].copy(); O["rec"] = O.row_id.str[:7]
for (v, sd), G in O.groupby(["validator", "seed"]):
    G = G.sort_values(["rec", "hour"]).reset_index(drop=True); y = G.sub_ec.values; sg = G.sg.values; p2 = G.day.values >= 179
    cm = G.groupby("rec").sg.transform(lambda s: s.expanding().mean()).values
    hidden = G.groupby("fold").rec.apply(set).to_dict(); fold_of = G.groupby("rec").fold.first().to_dict(); hs = lambda r: hidden.get(fold_of[r], set())
    rec = {r: g.index.values for r, g in G.groupby("rec")}
    preds = {"NONE": sg}
    for arm, A in ARMS_M.items(): preds[arm] = apply(rec, sg, cm, A, hs)[0]
    for arm in ("ECV", "V12"):
        for bn in ("NONE", "V7"):
            b, n = preds[bn], preds[arm]
            res.append(dict(val=v, seed=str(sd), arm=arm, base=bn, all_pct=100 * (rm(n - y) / rm(b - y) - 1), p2_pct=100 * (rm((n - y)[p2]) / rm((b - y)[p2]) - 1)))
            if v == "DIAG10": dsse[("DIAG10", arm, bn, sd)] = pd.Series(((n - y) ** 2 - (b - y) ** 2)[p2]).groupby(G.rec.values[p2]).sum()
# ---------- EV ----------
FO = pd.read_csv(os.path.join(HERE, "ev_validator", "ev_folds_v1.csv")); hid = {k: set(g.rec[g.role.isin(["val", "del"])]) for k, g in FO.groupby("fold")}
foldof = {r: k for k, g in FO[FO.role == "val"].groupby("fold") for r in g.rec}
lab_ev = lambda r: TRAIN & ~np.array([rid(i) in hid[foldof[r]] for i in range(N)])
R = pd.read_csv(os.path.join(HERE, "ev_validator", "ec", "out_v1", "v2_ev_ec_rows.csv")); R = R[(R.arm == "BASE") & (R.seed.astype(str) != "avg")]
anch_ev = set(R[R.anchored.astype(bool)].rec)
PE = pd.read_csv(os.path.join(env.LOCAL, "linker_v10", "v12_picks_ev.csv"))
ECVn_e, V12_e = cands(PE, lab_ev, anch_ev)
print("EV: v7(EA2) anchored", len(anch_ev), "| ECV new", len(ECVn_e), "| V12", len(V12_e), flush=True)
ev_ds = {}
for sd, G in R.groupby(R.seed.astype(str)):
    G = G.sort_values(["rec", "hour"]).reset_index(drop=True); y = G.y.values; sg = G.SG2.values; ea = G.EA2.values; p2 = G.day.values >= 179
    cm = G.groupby("rec").SG2.transform(lambda s: s.expanding().mean()).values; rec = {r: g.index.values for r, g in G.groupby("rec")}
    hs = lambda r: hid[int(G.fold.values[rec[r][0]])]
    ecv, n1 = apply(rec, sg, cm, ECVn_e, hs); ecv = np.where(np.isin(G.rec.values, list(ECVn_e)) & (ecv != sg), ecv, ea)
    preds = {"NONE": sg, "V7": ea, "ECV": ecv, "V12": apply(rec, sg, cm, V12_e, hs)[0]}
    for arm in ("ECV", "V12"):
        for bn in ("NONE", "V7"):
            b, n = preds[bn], preds[arm]
            res.append(dict(val="EV", seed=sd, arm=arm, base=bn, all_pct=100 * (rm(n - y) / rm(b - y) - 1), p2_pct=100 * (rm((n - y)[p2]) / rm((b - y)[p2]) - 1)))
            ev_ds.setdefault((arm, bn), []).append(pd.Series(((n - y) ** 2 - (b - y) ** 2)[p2]).groupby(G.rec.values[p2]).sum())
S = pd.DataFrame(res); pd.set_option("display.width", 240)
print("\npass-2 RMSE % change:"); print(S.pivot_table(index=["arm", "base"], columns=["val", "seed"], values="p2_pct").round(2).to_string())
print("\nall-rows % change (seed mean):"); print(S.pivot_table(index=["arm", "base"], columns="val", values="all_pct", aggfunc="mean").round(2).to_string())
def boot(d): return float((d[rng.integers(0, len(d), (20000, len(d)))].sum(1) >= 0).mean())
for arm in ("ECV", "V12"):
    verdict = True
    for bn in ("NONE", "V7"):
        g = S[(S.arm == arm) & (S.base == bn)]; c1 = bool((g.p2_pct < 0).all())
        Pd = [boot(dsse[("DIAG10", arm, bn, sd)].values) for sd in (47, 1414, 6464)]
        Dd = pd.concat([dsse[("DIAG10", arm, bn, sd)] for sd in (47, 1414, 6464)], axis=1).mean(axis=1)
        De = pd.concat(ev_ds[(arm, bn)], axis=1).mean(axis=1); Pe = boot(De.values)
        t3d, t3e = Dd.sort_values().iloc[3:].sum(), De.sort_values().iloc[3:].sum()
        ok = c1 and all(p < .0125 for p in Pd) and Pe < .0125 and t3d < 0 and t3e < 0; verdict &= ok
        print(f"\n{arm} vs {bn}: all p2 better {c1} | DIAG10 P {np.round(Pd, 4).tolist()} | EV P {Pe:.4f} | drop-top3 DIAG10 {t3d:+.3f} EV {t3e:+.3f} | "
              f"changed p2 recs DIAG10 {(Dd.abs() > 1e-12).sum()} EV {(De.abs() > 1e-12).sum()} -> {'pass' if ok else 'fail'}")
        print("   top gains DIAG10:", (-Dd).sort_values(ascending=False).head(4).round(3).to_dict(), "| worst:", (-Dd).sort_values().head(3).round(3).to_dict())
        print("   top gains EV:", (-De).sort_values(ascending=False).head(4).round(3).to_dict(), "| worst:", (-De).sort_values().head(3).round(3).to_dict())
    print(f"ECJ1 {arm}:", "ADOPT" if verdict else "FAIL")
