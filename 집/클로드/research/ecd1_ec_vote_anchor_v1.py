# -*- coding: utf-8 -*-
"""ECD1 (2026-10-13 집 클로드; 사용자: '날짜 투표 실험, 후보 결정은 EC 특성에 맞게').  사전 고정(이 파일 커밋 후 실행, v14 링커 출력 확인 전 작성).  EC 만.
링커 v14(날짜 투표 + L/M 학습, local/linker_v10/v14_{mirror,ev}.csv). EC 맞춤 결정:
  후보(상위 6) 중 그 검증기 폴드에서 숨김이 아닌 '보이는 정답 기록' 만 값에 씀:
    A = Σ p_k·EC23_k / Σ p_k (보이는 정답 후보),  확신 conf = Σ{보이는 정답 후보 중 |EC23_k − A| ≤ .05} p_k / Σ(상위 6 전체 p)  (숨김·평가 후보 확률은 확신을 깎음)
  앵커: conf ≥ .8.
안(후보 2개): ECD-L = V7 ∪ (V7 앵커 없는 기록 중 L 로 conf ≥ .8, 값 A_L);  ECD-M = 같은 방식 M.
기준 2개: NONE(앵커 없음 ≈ 12회차), V7(13회차; rv1 은 v7 거울 짝 23시 EC, EV 는 저장 EA2).  식(EA2): clip(sg_h + (A − sg 0..h 누적평균), 학습 EC 범위).
검증기: rv1 DIAG10·P2LOO·EL1 x 시드 47/1414/6464(거울 기하 v14_mirror), EV x 시드 7/101/2024(v14_ev).
판정(본페로니 2안 α .0125; 안마다 NONE·V7 각각에 대해 모두 만족해야 ADOPT):
  (1) 12칸 모두 2차 행 RMSE 개선; (2) DIAG10(시드별)·EV(시드 평균) 2차 기록 부트스트랩(20000) P(악화) < .0125; (3) DIAG10(시드 평균)·EV 2차 이득 상위 3기록 제외 ΔSSE 합 < 0.
서술: 전체 행, 새 앵커 수, 거울 새 앵커 EC 정확(|A − 실제 0시 EC| ≤ .05) 비율, 날짜 투표 출처 분포, 이득·손해 상위 기록.
"""
import os, sys
import env  # noqa: F401
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "linker_v4")); from feat4 import EC, FARM, NUM, N, TRAIN  # noqa: E402
rid = lambda i: f"{FARM[i]}_{NUM[i]:03d}"; idx = {rid(i): i for i in range(N)}
rm = lambda e: float(np.sqrt(np.mean(np.square(e)))); rng = np.random.default_rng(20261028)
lo_, hi_ = np.nanmin(EC[TRAIN]), np.nanmax(EC[TRAIN])
L7 = pd.read_csv(os.path.join(env.LOCAL, "linker_v7", "cv_all400_choice_v7m.csv"), encoding="utf-8-sig")
V7 = {r: p for r, p, q, l, d in zip(L7.rec, L7.prev, L7.p, L7.prev_is_labelled, L7.dT_air) if q >= .8 and l and abs(d) <= 1.5}
PM = pd.read_csv(os.path.join(env.LOCAL, "linker_v10", "v14_mirror.csv")).set_index("rec")
PE = pd.read_csv(os.path.join(env.LOCAL, "linker_v10", "v14_ev.csv")).set_index("rec")
def ec_anchor(row, nm, hidden):
    tot = sum(row[f"{nm}_p{k}"] for k in range(6) if pd.notna(row.get(f"{nm}_prev{k}")))
    vis = [(row[f"{nm}_p{k}"], EC[idx[row[f"{nm}_prev{k}"]], 23]) for k in range(6)
           if pd.notna(row.get(f"{nm}_prev{k}")) and bool(row[f"{nm}_lab{k}"]) and row[f"{nm}_prev{k}"] not in hidden and np.isfinite(EC[idx[row[f"{nm}_prev{k}"]], 23])]
    if not vis or tot <= 0: return None, 0.0
    w = np.array([v[0] for v in vis]); e = np.array([v[1] for v in vis]); A = float(w @ e / w.sum())
    return A, float(w[np.abs(e - A) <= .05].sum() / tot)
def apply(rec, sg, cm, vals):
    new = sg.copy()
    for r, ix in rec.items():
        if r in vals: new[ix] = np.clip(sg[ix] + (vals[r] - cm[ix]), lo_, hi_)
    return new
res = []; dsse = {}; desc = {}
O = pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "ec_rv1_preds_v1.csv")); O = O[O.tag == "CUR"].copy(); O["rec"] = O.row_id.str[:7]
for (v, sd), G in O.groupby(["validator", "seed"]):
    G = G.sort_values(["rec", "hour"]).reset_index(drop=True); y = G.sub_ec.values; sg = G.sg.values; p2 = G.day.values >= 179
    cm = G.groupby("rec").sg.transform(lambda s: s.expanding().mean()).values
    hidden = G.groupby("fold").rec.apply(set).to_dict(); fold_of = G.groupby("rec").fold.first().to_dict(); hs = lambda r: hidden.get(fold_of[r], set())
    rec = {r: g.index.values for r, g in G.groupby("rec")}
    v7v = {r: EC[idx[p], 23] for r, p in V7.items() if r in rec and p not in hs(r)}
    preds = {"NONE": sg, "V7": apply(rec, sg, cm, v7v)}
    for nm in ("L", "M"):
        newv = {}
        for r in rec:
            if r in v7v or r not in PM.index: continue
            A, c = ec_anchor(PM.loc[r], nm, hs(r))
            if A is not None and c >= .8: newv[r] = A
        preds["ECD-" + nm] = apply(rec, sg, cm, {**newv, **v7v})
        if v == "DIAG10" and sd == 47:
            acc = [abs(A - EC[idx[r], 0]) <= .05 for r, A in newv.items()]
            desc[nm] = (len(newv), float(np.mean(acc)) if acc else np.nan, sum(int(r[4:7]) >= 179 for r in newv))
    for arm in ("ECD-L", "ECD-M"):
        for bn in ("NONE", "V7"):
            b, n = preds[bn], preds[arm]
            res.append(dict(val=v, seed=str(sd), arm=arm, base=bn, all_pct=100 * (rm(n - y) / rm(b - y) - 1), p2_pct=100 * (rm((n - y)[p2]) / rm((b - y)[p2]) - 1)))
            if v == "DIAG10": dsse[("DIAG10", arm, bn, sd)] = pd.Series(((n - y) ** 2 - (b - y) ** 2)[p2]).groupby(G.rec.values[p2]).sum()
print("mirror(DIAG10 s47) new anchors (n, EC-accurate share, pass-2 n):", desc, "| date_src:", PM.date_src.value_counts().to_dict(), flush=True)
FO = pd.read_csv(os.path.join(HERE, "ev_validator", "ev_folds_v1.csv")); hid = {k: set(g.rec[g.role.isin(["val", "del"])]) for k, g in FO.groupby("fold")}
R = pd.read_csv(os.path.join(HERE, "ev_validator", "ec", "out_v1", "v2_ev_ec_rows.csv")); R = R[(R.arm == "BASE") & (R.seed.astype(str) != "avg")]
anch_ev = set(R[R.anchored.astype(bool)].rec); ev_ds = {}; nev = {}
for sd, G in R.groupby(R.seed.astype(str)):
    G = G.sort_values(["rec", "hour"]).reset_index(drop=True); y = G.y.values; sg = G.SG2.values; ea = G.EA2.values; p2 = G.day.values >= 179
    cm = G.groupby("rec").SG2.transform(lambda s: s.expanding().mean()).values; rec = {r: g.index.values for r, g in G.groupby("rec")}
    preds = {"NONE": sg, "V7": ea}
    for nm in ("L", "M"):
        newv = {}
        for r in rec:
            if r in anch_ev or r not in PE.index: continue
            A, c = ec_anchor(PE.loc[r], nm, hid[int(G.fold.values[rec[r][0]])])
            if A is not None and c >= .8: newv[r] = A
        nev[nm] = len(newv); x = apply(rec, sg, cm, newv); preds["ECD-" + nm] = np.where(np.isin(G.rec.values, list(newv)), x, ea)
    for arm in ("ECD-L", "ECD-M"):
        for bn in ("NONE", "V7"):
            b, n = preds[bn], preds[arm]
            res.append(dict(val="EV", seed=sd, arm=arm, base=bn, all_pct=100 * (rm(n - y) / rm(b - y) - 1), p2_pct=100 * (rm((n - y)[p2]) / rm((b - y)[p2]) - 1)))
            ev_ds.setdefault((arm, bn), []).append(pd.Series(((n - y) ** 2 - (b - y) ** 2)[p2]).groupby(G.rec.values[p2]).sum())
print("EV new anchors:", nev)
S = pd.DataFrame(res); pd.set_option("display.width", 240)
print("\npass-2 RMSE % change:"); print(S.pivot_table(index=["arm", "base"], columns=["val", "seed"], values="p2_pct").round(2).to_string())
print("\nall-rows % change (seed mean):"); print(S.pivot_table(index=["arm", "base"], columns="val", values="all_pct", aggfunc="mean").round(2).to_string())
boot = lambda d: float((d[rng.integers(0, len(d), (20000, len(d)))].sum(1) >= 0).mean())
for arm in ("ECD-L", "ECD-M"):
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
    print(f"ECD1 {arm}:", "ADOPT" if verdict else "FAIL")
