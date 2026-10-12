# -*- coding: utf-8 -*-
"""ECN2 (2026-10-13 집 클로드; 사용자: '없음 비율 보정해서 다시').  사전 고정(이 파일 커밋 후 실행, v15b 출력 확인 전 작성).  EC 만.
1단계 선택(거울 CV 짝찾기만, EC 점수 안 봄): PI ∈ {.15, .25, .35} 의 v15pi*_mirror 에서 1순위 p0 ≥ .8 & 정답 기록 앵커의 깨끗한 정답 기준
  정확도(전체) ≥ .98 인 판 중 포괄(전체 + 2차) 합 최대; 같으면 작은 PI. 하나도 없으면 정확도 최고 판.
2단계 판정(선택된 PI 하나, 후보 2개 → 본페로니 α .0125): N8 = V7 ∪ (V7 없는 기록 중 p0 ≥ .8·정답·숨김 아님, 값 = 짝 23시 EC);
  S8 = v15b 단독(v7 안 씀; 모든 기록 p0 ≥ .8·정답·숨김 아님).  기준·검증기·판정 규칙은 ECD1/ECN1 과 같음(NONE·V7 각각 모두 통과해야 ADOPT).
"""
import os, sys
import env  # noqa: F401
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "linker_v4")); from feat4 import EC, FARM, NUM, N, TRAIN, S3  # noqa: E402
rid = lambda i: f"{FARM[i]}_{NUM[i]:03d}"; idx = {rid(i): i for i in range(N)}
TLa = pd.read_pickle(os.path.join(S3, "truth.pkl")); TP = {rid(int(b)): rid(int(a)) for a, b in zip(TLa.a1, TLa.b)}
CAL = pd.read_csv(os.path.join(HERE, "ec0h", "rise", "dm11_calendar_v1.csv")); DAYr = dict(zip(CAL.rec, CAL.day))
def mirror_lab(r):
    b = idx[r]; f = FARM[b]; g = "F47" if f == "F13" else "F13"; sh = -2 if f == "F13" else 2; lo = NUM[FARM == f].min()
    s = lo + ((NUM[b] - lo) // 10) * 10; m = s + sh
    touch = ((FARM == f) & (NUM >= s - 1) & (NUM <= s + 10)); touch[(FARM == g) & (NUM >= m - 1) & (NUM <= m + 10)] = True
    return TRAIN & ~touch
clean = lambda r: r in TP and mirror_lab(r)[idx[TP[r]]] and DAYr.get(r, np.nan) - DAYr.get(TP[r], np.nan) == 1
cands = []
for pi in (15, 25, 35):
    D = pd.read_csv(os.path.join(env.LOCAL, "linker_v10", f"v15pi{pi}_mirror.csv")); out = {}
    for part, sel_ in (("all", lambda r: True), ("p2", lambda r: int(r[4:7]) >= 179)):
        S_ = D[[sel_(r) for r in D.rec]]; can = [r for r in S_.rec if clean(r)]; A = S_[(S_.p0 >= .8) & S_.lab0]
        ok = sum(clean(r) and TP[r] == p for r, p in zip(A.rec, A.prev0)); jd = sum(clean(r) for r in A.rec)
        out[part] = (ok / max(jd, 1), ok / max(len(can), 1), len(A), len(A) - jd)
    print(f"PI .{pi}: all precision {out['all'][0]:.3f} recall {out['all'][1]:.3f} anchors {out['all'][2]} unverifiable {out['all'][3]} | pass-2 precision {out['p2'][0]:.3f} recall {out['p2'][1]:.3f} anchors {out['p2'][2]} unverifiable {out['p2'][3]}")
    cands.append((pi, out["all"][0], out["all"][1] + out["p2"][1]))
el = [c for c in cands if c[1] >= .98]
PIs = (max(el, key=lambda c: (c[2], -c[0])) if el else max(cands, key=lambda c: c[1]))[0]
print("selected PI", PIs, flush=True)
rm = lambda e: float(np.sqrt(np.mean(np.square(e)))); rng = np.random.default_rng(20261030)
lo_, hi_ = np.nanmin(EC[TRAIN]), np.nanmax(EC[TRAIN])
L7 = pd.read_csv(os.path.join(env.LOCAL, "linker_v7", "cv_all400_choice_v7m.csv"), encoding="utf-8-sig")
V7 = {r: p for r, p, q, l, d in zip(L7.rec, L7.prev, L7.p, L7.prev_is_labelled, L7.dT_air) if q >= .8 and l and abs(d) <= 1.5}
PM = pd.read_csv(os.path.join(env.LOCAL, "linker_v10", f"v15pi{PIs}_mirror.csv")).set_index("rec")
PE = pd.read_csv(os.path.join(env.LOCAL, "linker_v10", f"v15pi{PIs}_ev.csv")).set_index("rec")
def ec_anchor(row, nm, hidden):
    th = .8; pv = row.get("prev0")
    if pd.isna(pv) or not bool(row["lab0"]) or pv in hidden or row["p0"] < th or not np.isfinite(EC[idx[pv], 23]): return None, 0.0
    return float(EC[idx[pv], 23]), 1.0
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
    for nm in ("N8", "S8"):
        newv = {}
        for r in rec:
            if (nm == "N8" and r in v7v) or r not in PM.index: continue
            A, c = ec_anchor(PM.loc[r], nm, hs(r))
            if A is not None and c >= .8: newv[r] = A
        preds[nm] = apply(rec, sg, cm, {**newv, **v7v} if nm == "N8" else newv)
        if v == "DIAG10" and sd == 47:
            acc = [abs(A - EC[idx[r], 0]) <= .05 for r, A in newv.items()]
            desc[nm] = (len(newv), float(np.mean(acc)) if acc else np.nan, sum(int(r[4:7]) >= 179 for r in newv))
    for arm in ("N8", "S8"):
        for bn in ("NONE", "V7"):
            b, n = preds[bn], preds[arm]
            res.append(dict(val=v, seed=str(sd), arm=arm, base=bn, all_pct=100 * (rm(n - y) / rm(b - y) - 1), p2_pct=100 * (rm((n - y)[p2]) / rm((b - y)[p2]) - 1)))
            if v == "DIAG10": dsse[("DIAG10", arm, bn, sd)] = pd.Series(((n - y) ** 2 - (b - y) ** 2)[p2]).groupby(G.rec.values[p2]).sum()
print("mirror(DIAG10 s47) new anchors (n, EC-accurate share, pass-2 n):", desc, "| date_src:", PM.date_src.value_counts().to_dict(), flush=True)
KP = [("F47_146", "F13_143"), ("F13_177", "F47_177"), ("F13_214", "F47_120"), ("F13_149", "F13_242"), ("F13_116", "F47_130"), ("F47_079", "F13_072")]
print("harm-history pairs in v15 mirror:", {r: (PM.prev0.get(r), round(PM.p0.get(r, np.nan), 3), round(PM.p_null.get(r, np.nan), 3)) for r, _ in KP})
FO = pd.read_csv(os.path.join(HERE, "ev_validator", "ev_folds_v1.csv")); hid = {k: set(g.rec[g.role.isin(["val", "del"])]) for k, g in FO.groupby("fold")}
R = pd.read_csv(os.path.join(HERE, "ev_validator", "ec", "out_v1", "v2_ev_ec_rows.csv")); R = R[(R.arm == "BASE") & (R.seed.astype(str) != "avg")]
anch_ev = set(R[R.anchored.astype(bool)].rec); ev_ds = {}; nev = {}
for sd, G in R.groupby(R.seed.astype(str)):
    G = G.sort_values(["rec", "hour"]).reset_index(drop=True); y = G.y.values; sg = G.SG2.values; ea = G.EA2.values; p2 = G.day.values >= 179
    cm = G.groupby("rec").SG2.transform(lambda s: s.expanding().mean()).values; rec = {r: g.index.values for r, g in G.groupby("rec")}
    preds = {"NONE": sg, "V7": ea}
    for nm in ("N8", "S8"):
        newv = {}
        for r in rec:
            if (nm == "N8" and r in anch_ev) or r not in PE.index: continue
            A, c = ec_anchor(PE.loc[r], nm, hid[int(G.fold.values[rec[r][0]])])
            if A is not None and c >= .8: newv[r] = A
        nev[nm] = len(newv); x = apply(rec, sg, cm, newv); preds[nm] = np.where(np.isin(G.rec.values, list(newv)), x, ea) if nm == "N8" else x
    for arm in ("N8", "S8"):
        for bn in ("NONE", "V7"):
            b, n = preds[bn], preds[arm]
            res.append(dict(val="EV", seed=sd, arm=arm, base=bn, all_pct=100 * (rm(n - y) / rm(b - y) - 1), p2_pct=100 * (rm((n - y)[p2]) / rm((b - y)[p2]) - 1)))
            ev_ds.setdefault((arm, bn), []).append(pd.Series(((n - y) ** 2 - (b - y) ** 2)[p2]).groupby(G.rec.values[p2]).sum())
print("EV new anchors:", nev)
S = pd.DataFrame(res); pd.set_option("display.width", 240)
print("\npass-2 RMSE % change:"); print(S.pivot_table(index=["arm", "base"], columns=["val", "seed"], values="p2_pct").round(2).to_string())
print("\nall-rows % change (seed mean):"); print(S.pivot_table(index=["arm", "base"], columns="val", values="all_pct", aggfunc="mean").round(2).to_string())
boot = lambda d: float((d[rng.integers(0, len(d), (20000, len(d)))].sum(1) >= 0).mean())
for arm in ("N8", "S8"):
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
    print(f"ECN2 {arm}:", "ADOPT" if verdict else "FAIL")
