# -*- coding: utf-8 -*-
"""TD8 (2026-10-13 집 클로드; 사용자: 비평 통과 순서 '①규정 안 달력 v12 → ②(수정) 차단 짝은 두 번째 방법 확인 → ③강화 기준').  사전 고정(이 파일 커밋 후 실행).  온도만.
근거: 6.750(비평: 이득 소수 기록·차단 해제 복권형·DM11 규정·기하 혼합), 6.754(진짜 짝이면 공기 점프에 배지 안 따라감, 차단 해제 손해는 전부 틀린 짝).
앵커 집합(검증기마다 그 기하에서 고른 짝):
  v7: 13회차 그대로(EV 는 저장 v22, DIAG10 은 v7 거울 CV 앵커 p ≥ .8·정답·|dT| ≤ 1.5).
  + v12(학습 전용 달력 DMT2, local/linker_v10/v12_picks_{ev,mirror}.csv): p ≥ .6 & 정답 기록 & date_status ≠ mismatch & v7 앵커 없는 기록;
      |dT| ≤ 1.5 → 추가;  |dT| > 1.5 → HP2 가 같은 짝일 때만 추가.
      HP2 = 대상 0시 외기 4칸이 정확히 같은 보이는 정답 기록들의 DMT2 날짜 d·구간 s → 같은 구간 d−1 의 보이는 정답 기록 중 '다음날이 이미 확실 연결로 보이는 정답 기록'인 것 제외 → 0시 지문(8변수) 거리 최소. 날짜 없으면 HP2 없음(추가 안 함).
  식: 13회차(q18/q + BETA_h·clip(A − .36 − q0, ±2)); 새 앵커는 dT 차단 없음.
판정(모두 만족해야 ADOPT): EV 4칸·DIAG10 4칸 모두 2차 행 RMSE 개선; 각 검증기 2차 기록만 부트스트랩(4칸 평균 ΔSSE, 20000) P(악화) < .025;
  각 검증기 2차에서 이득 상위 3기록 제외 후 ΔSSE 합 < 0.  서술: 전체 행, G·EL1·EXT10·EXT12, 추가 앵커 수, 차단 짝 중 HP2 일치 수.
"""
import os, sys, importlib.util
import env  # noqa: F401
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.argv = ["x"]
s_ = importlib.util.spec_from_file_location("ta7", os.path.join(HERE, "ta7_anchor_sweep_v1.py")); A7 = importlib.util.module_from_spec(s_); s_.loader.exec_module(A7)
A7.BETA = np.load(os.path.join(env.LOCAL, "linker_v7", "ta8_beta.npy"))
sys.path.insert(0, os.path.join(HERE, "linker_v4")); from feat4 import ST, IN_, X as OUTX, TRAIN, FARM, NUM, N, S3  # noqa: E402
rid = lambda i: f"{FARM[i]}_{NUM[i]:03d}"; idx = {rid(i): i for i in range(N)}
DT = pd.read_csv(os.path.join(env.LOCAL, "dmt2_days.csv")); DAYT = np.array([dict(zip(DT.rec, DT.day)).get(rid(i), np.nan) for i in range(N)]); SEGT = np.array([dict(zip(DT.rec, DT.seg)).get(rid(i), np.nan) for i in range(N)])
TLc = pd.read_pickle(S3 + r"\truth.pkl"); TLc = TLc[TLc.cert]; SUCC = {}
for a_, b_ in zip(TLc.a1, TLc.b): SUCC.setdefault(int(a_), []).append(int(b_))
_V = ["act_heating", "act_thermal", "act_vent", "act_shade", "act_circfan", "act_co2", "act_fog", "in_co2"]
_X = pd.concat([pd.read_csv(os.path.join(env.DATA, f)) for f in ("train_X.csv", "test_X.csv")]); _X = _X[_X.row_id.str[:3].isin(["F13", "F47"])]
_X["rec"] = _X.row_id.str[:7]; _X["h"] = _X.row_id.str[8:].astype(int)
FPA = np.stack([_X.pivot(index="rec", columns="h", values=v).reindex(index=[rid(i) for i in range(N)], columns=range(24)).values for v in _V], 2)
FPA = (FPA - np.nanmean(FPA, (0, 1))) / (np.nanstd(FPA, (0, 1)) + 1e-9)
def hp2(b, lab):
    vi = np.where(lab)[0]; eq = np.all(np.abs(OUTX[vi, 0] - OUTX[b, 0]) < 1e-6, axis=1); m = vi[eq]; m = m[np.isfinite(DAYT[m])]
    if len(m) == 0: return None
    d = pd.Series(DAYT[m]).mode().iloc[0]; sg = pd.Series(SEGT[m][DAYT[m] == d]).mode().iloc[0]
    cand = [a for a in vi if DAYT[a] == d - 1 and SEGT[a] == sg and not any(lab[s] and DAYT[s] == d for s in SUCC.get(a, []))]
    if not cand: return None
    return rid(min(cand, key=lambda a: float(np.nanmean(np.minimum(np.abs(FPA[a, 0] - FPA[b, 0]), 4)))))
def new_anchors(P, labfun, v7set):
    out = {}; ng = nga = 0
    for r, pv, p, dT, ds, lb in zip(P.rec, P.prev, P.p, P.dT, P.date_status, P.labelled):
        if r in v7set or not lb or p < .6 or ds == "mismatch": continue
        if abs(dT) <= 1.5: out[r] = pv; continue
        ng += 1
        if hp2(idx[r], labfun(r)) == pv: out[r] = pv; nga += 1
    return out, ng, nga
rm = lambda e: float(np.sqrt(np.mean(np.square(e)))); rng = np.random.default_rng(20261026)
def judge(name, recs, dsse_cells):
    D = pd.concat(dsse_cells, axis=1).mean(axis=1); v = D.values
    P = float((v[rng.integers(0, len(v), (20000, len(v)))].sum(1) >= 0).mean()); top3 = D.sort_values().iloc[3:].sum()
    return D, P, top3
# ---------- EV ----------
FO = pd.read_csv(os.path.join(HERE, "ev_validator", "ev_folds_v1.csv")); hid = {k: set(g.rec[g.role.isin(["val", "del"])]) for k, g in FO.groupby("fold")}
foldof = {r: k for k, g in FO[FO.role == "val"].groupby("fold") for r in g.rec}
L7 = pd.read_csv(os.path.join(env.LOCAL, "linker_v7", "cv_all400_choice_v7m.csv"), encoding="utf-8-sig")
v7set = {r for r, q, l, d in zip(L7.rec, L7.p, L7.prev_is_labelled, L7.dT_air) if q >= .8 and l and abs(d) <= 1.5}
PE = pd.read_csv(os.path.join(env.LOCAL, "linker_v10", "v12_picks_ev.csv"))
lab_ev = lambda r: TRAIN & ~np.array([rid(i) in hid[foldof[r]] for i in range(N)])
NEW_EV, ng, nga = new_anchors(PE, lab_ev, v7set)
print(f"EV: new v12 anchors {len(NEW_EV)} (gated {ng}, HP2-agreed {nga})", flush=True)
DD = pd.read_pickle(os.path.join(HERE, "ev_validator", "temp", "local_ckpt", "ev_all.pkl")); DD["rec"] = DD.row_id.str[:7]; DD = DD.rename(columns={"sub_temp": "y"}).reset_index(drop=True)
y = DD.y.values; p2 = DD.day.values >= 179; ev_p2, ev_all, ev_ds = [], [], []
for c in ["7A", "7B", "101A", "101B"]:
    base = DD[f"v22_a_{c}"].values; new = base.copy(); q18 = DD[f"v18_a_{c}"].values
    for rec, G in DD.groupby("rec"):
        if rec not in NEW_EV or NEW_EV[rec] in hid[int(G.fold.iloc[0])]: continue
        ix = G.index.values; hh = G.hour.values; q0 = q18[ix[hh == 0]][0]
        new[ix] = q18[ix] + A7.BETA[hh] * np.clip(ST[idx[NEW_EV[rec]], 23] - .36 - q0, -2, 2)
    ev_p2.append(100 * (rm((new - y)[p2]) / rm((base - y)[p2]) - 1)); ev_all.append(100 * (rm(new - y) / rm(base - y) - 1))
    ev_ds.append(pd.Series(((new - y) ** 2 - (base - y) ** 2)[p2]).groupby(DD.rec.values[p2]).sum())
Dev, Pev, t3ev = judge("EV", None, ev_ds)
print(f"EV pass-2 {[round(x, 2) for x in ev_p2]} | all {[round(x, 2) for x in ev_all]} | P2-only P {Pev:.4f} | drop-top3 sum {t3ev:+.2f} | changed p2 records {(Dev.abs() > 1e-12).sum()}", flush=True)
# ---------- DIAG10 family ----------
PM = pd.read_csv(os.path.join(env.LOCAL, "linker_v10", "v12_picks_mirror.csv"))
def mirror_lab(r):
    b = idx[r]; f = FARM[b]; g = "F47" if f == "F13" else "F13"; sh = -2 if f == "F13" else 2; lo = NUM[FARM == f].min()
    s = lo + ((NUM[b] - lo) // 10) * 10; m = s + sh
    touch = ((FARM == f) & (NUM >= s - 1) & (NUM <= s + 10)); touch[(FARM == g) & (NUM >= m - 1) & (NUM <= m + 10)] = True
    return TRAIN & ~touch
NEW_M, ngm, ngam = new_anchors(PM, mirror_lab, v7set)
print(f"DIAG10: new v12 anchors {len(NEW_M)} (gated {ngm}, HP2-agreed {ngam})", flush=True)
L7f = L7[L7.prev_is_labelled.astype(bool) & L7.anchor_st23.notna() & (L7.steps_to_anchor == 1) & L7.rec.isin(v7set)]
add = pd.DataFrame(dict(rec=list(NEW_M), prev=list(NEW_M.values()), p=1.0, prev_is_labelled=True, anchor_st23=[float(ST[idx[p], 23]) for p in NEW_M.values()], dT_air=0.0, steps_to_anchor=1))
LC = pd.concat([L7f, add], ignore_index=True)
R = A7.A2.A1.frames(); out = []; dres = None
for Lname in ("DIAG10", "G", "EL1", "EXT10", "EXT12"):
    if Lname not in R: continue
    M, cells = R[Lname]; yy = M.sub_temp.values; pp = M.day.values >= 179; kb, Ab = A7.build(M, L7f); kc, Ac = A7.build(M, LC); dp, da, ds = [], [], []
    for q in cells:
        b = A7.correct(M, q, kb, Ab, .8, "const", "meas", 2, 1.5); c = A7.correct(M, q, kc, Ac, .8, "const", "meas", 2, 1.5)
        dp.append(100 * (rm((c - yy)[pp]) / rm((b - yy)[pp]) - 1)); da.append(100 * (rm(c - yy) / rm(b - yy) - 1)); ds.append(pd.Series(((c - yy) ** 2 - (b - yy) ** 2)[pp]).groupby(kb[pp]).sum())
    out.append(dict(layout=Lname, all_pct=" ".join(f"{x:+.2f}" for x in da), p2_pct=" ".join(f"{x:+.2f}" for x in dp)))
    if Lname == "DIAG10": dres = (dp, *judge("DIAG10", None, ds))
dp, Dd, Pd, t3d = dres
print(f"DIAG10 pass-2 {[round(x, 2) for x in dp]} | P2-only P {Pd:.4f} | drop-top3 sum {t3d:+.2f} | changed p2 records {(Dd.abs() > 1e-12).sum()}")
pd.set_option("display.width", 200); print(pd.DataFrame(out).to_string(index=False))
ok = all(x < 0 for x in ev_p2) and all(x < 0 for x in dp) and Pev < .025 and Pd < .025 and t3ev < 0 and t3d < 0
print("TD8:", "ADOPT" if ok else "FAIL")
print("top pass-2 gains EV:", (-Dev).sort_values(ascending=False).head(5).round(2).to_dict(), "| DIAG10:", (-Dd).sort_values(ascending=False).head(5).round(2).to_dict())
