# TD8b (2026-10-13 집 클로드) 사후 서술: TD8 EV 전체 행 이득이 얼마나 넓게 퍼졌나.
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
    ev_ds.append(pd.Series(((new - y) ** 2 - (base - y) ** 2)).groupby(DD.rec.values).sum())

D = pd.concat(ev_ds, axis=1).mean(axis=1); D = D[D.abs() > 1e-12]
print("EV ALL rows, changed records", len(D), "| improved", int((D < 0).sum()), "worsened", int((D > 0).sum()), "| sum", round(D.sum(), 2))
for k in (0, 1, 3, 5, 10): print(f"   drop top-{k} gains: sum {D.sort_values().iloc[k:].sum():+.2f}")
print("top gains:", (-D).sort_values(ascending=False).head(8).round(1).to_dict())
