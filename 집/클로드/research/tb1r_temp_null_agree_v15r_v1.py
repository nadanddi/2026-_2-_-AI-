# -*- coding: utf-8 -*-
"""[재실행 판 — 오류 재조사: v15 짝을 규정 수정판 v15R(지문 표준화 학습 기록만, linker_v10/v15r_rule_scaling_v1.py) 로 교체, 그 밖 규칙·기준 동일] TB1 (2026-10-13 집 클로드; 사용자: '구조 문제·더 나은 활용 둘 다 차근차근' — B1: EC 에서 만든 '없음' 링커·두 방법 일치를 온도 앵커에).  사전 고정(커밋 후 실행).  온도.
근거: 온도 앵커형 제출 3번 리더보드 개선이 로컬 예상의 1.3~3.6배(6.788) → 온도 앵커 확대가 유력 방향.  v15(없음 선택지, PI .15) 는 확신 ≥ .8 정확 .984,
  V15H(v15 ≥ .5 & HP2 같은 짝)는 추가 짝 정확 1.000(6.787).  과거 정보만(전날 정답).
안(13회차 = v7 앵커 + 측정 감쇠 공식 위에 새 앵커 추가; 새 앵커는 dT 차단 없음, TD8 과 같은 식):
  TN8 = v15 p0 ≥ .8 & 정답 & v7 앵커 없음;  TNH = v15 p0 ≥ .5 & HP2 같은 짝 & 정답 & v7 없음;  TUH = TD8 새 앵커 ∪ TNH(겹치면 TD8 짝).
  기하: EV = v15r15_ev / v12_picks_ev, DIAG10 계열 = v15r15_mirror / v12_picks_mirror(거울 G0, TD8 과 같은 mirror_lab).
판정(TD8 과 같은 강화 기준, 안 3개 본페로니 α .0083): EV 4칸·DIAG10 4칸 모두 2차 개선; 각 검증기 2차 기록 부트스트랩 P < .0083; 각 검증기 상위 3기록 제외 ΔSSE 합 < 0.
  보조 경로(사용자 10-13 작은 표본 경로): 8칸 개선 & 두 검증기 2차 손해 기록 0.  둘 중 하나 → ADOPT.  서술: 전체 행, G·EL1·EXT10·EXT12, 추가 앵커 수, 짝 정확도.
"""
import os, sys
import env  # noqa: F401
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, "td8_rule_clean_strict_v1.py"), encoding="utf-8").read(); __file__ = os.path.join(HERE, "td8_rule_clean_strict_v1.py")
exec(src[:src.index("# ---------- EV ----------")])
rng = np.random.default_rng(20261116)
TLx = pd.read_pickle(S3 + r"\truth.pkl"); CAL = pd.read_csv(os.path.join(HERE, "ec0h", "rise", "dm11_calendar_v1.csv")); D11 = dict(zip(CAL.rec, CAL.day))
CLEAN = {rid(int(b)): rid(int(a)) for a, b in zip(TLx.a1, TLx.b) if D11.get(rid(int(b)), np.nan) - D11.get(rid(int(a)), np.nan) == 1}
def v15_new(P, labfun, v7set, T, need_hp2):
    out = {}
    for r, pv, p0, lb in zip(P.rec, P.prev0, P.p0, P.lab0):
        if r in v7set or not bool(lb) or not isinstance(pv, str) or p0 < T: continue
        if need_hp2 and hp2(idx[r], labfun(r)) != pv: continue
        out[r] = pv
    return out
FO = pd.read_csv(os.path.join(HERE, "ev_validator", "ev_folds_v1.csv")); hid = {k: set(g.rec[g.role.isin(["val", "del"])]) for k, g in FO.groupby("fold")}
foldof = {r: k for k, g in FO[FO.role == "val"].groupby("fold") for r in g.rec}
L7 = pd.read_csv(os.path.join(env.LOCAL, "linker_v7", "cv_all400_choice_v7m.csv"), encoding="utf-8-sig")
v7set = {r for r, q, l, d in zip(L7.rec, L7.p, L7.prev_is_labelled, L7.dT_air) if q >= .8 and l and abs(d) <= 1.5}
lab_ev = lambda r: TRAIN & ~np.array([rid(i) in hid[foldof[r]] for i in range(N)])
def mirror_lab(r):
    b = idx[r]; f = FARM[b]; g = "F47" if f == "F13" else "F13"; sh = -2 if f == "F13" else 2; lo = NUM[FARM == f].min()
    s = lo + ((NUM[b] - lo) // 10) * 10; m = s + sh
    touch = ((FARM == f) & (NUM >= s - 1) & (NUM <= s + 10)); touch[(FARM == g) & (NUM >= m - 1) & (NUM <= m + 10)] = True
    return TRAIN & ~touch
PE15 = pd.read_csv(os.path.join(env.LOCAL, "linker_v10", "v15r15_ev.csv")); PM15 = pd.read_csv(os.path.join(env.LOCAL, "linker_v10", "v15r15_mirror.csv"))
PE12 = pd.read_csv(os.path.join(env.LOCAL, "linker_v10", "v12_picks_ev.csv")); PM12 = pd.read_csv(os.path.join(env.LOCAL, "linker_v10", "v12_picks_mirror.csv"))
TD8E, _, _ = new_anchors(PE12, lab_ev, v7set); TD8M, _, _ = new_anchors(PM12, mirror_lab, v7set)
ARMS = {}
ARMS["TN8"] = (v15_new(PE15, lab_ev, v7set, .8, False), v15_new(PM15, mirror_lab, v7set, .8, False))
ARMS["TNH"] = (v15_new(PE15, lab_ev, v7set, .5, True), v15_new(PM15, mirror_lab, v7set, .5, True))
ARMS["TUH"] = ({**ARMS["TNH"][0], **TD8E}, {**ARMS["TNH"][1], **TD8M})
for a, (E_, M_) in ARMS.items():
    accE = [CLEAN.get(r) == p for r, p in E_.items() if r in CLEAN]; accM = [CLEAN.get(r) == p for r, p in M_.items() if r in CLEAN]
    print(f"{a}: new anchors EV {len(E_)} (acc {np.mean(accE) if accE else float('nan'):.3f}, n {len(accE)}) | mirror {len(M_)} (acc {np.mean(accM) if accM else float('nan'):.3f}, n {len(accM)})", flush=True)
DD = pd.read_pickle(os.path.join(HERE, "ev_validator", "temp", "local_ckpt", "ev_all.pkl")); DD["rec"] = DD.row_id.str[:7]; DD = DD.rename(columns={"sub_temp": "y"}).reset_index(drop=True)
y = DD.y.values; p2 = DD.day.values >= 179
R = A7.A2.A1.frames(); L7f = L7[L7.prev_is_labelled.astype(bool) & L7.anchor_st23.notna() & (L7.steps_to_anchor == 1) & L7.rec.isin(v7set)]
alpha = .025 / 3   # Bonferroni over 3 arms
for a, (NEW_EV, NEW_M) in ARMS.items():
    ev_p2, ev_all, ev_ds = [], [], []
    for c in ["7A", "7B", "101A", "101B"]:
        base = DD[f"v22_a_{c}"].values; new = base.copy(); q18 = DD[f"v18_a_{c}"].values
        for rec, G in DD.groupby("rec"):
            if rec not in NEW_EV or NEW_EV[rec] in hid[int(G.fold.iloc[0])]: continue
            ix = G.index.values; hh = G.hour.values; q0 = q18[ix[hh == 0]][0]
            new[ix] = q18[ix] + A7.BETA[hh] * np.clip(ST[idx[NEW_EV[rec]], 23] - .36 - q0, -2, 2)
        ev_p2.append(100 * (rm((new - y)[p2]) / rm((base - y)[p2]) - 1)); ev_all.append(100 * (rm(new - y) / rm(base - y) - 1))
        ev_ds.append(pd.Series(((new - y) ** 2 - (base - y) ** 2)[p2]).groupby(DD.rec.values[p2]).sum())
    Dev, Pev, t3ev = judge("EV", None, ev_ds); hev = int((pd.concat(ev_ds, axis=1) > 1e-6).any(axis=1).sum())
    add = pd.DataFrame(dict(rec=list(NEW_M), prev=list(NEW_M.values()), p=1.0, prev_is_labelled=True, anchor_st23=[float(ST[idx[p], 23]) for p in NEW_M.values()], dT_air=0.0, steps_to_anchor=1))
    LC = pd.concat([L7f, add], ignore_index=True); out = []; dres = None
    for Lname in ("DIAG10", "G", "EL1", "EXT10", "EXT12"):
        if Lname not in R: continue
        M, cells = R[Lname]; yy = M.sub_temp.values; pp = M.day.values >= 179; kb, Ab = A7.build(M, L7f); kc, Ac = A7.build(M, LC); dp, da, ds = [], [], []
        for q in cells:
            b = A7.correct(M, q, kb, Ab, .8, "const", "meas", 2, 1.5); c = A7.correct(M, q, kc, Ac, .8, "const", "meas", 2, 1.5)
            dp.append(100 * (rm((c - yy)[pp]) / rm((b - yy)[pp]) - 1)); da.append(100 * (rm(c - yy) / rm(b - yy) - 1)); ds.append(pd.Series(((c - yy) ** 2 - (b - yy) ** 2)[pp]).groupby(kb[pp]).sum())
        out.append(dict(layout=Lname, all_pct=" ".join(f"{x:+.2f}" for x in da), p2_pct=" ".join(f"{x:+.2f}" for x in dp)))
        if Lname == "DIAG10": dres = (dp, *judge("DIAG10", None, ds), int((pd.concat(ds, axis=1) > 1e-6).any(axis=1).sum()))
    dp, Dd, Pd, t3d, hd = dres
    allbetter = all(x < 0 for x in ev_p2) and all(x < 0 for x in dp)
    strict = allbetter and Pev < alpha and Pd < alpha and t3ev < 0 and t3d < 0; small = allbetter and hev == 0 and hd == 0
    print(f"\n== {a}: EV pass-2 {[round(x, 2) for x in ev_p2]} all {[round(x, 2) for x in ev_all]} | P {Pev:.4f} drop-top3 {t3ev:+.2f} changed {(Dev.abs() > 1e-12).sum()} harmed {hev}")
    print(f"   DIAG10 pass-2 {[round(x, 2) for x in dp]} | P {Pd:.4f} drop-top3 {t3d:+.2f} changed {(Dd.abs() > 1e-12).sum()} harmed {hd}")
    pd.set_option("display.width", 200); print(pd.DataFrame(out).to_string(index=False))
    print("   top pass-2 gains EV:", (-Dev).sort_values(ascending=False).head(5).round(2).to_dict(), "| worst:", (-Dev).sort_values().head(3).round(2).to_dict())
    print("   top pass-2 gains DIAG10:", (-Dd).sort_values(ascending=False).head(5).round(2).to_dict(), "| worst:", (-Dd).sort_values().head(3).round(2).to_dict())
    print(f"TB1 {a}:", "ADOPT" + (" (strict)" if strict else " (small-sample route)") if (strict or small) else "FAIL")
# eval application (descriptive)
E15 = pd.read_csv(os.path.join(env.LOCAL, "linker_v10", "v15r15_eval.csv")); E12 = pd.read_csv(os.path.join(env.LOCAL, "linker_v10", "v12_picks_eval.csv"))
E7 = pd.read_csv(os.path.join(env.LOCAL, "linker_v7", "eval60_predecessor_v7.csv"), encoding="utf-8-sig")
v7e = set(E7[(E7.p_h0 >= .8) & E7.prev_is_labelled.astype(bool) & (E7.dT_air.abs() <= 1.5)].row)
allv = lambda r: TRAIN.copy()
ev = {"TN8": v15_new(E15, allv, v7e, .8, False), "TNH": v15_new(E15, allv, v7e, .5, True)}; td8e, _, _ = new_anchors(E12, allv, v7e); ev["TUH"] = {**ev["TNH"], **td8e}
for a, d in ev.items(): print(f"eval {a}: {len(d)} new anchors", dict(sorted(d.items())))
