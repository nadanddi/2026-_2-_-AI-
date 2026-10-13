# -*- coding: utf-8 -*-
"""TQ1 (2026-10-14 집 클로드; 사용자: 'LK2 q 로 앵커 짝 선택해봐').  사전 고정(커밋 후 실행).  온도.
LK2(6.828): 짝 정답 확률 q 가 p0 보다 정확(EV AUC .907 vs .806, q ≥ .8 110짝 .936 vs p0 ≥ .8 69짝 .899).
기준 = 14회차형: v7 앵커 + TNA8(v15R 짝이 정답 기록이고 (p0 ≥ .8 또는 HP2 일치)), 측정 감쇠 공식(13차).
후보 TQ8 = 기준 + 새 앵커: v15R 1순위 짝이 정답 기록(lab0, 그 폴드 숨김 아님)이고 q ≥ .8, v7·TNA8 앵커 없는 기록.  새 앵커식은 TNA8 과 같음(dT 차단 없음).
q 교차 적합(DS9 와 같음): EV 짝 = 거울 표로 맞춘 LK2 모형의 q, 거울 짝(DIAG10 계열) = EV 표로 맞춘 같은 모형의 q.
판정(TB1 과 같은 기준, 안 1개 α .025): EV 4칸·DIAG10 4칸 모두 2차 개선 AND 각 검증기 2차 기록 부트스트랩 P < .025 AND 상위 3기록 제외 ΔSSE 합 < 0;
  또는 작은 표본 길: 8칸 개선 & 두 검증기 2차 손해 기록 0.  서술: 전체 행, G·EL1·EXT10·EXT12, 새 앵커 수·짝 정확도(진단), 평가 적용 수.
"""
import os, sys
import env  # noqa: F401
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, "tb1_temp_null_agree_anchor_v1.py"), encoding="utf-8").read(); __file__ = os.path.join(HERE, "tb1_temp_null_agree_anchor_v1.py")
exec(src[:src.index("PE15 = ")])
__file__ = os.path.join(HERE, "tq1_temp_qlink_anchor_v1.py"); rng = np.random.default_rng(20261204)
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
_ET = pd.read_csv(os.path.join(HERE, "lk2_ev_hops_q_v1.csv")); _MT = pd.read_csv(os.path.join(HERE, "lk2_mirror_hops_q_v1.csv"))
_F = ["lp0", "margin", "p_null", "cross", "dTin", "dTout", "date_yes", "date_no", "same_id_earlier", "hp2_agree"]
_fit = _ET[_ET.ok.notna()]; _sc = StandardScaler().fit(_fit[_F]); _LR = LogisticRegression(C=1.0, max_iter=1000).fit(_sc.transform(_fit[_F]), _fit.ok.astype(int))
_MT["qx"] = _LR.predict_proba(_sc.transform(_MT[_F]))[:, 1]; QM = dict(zip(_MT.rec, _MT.qx)); QE = dict(zip(_ET.rec, _ET.q))
PE = pd.read_csv(os.path.join(env.LOCAL, "linker_v10", "v15r15_ev.csv")); PMx = pd.read_csv(os.path.join(env.LOCAL, "linker_v10", "v15r15_mirror.csv"))
TNA8E = {**v15_new(PE, lab_ev, v7set, 0.0, True), **v15_new(PE, lab_ev, v7set, .8, False)}
TNA8M = {**v15_new(PMx, mirror_lab, v7set, 0.0, True), **v15_new(PMx, mirror_lab, v7set, .8, False)}
def qnew(P, Q, have):
    return {r: pv for r, pv, lb in zip(P.rec, P.prev0, P.lab0) if isinstance(pv, str) and bool(lb) and Q.get(r, 0) >= .8 and r not in v7set and r not in have}
QNE, QNM = qnew(PE, QE, TNA8E), qnew(PMx, QM, TNA8M)
for nm, d in (("EV", QNE), ("mirror", QNM)):
    acc = [CLEAN.get(r) == p for r, p in d.items() if r in CLEAN]; print(f"TQ8 new anchors {nm}: {len(d)} (diag acc {np.mean(acc) if acc else float('nan'):.3f}, n {len(acc)})", flush=True)
DD = pd.read_pickle(os.path.join(HERE, "ev_validator", "temp", "local_ckpt", "ev_all.pkl")); DD["rec"] = DD.row_id.str[:7]; DD = DD.rename(columns={"sub_temp": "y"}).reset_index(drop=True)
y = DD.y.values; p2 = DD.day.values >= 179
def apply_ev(base, q18, links):
    new = base.copy()
    for rec, G in DD.groupby("rec"):
        if rec not in links or links[rec] in hid[int(G.fold.iloc[0])]: continue
        ix = G.index.values; hh = G.hour.values; q0 = q18[ix[hh == 0]][0]
        new[ix] = q18[ix] + A7.BETA[hh] * np.clip(ST[idx[links[rec]], 23] - .36 - q0, -2, 2)
    return new
ev_p2, ev_all, ev_ds = [], [], []
for c in ["7A", "7B", "101A", "101B"]:
    q18 = DD[f"v18_a_{c}"].values; base = apply_ev(DD[f"v22_a_{c}"].values, q18, TNA8E); new = apply_ev(base, q18, QNE)
    ev_p2.append(100 * (rm((new - y)[p2]) / rm((base - y)[p2]) - 1)); ev_all.append(100 * (rm(new - y) / rm(base - y) - 1))
    ev_ds.append(pd.Series(((new - y) ** 2 - (base - y) ** 2)[p2]).groupby(DD.rec.values[p2]).sum())
Dev, Pev, t3ev = judge("EV", None, ev_ds); hev = int((pd.concat(ev_ds, axis=1) > 1e-6).any(axis=1).sum())
R = A7.A2.A1.frames(); L7f = L7[L7.prev_is_labelled.astype(bool) & L7.anchor_st23.notna() & (L7.steps_to_anchor == 1) & L7.rec.isin(v7set)]
mk = lambda d: pd.DataFrame(dict(rec=list(d), prev=list(d.values()), p=1.0, prev_is_labelled=True, anchor_st23=[float(ST[idx[p], 23]) for p in d.values()], dT_air=0.0, steps_to_anchor=1))
LCb = pd.concat([L7f, mk(TNA8M)], ignore_index=True); LCn = pd.concat([LCb, mk(QNM)], ignore_index=True); out = []; dres = None
for Lname in ("DIAG10", "G", "EL1", "EXT10", "EXT12"):
    if Lname not in R: continue
    M, cells = R[Lname]; yy = M.sub_temp.values; pp = M.day.values >= 179; kb, Ab = A7.build(M, LCb); kc, Ac = A7.build(M, LCn); dp, da, ds = [], [], []
    for qc in cells:
        b = A7.correct(M, qc, kb, Ab, .8, "const", "meas", 2, 1.5); c = A7.correct(M, qc, kc, Ac, .8, "const", "meas", 2, 1.5)
        dp.append(100 * (rm((c - yy)[pp]) / rm((b - yy)[pp]) - 1)); da.append(100 * (rm(c - yy) / rm(b - yy) - 1)); ds.append(pd.Series(((c - yy) ** 2 - (b - yy) ** 2)[pp]).groupby(kb[pp]).sum())
    out.append(dict(layout=Lname, all_pct=" ".join(f"{x:+.2f}" for x in da), p2_pct=" ".join(f"{x:+.2f}" for x in dp)))
    if Lname == "DIAG10": dres = (dp, *judge("DIAG10", None, ds), int((pd.concat(ds, axis=1) > 1e-6).any(axis=1).sum()))
dp, Dd, Pd, t3d, hd = dres; alpha = .025
allbetter = all(x < 0 for x in ev_p2) and all(x < 0 for x in dp)
strict = allbetter and Pev < alpha and Pd < alpha and t3ev < 0 and t3d < 0; small = allbetter and hev == 0 and hd == 0
print(f"\n== TQ8: EV pass-2 {[round(x, 2) for x in ev_p2]} all {[round(x, 2) for x in ev_all]} | P {Pev:.4f} drop-top3 {t3ev:+.2f} changed {(Dev.abs() > 1e-12).sum()} harmed {hev}")
print(f"   DIAG10 pass-2 {[round(x, 2) for x in dp]} | P {Pd:.4f} drop-top3 {t3d:+.2f} changed {(Dd.abs() > 1e-12).sum()} harmed {hd}")
pd.set_option("display.width", 200); print(pd.DataFrame(out).to_string(index=False))
print("   pass-2 record dSSE EV (neg=better):", Dev[Dev.abs() > 1e-12].sort_values().round(2).to_dict())
print("   pass-2 record dSSE DIAG10:", Dd[Dd.abs() > 1e-12].sort_values().round(2).to_dict())
print("TQ1 TQ8:", "ADOPT" + (" (strict)" if strict else " (small-sample route)") if (strict or small) else "FAIL")
# eval application count (descriptive; q for eval hops not computed here)
