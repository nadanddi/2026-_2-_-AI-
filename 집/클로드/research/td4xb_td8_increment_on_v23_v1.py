# -*- coding: utf-8 -*-
"""TD4X-(b) (2026-10-14 집 클로드; 2차 비평 5순위, (a) 결정 규칙 통과: 평가 새 기록 3 — F13_189, F13_221, F47_233).  사전 고정(커밋 후 실행).  온도.
기준 V23 = v7 앵커 + TNA8 앵커(14회차 구성).  후보 = V23 + TD8 앵커 중 V23 에 없는 기록(규정 안 v12 >= .6, 날짜 불일치 제외, 차단 짝은 HP2 일치 시).
겹치면 TNA8 짝 유지.  앵커 식 = 13회차(q18/q + BETA_h clip(A − .36 − q0, ±2)).  하네스 = TB2 와 같음(EV v15r15_ev/v12_picks_ev, DIAG10 계열 거울).
판정(TB2 와 같은 기준, 단일 안 α .025): EV 4칸·DIAG10 4칸 모두 2차 개선 & 두 검증기 2차 기록 P < .025 & 상위 3기록 제외 ΔSSE < 0;
  또는 작은 표본 길(8칸 개선 & 두 검증기 2차 손해 기록 0).  서술: 전체 행, G·EL1·EXT10·EXT12.
"""
import os, sys
import env  # noqa: F401
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, "tb2_temp_agree_any_p_v1.py"), encoding="utf-8").read(); __file__ = os.path.join(HERE, "tb2_temp_agree_any_p_v1.py")
exec(src[:src.index("DD = pd.read_pickle")])
__file__ = os.path.join(HERE, "td4xb_td8_increment_on_v23_v1.py"); rng = np.random.default_rng(20261219)
BE, BM = ARMS["TNA8"]
CE = {**{r: p for r, p in TD8E.items() if r not in BE}, **BE}; CM = {**{r: p for r, p in TD8M.items() if r not in BM}, **BM}
print(f"base TNA8 EV {len(BE)} mirror {len(BM)} | candidate EV {len(CE)} (+{len(CE) - len(BE)}) mirror {len(CM)} (+{len(CM) - len(BM)})", flush=True)
DD = pd.read_pickle(os.path.join(HERE, "ev_validator", "temp", "local_ckpt", "ev_all.pkl")); DD["rec"] = DD.row_id.str[:7]; DD = DD.rename(columns={"sub_temp": "y"}).reset_index(drop=True)
y = DD.y.values; p2 = DD.day.values >= 179
def ev_apply(c, NEW):
    new = DD[f"v22_a_{c}"].values.copy(); q18 = DD[f"v18_a_{c}"].values
    for rec, G in DD.groupby("rec"):
        if rec not in NEW or NEW[rec] in hid[int(G.fold.iloc[0])]: continue
        ix = G.index.values; hh = G.hour.values; q0 = q18[ix[hh == 0]][0]
        new[ix] = q18[ix] + A7.BETA[hh] * np.clip(ST[idx[NEW[rec]], 23] - .36 - q0, -2, 2)
    return new
ev_p2, ev_all, ev_ds = [], [], []
for c in ["7A", "7B", "101A", "101B"]:
    b = ev_apply(c, BE); n = ev_apply(c, CE)
    ev_p2.append(100 * (rm((n - y)[p2]) / rm((b - y)[p2]) - 1)); ev_all.append(100 * (rm(n - y) / rm(b - y) - 1))
    ev_ds.append(pd.Series(((n - y) ** 2 - (b - y) ** 2)[p2]).groupby(DD.rec.values[p2]).sum())
Dev, Pev, t3ev = judge("EV", None, ev_ds); hev = int((pd.concat(ev_ds, axis=1) > 1e-6).any(axis=1).sum())
R = A7.A2.A1.frames(); L7f = L7[L7.prev_is_labelled.astype(bool) & L7.anchor_st23.notna() & (L7.steps_to_anchor == 1) & L7.rec.isin(v7set)]
mk = lambda NEW: pd.concat([L7f, pd.DataFrame(dict(rec=list(NEW), prev=list(NEW.values()), p=1.0, prev_is_labelled=True, anchor_st23=[float(ST[idx[p], 23]) for p in NEW.values()], dT_air=0.0, steps_to_anchor=1))], ignore_index=True)
LB, LCc = mk(BM), mk(CM); out = []; dres = None
for Lname in ("DIAG10", "G", "EL1", "EXT10", "EXT12"):
    if Lname not in R: continue
    M, cells = R[Lname]; yy = M.sub_temp.values; pp = M.day.values >= 179; kb, Ab = A7.build(M, LB); kc, Ac = A7.build(M, LCc); dp, da, ds = [], [], []
    for q in cells:
        b = A7.correct(M, q, kb, Ab, .8, "const", "meas", 2, 1.5); c = A7.correct(M, q, kc, Ac, .8, "const", "meas", 2, 1.5)
        dp.append(100 * (rm((c - yy)[pp]) / rm((b - yy)[pp]) - 1)); da.append(100 * (rm(c - yy) / rm(b - yy) - 1)); ds.append(pd.Series(((c - yy) ** 2 - (b - yy) ** 2)[pp]).groupby(kb[pp]).sum())
    out.append(dict(layout=Lname, all_pct=" ".join(f"{x:+.2f}" for x in da), p2_pct=" ".join(f"{x:+.2f}" for x in dp)))
    if Lname == "DIAG10": dres = (dp, *judge("DIAG10", None, ds), int((pd.concat(ds, axis=1) > 1e-6).any(axis=1).sum()))
dp, Dd, Pd, t3d, hd = dres
allbetter = all(x < 0 for x in ev_p2) and all(x < 0 for x in dp)
strict = allbetter and Pev < .025 and Pd < .025 and t3ev < 0 and t3d < 0; small = allbetter and hev == 0 and hd == 0
print(f"\nEV pass-2 {[round(x, 2) for x in ev_p2]} all {[round(x, 2) for x in ev_all]} | P {Pev:.4f} drop-top3 {t3ev:+.2f} changed {(Dev.abs() > 1e-12).sum()} harmed {hev}")
print(f"DIAG10 pass-2 {[round(x, 2) for x in dp]} | P {Pd:.4f} drop-top3 {t3d:+.2f} changed {(Dd.abs() > 1e-12).sum()} harmed {hd}")
pd.set_option("display.width", 200); print(pd.DataFrame(out).to_string(index=False))
print("changed pass-2 EV:", Dev[Dev.abs() > 1e-12].round(2).to_dict()); print("changed pass-2 DIAG10:", Dd[Dd.abs() > 1e-12].round(2).to_dict())
print("TD4X-(b):", "ADOPT" + (" (strict)" if strict else " (small-sample route)") if (strict or small) else "FAIL")
