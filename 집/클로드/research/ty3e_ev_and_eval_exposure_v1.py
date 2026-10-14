# -*- coding: utf-8 -*-
"""TY3E (2026-10-14 집 클로드).  서술(판정 아님).  TY3 Y3(농장별 β, 6.882 새 배치 ADOPT)의
(1) EV 하네스 결과 — EV 는 TY2 판정에 쓰였고 Y3 설계 동기(F13 악화)를 준 검증기라 참고만;
(2) 평가 적용량 — 평가 앵커 짝(v7 평가 + TNA8 평가, 14회차와 같음)의 전날 a 잔차 = y_a − p_a(p_a = EV 저장 v22 4칸 평균, a 의 검증 예측),
    평가 기록 b 의 시각 h 이동 = β_farm(b)[h] · r_a[h].  기록별 하루 평균·최대 이동, 행 RMS.
"""
import os, sys
import env  # noqa: F401
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, "ty2_yesterday_same_hour_anchor_v1.py"), encoding="utf-8").read(); __file__ = os.path.join(HERE, "ty2_yesterday_same_hour_anchor_v1.py")
exec(src[:src.index("V7P = dict(")])
__file__ = os.path.join(HERE, "ty3e_ev_and_eval_exposure_v1.py")
def beta(pairs):
    A_ = RP.loc[[a for a, _ in pairs]].values; B_ = RP.loc[[b for _, b in pairs]].values
    return np.array([max(0.0, np.nansum(A_[:, h] * B_[:, h]) / np.nansum(np.where(np.isfinite(B_[:, h]), A_[:, h], np.nan) ** 2)) for h in range(24)])
BF = {f: beta([(a, b) for a, b in PR1 if b[:3] == f]) for f in ("F13", "F47")}
V7P = {r: p for r, p in zip(L7.rec, L7.prev) if r in v7set}; PAIRS_E = {**V7P, **TNE}
fold_of = DD.groupby("rec").fold.first().to_dict(); recs = DD.rec.values; hrs = DD.hour.values
def ev_cell(c, extra):
    new = DD[f"v22_a_{c}"].values.copy(); q18 = DD[f"v18_a_{c}"].values
    for rec, G in DD.groupby("rec"):
        if rec not in extra or extra[rec] in hid[int(G.fold.iloc[0])]: continue
        ix = G.index.values; hh = G.hour.values; q0 = q18[ix[hh == 0]][0]
        new[ix] = q18[ix] + A7.BETA[hh] * np.clip(ST[idx[extra[rec]], 23] - .36 - q0, -2, 2)
    return new
rows = []; f47 = DD.farm.values == "F47"
for c in CELLS:
    b = ev_cell(c, TNE); res = pd.Series(y - b).groupby([recs, hrs]).mean(); n_ = b.copy()
    for rec in np.unique(recs):
        a = PAIRS_E.get(rec)
        if a is None or a in hid[fold_of[rec]] or a not in res.index.get_level_values(0): continue
        ix = np.where(recs == rec)[0]; n_[ix] = b[ix] + BF[rec[:3]][hrs[ix]] * np.nan_to_num(res.loc[a].reindex(hrs[ix]).values)
    f = lambda m: 100 * (rm((n_ - y)[m]) / rm((b - y)[m]) - 1)
    rows.append(dict(cell=c, all=f(np.ones(len(y), bool)), p2=f(p2), F47=f(f47), F13=f(~f47), p2_F47=f(p2 & f47), p2_F13=f(p2 & ~f47)))
print("EV (descriptive, Y3 farm beta):"); print(pd.DataFrame(rows).round(2).to_string(index=False))
# eval exposure
E15 = pd.read_csv(os.path.join(env.LOCAL, "linker_v10", "v15r15_eval.csv")); E7 = pd.read_csv(os.path.join(env.LOCAL, "linker_v7", "eval60_predecessor_v7.csv"), encoding="utf-8-sig")
v7e = E7[(E7.p_h0 >= .8) & E7.prev_is_labelled.astype(bool) & (E7.dT_air.abs() <= 1.5)]; v7map = dict(zip(v7e.row, v7e.prev_h0))
allv = lambda r: TRAIN.copy(); tna8e = {**v15_new(E15, allv, set(v7map), 0.0, True), **v15_new(E15, allv, set(v7map), .8, False)}
EP = {**v7map, **tna8e}; RA = RP   # pass-through: RP = EV v22 residual of each labelled record (its own validation prediction)
out = []
for r, a in sorted(EP.items()):
    if a not in RA.index: out.append(dict(rec=r, prev=a, note="prev residual missing")); continue
    sh = BF[r[:3]] * np.nan_to_num(RA.loc[a].reindex(range(24)).values)
    out.append(dict(rec=r, prev=a, day_mean=float(sh.mean()), max_abs=float(np.abs(sh).max()), pm_12_18=float(sh[12:19].mean())))
O = pd.DataFrame(out); pd.set_option("display.width", 200); print("\neval exposure:"); print(O.round(3).to_string(index=False))
sq = np.concatenate([(BF[r[:3]] * np.nan_to_num(RA.loc[a].reindex(range(24)).values)) for r, a in EP.items() if a in RA.index])
print(f"eval anchored records {len(EP)} | row RMS shift over 1440 rows {np.sqrt(np.sum(sq ** 2) / 1440):.4f} | records with |day mean| > .1: {int((O.day_mean.abs() > .1).sum())}")
O.to_csv(os.path.join(HERE, "ty3e_eval_exposure_v1.csv"), index=False)
