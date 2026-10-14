# -*- coding: utf-8 -*-
"""TY4 (2026-10-14 집 클로드; 최종 비평 6.884 필수 확인).  사전 고정(커밋 후 실행).  온도.
비평: TY3 '새 배치'는 TY2 와 같은 30기록(기록별 ΔSSE 상관 .81~.98) → 독립 증거 아님; EV 1차는 Y3 에서 +.25~+.30% 악화(β 를 1차에서 맞췄는데도);
  평가 노출 85% 가 검증 범위 2배 넘는 두 기록(F47_203·207).  F13 성분 기여 0.
TY4 = 방향 뒤집기(새 증거): 모듈을 F47 전용 + 이동 상한으로 고정하고, β 를 2차 cert 1일 쌍(대상 F47, 번호 >= 179; EV V23 잔차)에서 재고, **1차 기록에서 판정**.
  모듈: 대상 F47 기록(앵커 짝 a = v7 ∪ TNA8, 그 하네스 V23 의 a 잔차)에 p_h += clip(β_h · r_a[h], −.45, +.45)  (.45 = TY3 배치에서 본 오후 평균 이동 최대 — 검증 범위 밖 외삽 금지, 결과로 안 고름).
  β_h = 원점 회귀 기울기, 0 아래 0.  2차 F47 쌍이 8 미만이면 2차 전체 쌍으로(미리 정함).
하네스: EV(4칸, V23 = v22 + TNA8 EV) 와 DIAG10(거울, 4칸, V23 = v7 + TNA8 거울).  F13 대상은 손대지 않음.
판정(단일 안 α .025): 두 하네스 모두 1차 행 4칸 개선 & 각 하네스 1차 기록 부트스트랩 P(악화) < .025 & 상위 3기록 제외 ΔSSE < 0.
  통과 → Y3 계열(F47 전용·상한) 근거 성립.  실패 → TY 계열 보류 종료(2차 30기록 효과로 남김).
서술: 2차 행(표본 안이라 참고), 같은 폴드/다른 폴드 쌍 분해(EV), 평가 적용량(상한 후).
"""
import os, sys
import env  # noqa: F401
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, "tb2_temp_agree_any_p_v1.py"), encoding="utf-8").read(); __file__ = os.path.join(HERE, "tb2_temp_agree_any_p_v1.py")
exec(src[:src.index("DD = pd.read_pickle")])
__file__ = os.path.join(HERE, "ty4_reverse_direction_check_v1.py"); rng = np.random.default_rng(20261222); CAP = .45
TNE, TNM = ARMS["TNA8"]; V7P = {r: p for r, p in zip(L7.rec, L7.prev) if r in v7set}
DD = pd.read_pickle(os.path.join(HERE, "ev_validator", "temp", "local_ckpt", "ev_all.pkl")); DD["rec"] = DD.row_id.str[:7]; DD = DD.rename(columns={"sub_temp": "y"}).reset_index(drop=True)
CELLS = ["7A", "7B", "101A", "101B"]; y = DD.y.values; recs = DD.rec.values; hrs = DD.hour.values; fold_of = DD.groupby("rec").fold.first().to_dict()
def ev_cell(c):
    new = DD[f"v22_a_{c}"].values.copy(); q18 = DD[f"v18_a_{c}"].values
    for rec, G in DD.groupby("rec"):
        if rec not in TNE or TNE[rec] in hid[int(G.fold.iloc[0])]: continue
        ix = G.index.values; hh = G.hour.values; q0 = q18[ix[hh == 0]][0]
        new[ix] = q18[ix] + A7.BETA[hh] * np.clip(ST[idx[TNE[rec]], 23] - .36 - q0, -2, 2)
    return new
BASE_EV = {c: ev_cell(c) for c in CELLS}
RP = pd.Series(y - np.mean([BASE_EV[c] for c in CELLS], 0)).groupby([recs, hrs]).mean().unstack()
PR = [(rid(int(a)), rid(int(b))) for a, b, c in zip(TLx.a1, TLx.b, TLx.cert) if c]; PR = [(a, b) for a, b in PR if a in RP.index and b in RP.index]
P2F = [(a, b) for a, b in PR if int(b[4:]) >= 179 and b[:3] == "F47"]; P2A = [(a, b) for a, b in PR if int(b[4:]) >= 179]
PB = P2F if len(P2F) >= 8 else P2A
A_ = RP.loc[[a for a, _ in PB]].values; B_ = RP.loc[[b for _, b in PB]].values
BETA = np.array([max(0.0, np.nansum(A_[:, h] * B_[:, h]) / np.nansum(np.where(np.isfinite(B_[:, h]), A_[:, h], np.nan) ** 2)) for h in range(24)])
print(f"pass-2 F47 pairs {len(P2F)} (all pass-2 {len(P2A)}) -> using {len(PB)} | beta:", np.round(BETA, 3).tolist(), flush=True)
def apply(pred, yv, rr, hh, pairs, hidden_of):
    res = pd.Series(yv - pred).groupby([rr, hh]).mean(); have = set(res.index.get_level_values(0)); out = pred.copy(); n = 0
    for rec in np.unique(rr):
        a = pairs.get(rec)
        if rec[:3] != "F47" or a is None or a in hidden_of(rec) or a not in have: continue
        ix = np.where(rr == rec)[0]; out[ix] = pred[ix] + np.clip(BETA[hh[ix]] * np.nan_to_num(res.loc[a].reindex(hh[ix]).values), -CAP, CAP); n += 1
    return out, n
def judge_p1(ds):
    D = pd.concat(ds, axis=1).fillna(0).mean(axis=1); v = D.values; P = float((v[rng.integers(0, len(v), (20000, len(v)))].sum(1) >= 0).mean())
    return D, P, D.sort_values().iloc[3:].sum()
p1 = DD.day.values < 179; p2 = ~p1; out = {}
PE = {**V7P, **TNE}; c1, c2, ds = [], [], []
for c in CELLS:
    b = BASE_EV[c]; n_, k = apply(b, y, recs, hrs, PE, lambda r: hid[fold_of[r]])
    c1.append(100 * (rm((n_ - y)[p1]) / rm((b - y)[p1]) - 1)); c2.append(100 * (rm((n_ - y)[p2]) / rm((b - y)[p2]) - 1))
    ds.append(pd.Series(((n_ - y) ** 2 - (b - y) ** 2)[p1]).groupby(recs[p1]).sum())
Dv, Pv, t3v = judge_p1(ds); out["EV"] = (c1, c2, Pv, t3v, Dv, k)
R = A7.A2.A1.frames(); M, cells = R["DIAG10"]; yy = M.sub_temp.values; mrec = np.array([f"{f}_{int(d):03d}" for f, d in zip(M.farm, M.day)]); mh = M.hour.values.astype(int)
L7f = L7[L7.prev_is_labelled.astype(bool) & L7.anchor_st23.notna() & (L7.steps_to_anchor == 1) & L7.rec.isin(v7set)]
LC = pd.concat([L7f, pd.DataFrame(dict(rec=list(TNM), prev=list(TNM.values()), p=1.0, prev_is_labelled=True, anchor_st23=[float(ST[idx[p], 23]) for p in TNM.values()], dT_air=0.0, steps_to_anchor=1))], ignore_index=True)
kc, Ac = A7.build(M, LC); q1 = M.day.values < 179; c1, c2, ds = [], [], []
for q in cells:
    b = A7.correct(M, q, kc, Ac, .8, "const", "meas", 2, 1.5); n_, k = apply(b, yy, mrec, mh, {**V7P, **TNM}, lambda r: set())
    c1.append(100 * (rm((n_ - yy)[q1]) / rm((b - yy)[q1]) - 1)); c2.append(100 * (rm((n_ - yy)[~q1]) / rm((b - yy)[~q1]) - 1))
    ds.append(pd.Series(((n_ - yy) ** 2 - (b - yy) ** 2)[q1]).groupby(mrec[q1]).sum())
Dd, Pd, t3d = judge_p1(ds); out["DIAG10"] = (c1, c2, Pd, t3d, Dd, k)
ok = True
for nm, (c1, c2, P, t3, D, k) in out.items():
    good = all(x < 0 for x in c1) and P < .025 and t3 < 0; ok &= good
    print(f"\n{nm}: corrected F47 records {k} | pass-1 cells {[round(x, 2) for x in c1]} | P {P:.4f} drop-top3 {t3:+.2f} changed {(D.abs() > 1e-12).sum()} harmed {(D > 1e-6).sum()} -> {'ok' if good else 'not'}")
    print(f"   pass-2 (in-sample beta, reference) {[round(x, 2) for x in c2]}")
    print("   top gains:", (-D).sort_values(ascending=False).head(5).round(2).to_dict(), "| worst:", (-D).sort_values().head(5).round(2).to_dict())
print("\nTY4 reverse-direction:", "PASS" if ok else "FAIL")
# eval exposure after cap (descriptive)
E15 = pd.read_csv(os.path.join(env.LOCAL, "linker_v10", "v15r15_eval.csv")); E7 = pd.read_csv(os.path.join(env.LOCAL, "linker_v7", "eval60_predecessor_v7.csv"), encoding="utf-8-sig")
v7e = E7[(E7.p_h0 >= .8) & E7.prev_is_labelled.astype(bool) & (E7.dT_air.abs() <= 1.5)]; v7map = dict(zip(v7e.row, v7e.prev_h0))
allv = lambda r: TRAIN.copy(); EP = {**v7map, **v15_new(E15, allv, set(v7map), 0.0, True), **v15_new(E15, allv, set(v7map), .8, False)}
sh = {r: np.clip(BETA * np.nan_to_num(RP.loc[a].reindex(range(24)).values), -CAP, CAP) for r, a in EP.items() if r[:3] == "F47" and a in RP.index}
print("eval F47 anchored", len(sh), "| day-mean shifts:", {r: round(float(v.mean()), 3) for r, v in sorted(sh.items()) if abs(v.mean()) > .02}, f"| row RMS over 1440 {np.sqrt(sum(np.sum(v ** 2) for v in sh.values()) / 1440):.4f}")
