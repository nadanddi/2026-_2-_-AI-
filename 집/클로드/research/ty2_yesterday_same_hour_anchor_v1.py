# -*- coding: utf-8 -*-
"""TY2 (2026-10-14 집 클로드; 사용자: '온도쪽 가설 해봐').  사전 고정(커밋 후 실행).  온도.  과거 정보만(전날 = 정답 있는 학습 기록).
TY1(서술, 이 파일 전): 최종 예측 잔차의 '전날 같은 시각' 지속 — 오후 12~17시 상관 .21~.26(전날 23시 잔차 통제 후에도 .21), 2차 .50~.54, F47 .40, F13 ≈ 0;
  밤·새벽은 앵커가 이미 씀(부분 상관 ≈ 0).  = 동마다 배지가 낮에 데워지는 정도의 차이가 다음날에도 이어짐.
규칙: 13회차 앵커 식이 붙는 기록(전날 짝 a = v7 앵커 ∪ TNA8, 그 기하에서 보이는 정답 기록)에 대해 모든 시각 h:
  p_h += β_h · r_a[h],  r_a = 전날 기록 a 의 정답 − 같은 하네스의 a 자신 최종 예측(검증 때 예측).
  β_h = 1차 cert 1일 쌍(대상 번호 < 179)의 EV 잔차(v22 4칸 평균)로 시각별 원점 회귀 기울기, 0 아래는 0 (24개 숫자, 2차 안 씀; 시각 창 고르지 않음).
하네스: EV(ev_all.pkl v22 칸 + TNA8 EV 앵커; 앵커 짝 = v7 목록(폴드 숨김 빠짐) ∪ TNA8 EV) / DIAG10 계열(거울, tb2 와 같은 frames·셀, 앵커 v7 + TNA8 거울).
  기준 = 같은 하네스의 v23(v7 + TNA8) 최종.  [한계: 전날 a 의 예측은 a 가 검증이던 폴드 모델 — 그 모델이 대상 정답을 봤을 수 있음(2차 노출, RS3 와 같은 종류)]
판정(TB2 기준 + EXT10 보호, 단일 안 α .025): EV 4칸·DIAG10 4칸 모두 2차 개선 & (두 검증기 2차 기록 P < .025 & 상위 3기록 제외 ΔSSE < 0 또는 두 검증기 2차 손해 기록 0)
  & EXT10 전체 행 4칸 모두 악화 아님.  서술: 전체 행, G·EL1·EXT12, F13/F47, 시각대별.
"""
import os, sys
import env  # noqa: F401
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, "tb2_temp_agree_any_p_v1.py"), encoding="utf-8").read(); __file__ = os.path.join(HERE, "tb2_temp_agree_any_p_v1.py")
exec(src[:src.index("DD = pd.read_pickle")])
__file__ = os.path.join(HERE, "ty2_yesterday_same_hour_anchor_v1.py"); rng = np.random.default_rng(20261220)
TNE, TNM = ARMS["TNA8"]
DD = pd.read_pickle(os.path.join(HERE, "ev_validator", "temp", "local_ckpt", "ev_all.pkl")); DD["rec"] = DD.row_id.str[:7]; DD = DD.rename(columns={"sub_temp": "y"}).reset_index(drop=True)
CELLS = ["7A", "7B", "101A", "101B"]; y = DD.y.values; p2 = DD.day.values >= 179
# beta from pass-1 cert pairs (EV v22 residuals)
Rv = (DD.y - DD[[f"v22_a_{c}" for c in CELLS]].mean(axis=1)).values; RP = pd.Series(Rv).groupby([DD.rec.values, DD.hour.values]).mean().unstack()
PR = [(rid(int(a)), rid(int(b))) for a, b, c in zip(TLx.a1, TLx.b, TLx.cert) if c]; PR1 = [(a, b) for a, b in PR if a in RP.index and b in RP.index and int(b[4:]) < 179]
A_ = RP.loc[[a for a, _ in PR1]].values; B_ = RP.loc[[b for _, b in PR1]].values
BETA_Y = np.array([max(0.0, np.nansum(A_[:, h] * B_[:, h]) / np.nansum(np.where(np.isfinite(B_[:, h]), A_[:, h], np.nan) ** 2)) for h in range(24)])
print(f"pass-1 pairs {len(PR1)} | beta_Y by hour:", np.round(BETA_Y, 3).tolist(), flush=True)
V7P = dict(zip(L7.rec, L7.prev)); V7P = {r: V7P[r] for r in v7set}
def ev_cell(c, anchors_extra):
    new = DD[f"v22_a_{c}"].values.copy(); q18 = DD[f"v18_a_{c}"].values
    for rec, G in DD.groupby("rec"):
        if rec not in anchors_extra or anchors_extra[rec] in hid[int(G.fold.iloc[0])]: continue
        ix = G.index.values; hh = G.hour.values; q0 = q18[ix[hh == 0]][0]
        new[ix] = q18[ix] + A7.BETA[hh] * np.clip(ST[idx[anchors_extra[rec]], 23] - .36 - q0, -2, 2)
    return new
def add_y(pred, yv, recs, hours, pairs, hidden_of):
    res = pd.Series(yv - pred).groupby([recs, hours]).mean(); out = pred.copy(); n = 0
    for rec in np.unique(recs):
        a = pairs.get(rec)
        if a is None or a in hidden_of(rec) or a not in res.index.get_level_values(0): continue
        ix = np.where(recs == rec)[0]; ra = res.loc[a]
        out[ix] = pred[ix] + BETA_Y[hours[ix]] * np.nan_to_num(ra.reindex(hours[ix]).values); n += 1
    return out, n
fold_of = DD.groupby("rec").fold.first().to_dict(); recs = DD.rec.values; hrs = DD.hour.values
PAIRS_E = {**V7P, **TNE}
ev_p2, ev_all, ev_ds, ev_f = [], [], [], []
for c in CELLS:
    b = ev_cell(c, TNE); n_, k = add_y(b, y, recs, hrs, PAIRS_E, lambda r: hid[fold_of[r]])
    ev_p2.append(100 * (rm((n_ - y)[p2]) / rm((b - y)[p2]) - 1)); ev_all.append(100 * (rm(n_ - y) / rm(b - y) - 1))
    ev_ds.append(pd.Series(((n_ - y) ** 2 - (b - y) ** 2)[p2]).groupby(recs[p2]).sum())
    f47 = DD.farm.values == "F47"; ev_f.append((100 * (rm((n_ - y)[f47]) / rm((b - y)[f47]) - 1), 100 * (rm((n_ - y)[~f47]) / rm((b - y)[~f47]) - 1)))
print(f"EV records corrected: {k}", flush=True)
Dev, Pev, t3ev = judge("EV", None, ev_ds); hev = int((pd.concat(ev_ds, axis=1) > 1e-6).any(axis=1).sum())
R = A7.A2.A1.frames(); L7f = L7[L7.prev_is_labelled.astype(bool) & L7.anchor_st23.notna() & (L7.steps_to_anchor == 1) & L7.rec.isin(v7set)]
LC = pd.concat([L7f, pd.DataFrame(dict(rec=list(TNM), prev=list(TNM.values()), p=1.0, prev_is_labelled=True, anchor_st23=[float(ST[idx[p], 23]) for p in TNM.values()], dT_air=0.0, steps_to_anchor=1))], ignore_index=True)
PAIRS_M = {**V7P, **TNM}; out = []; dres = None; ext10_ok = True
for Lname in ("DIAG10", "G", "EL1", "EXT10", "EXT12"):
    if Lname not in R: continue
    M, cells = R[Lname]; yy = M.sub_temp.values; pp = M.day.values >= 179; kc, Ac = A7.build(M, LC)
    mrec = np.array([f"{f}_{int(d):03d}" for f, d in zip(M.farm, M.day)]); mh = M.hour.values.astype(int); dp, da, ds = [], [], []
    hidM = lambda r: set()   # mirror harness: prev visibility already enforced by anchor list (mirror_lab picks); prev prediction = its own val prediction
    for q in cells:
        b = A7.correct(M, q, kc, Ac, .8, "const", "meas", 2, 1.5)
        c, km = add_y(b, yy, mrec, mh, PAIRS_M, hidM)
        dp.append(100 * (rm((c - yy)[pp]) / rm((b - yy)[pp]) - 1)); da.append(100 * (rm(c - yy) / rm(b - yy) - 1)); ds.append(pd.Series(((c - yy) ** 2 - (b - yy) ** 2)[pp]).groupby(mrec[pp]).sum())
    out.append(dict(layout=Lname, corrected=km, all_pct=" ".join(f"{x:+.2f}" for x in da), p2_pct=" ".join(f"{x:+.2f}" for x in dp)))
    if Lname == "DIAG10": dres = (dp, *judge("DIAG10", None, ds), int((pd.concat(ds, axis=1) > 1e-6).any(axis=1).sum()))
    if Lname == "EXT10": ext10_ok = all(x <= 0 for x in da)
dp, Dd, Pd, t3d, hd = dres
allbetter = all(x < 0 for x in ev_p2) and all(x < 0 for x in dp)
strict = allbetter and Pev < .025 and Pd < .025 and t3ev < 0 and t3d < 0; small = allbetter and hev == 0 and hd == 0
print(f"\nEV pass-2 {[round(x, 2) for x in ev_p2]} all {[round(x, 2) for x in ev_all]} | F47/F13 all {[(round(a, 2), round(b_, 2)) for a, b_ in ev_f]} | P {Pev:.4f} drop-top3 {t3ev:+.2f} changed {(Dev.abs() > 1e-12).sum()} harmed {hev}")
print(f"DIAG10 pass-2 {[round(x, 2) for x in dp]} | P {Pd:.4f} drop-top3 {t3d:+.2f} changed {(Dd.abs() > 1e-12).sum()} harmed {hd}")
pd.set_option("display.width", 200); print(pd.DataFrame(out).to_string(index=False))
print("top pass-2 EV:", (-Dev).sort_values(ascending=False).head(5).round(2).to_dict(), "| worst:", (-Dev).sort_values().head(4).round(2).to_dict())
print("top pass-2 DIAG10:", (-Dd).sort_values(ascending=False).head(5).round(2).to_dict(), "| worst:", (-Dd).sort_values().head(4).round(2).to_dict())
ok = (strict or small) and ext10_ok
print(f"EXT10 guard {'PASS' if ext10_ok else 'FAIL'}")
print("TY2:", ("ADOPT" + (" (strict)" if strict else " (small-sample route)")) if ok else "FAIL")
