# -*- coding: utf-8 -*-
"""TY3 (2026-10-14 집 클로드).  사전 고정(커밋 후 실행).  온도.  과거 정보만.
TY2(6.881): 전날 같은 시각 잔차 앵커 — 8칸 모두 2차 개선(EV −.84~−.87%, DIAG10 −1.0%), DIAG10 P .0066, EV P .0252(기준 .025 근소 미달) → FAIL; EV 전체 행 F13 +.13~+.15% 악화.
같은 검증기로 다시 판정하지 않고, TY2 판정에 안 쓴 DIAG10 계열 새 배치 F·H·I·J·K·L(tt1_ckpt 별도 학습 배치; G 는 TY2 서술에 나와 제외)에서 두 안을 판정:
  Y2 = TY2 그대로(β_h 공통, 1차 cert 쌍 EV 잔차 원점 회귀, 0 아래 0) — 독립 재확인.
  Y3 = β_h 를 대상 농장별로(같은 방식, 그 농장 1차 쌍만) — [사후 동기: TY1·TY2 에서 F13 지속성 ≈ 0 · F13 악화를 본 뒤 설계].
기준 V23 = TC1c 사슬 셀 + v7 + TNA8 거울 앵커(tb4 와 같은 틀).  규칙: 앵커 짝 a(v7 ∪ TNA8 거울)가 있고 a 가 그 배치 프레임에 있으면 p_h += β_h · (y_a[h] − p_a[h]) (p_a = 같은 칸 V23 의 a 예측).
판정(안 2개, α .0125 각각): 6배치 x 4셀 = 24셀 모두 2차 개선 & 2차 기록 부트스트랩(기록마다 24셀 평균 ΔSSE, 20000) P(악화) < .0125 & 상위 3기록 제외 ΔSSE < 0
  & EXT10 전체 행 4셀 모두 악화 아님(보호).  서술: 전체 행, F13/F47, EXT12, 시각대.
"""
import os, sys
import env  # noqa: F401
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, "tb2_temp_agree_any_p_v1.py"), encoding="utf-8").read(); __file__ = os.path.join(HERE, "tb2_temp_agree_any_p_v1.py")
exec(src[:src.index("DD = pd.read_pickle")])
TNM = dict(ARMS["TNA8"][1]); V7P = {r: p for r, p in zip(L7.rec, L7.prev) if r in v7set}; PAIRS = {**V7P, **TNM}; TLY = TLx
DD = pd.read_pickle(os.path.join(HERE, "ev_validator", "temp", "local_ckpt", "ev_all.pkl")); DD["rec"] = DD.row_id.str[:7]
Rv = (DD.sub_temp - DD[[f"v22_a_{c}" for c in ("7A", "7B", "101A", "101B")]].mean(axis=1)).values; RP = pd.Series(Rv).groupby([DD.rec.values, DD.hour.values]).mean().unstack()
PR1 = [(rid(int(a)), rid(int(b))) for a, b, c in zip(TLY.a1, TLY.b, TLY.cert) if c]; PR1 = [(a, b) for a, b in PR1 if a in RP.index and b in RP.index and int(b[4:]) < 179]
def beta(pairs):
    A_ = RP.loc[[a for a, _ in pairs]].values; B_ = RP.loc[[b for _, b in pairs]].values
    return np.array([max(0.0, np.nansum(A_[:, h] * B_[:, h]) / np.nansum(np.where(np.isfinite(B_[:, h]), A_[:, h], np.nan) ** 2)) for h in range(24)])
BP = beta(PR1); BF = {f: beta([(a, b) for a, b in PR1 if b[:3] == f]) for f in ("F13", "F47")}
print("beta pooled", np.round(BP, 3).tolist()); [print("beta", f, np.round(v, 3).tolist()) for f, v in BF.items()]
src = open(os.path.join(HERE, "ext1_combo_b4_tn8_window_v1.py"), encoding="utf-8").read(); __file__ = os.path.join(HERE, "ext1_combo_b4_tn8_window_v1.py")
exec(src[:src.index("res = {k: []")])
__file__ = os.path.join(HERE, "ty3_yesterday_hour_fresh_layouts_v1.py"); rng = np.random.default_rng(20261221)
LV = pd.concat([L7f, pd.DataFrame(dict(rec=list(TNM), prev=list(TNM.values()), p=1.0, prev_is_labelled=True, anchor_st23=[float(ST[idx[p], 23]) for p in TNM.values()], dT_air=0.0, steps_to_anchor=1))], ignore_index=True)
def add_y(pred, yv, recs, hours, farm_beta):
    res = pd.Series(yv - pred).groupby([recs, hours]).mean(); have = set(res.index.get_level_values(0)); out = pred.copy(); n = 0
    for rec in np.unique(recs):
        a = PAIRS.get(rec)
        if a is None or a not in have: continue
        ix = np.where(recs == rec)[0]; bt = BF[rec[:3]] if farm_beta else BP
        out[ix] = pred[ix] + bt[hours[ix]] * np.nan_to_num(res.loc[a].reindex(hours[ix]).values); n += 1
    return out, n
JUDGE = ["F", "H", "I", "J", "K", "L"]; recd = {"Y2": [], "Y3": []}; cells_ok = {"Y2": True, "Y3": True}; ext_ok = {"Y2": True, "Y3": True}; lines = {"Y2": [], "Y3": []}
for L in JUDGE + ["EXT10", "EXT12"]:
    d, pre, bs, cx = T.LAY[L]; M1 = T.layout_frame(L, CT); y = M1.sub_temp.values; p2 = M1.day.values >= 179; f47 = M1.farm.values == "F47"
    mrec = np.array([f"{f}_{int(dd):03d}" for f, dd in zip(M1.farm, M1.day)]); mh = M1.hour.values.astype(int)
    base = [anch(M1, q, LV) for q in cells(M1, D1, 7, bs, cx)]
    for k, fb in (("Y2", False), ("Y3", True)):
        new = [add_y(b, y, mrec, mh, fb) for b in base]; nc = new[0][1]; new = [x[0] for x in new]
        ch = [rm(n - y) / rm(b - y) - 1 for n, b in zip(new, base)]; ch2 = [rm((n - y)[p2]) / rm((b - y)[p2]) - 1 for n, b in zip(new, base)]
        chf = [(rm((n - y)[f47]) / rm((b - y)[f47]) - 1, rm((n - y)[~f47]) / rm((b - y)[~f47]) - 1) for n, b in zip(new, base)]
        lines[k].append(f"{L:6s} corrected {nc:3d} | all {' '.join('%+.2f' % (100 * x) for x in ch)} | p2 {' '.join('%+.2f' % (100 * x) for x in ch2)} | F47/F13 {' '.join('%+.2f/%+.2f' % (100 * a, 100 * b) for a, b in chf)}")
        if L in JUDGE:
            cells_ok[k] &= all(x < 0 for x in ch2)
            recd[k].append(pd.concat([pd.Series(((n - y) ** 2 - (b - y) ** 2)[p2]).groupby(mrec[p2]).sum() for n, b in zip(new, base)], axis=1).mean(axis=1))
        if L == "EXT10": ext_ok[k] = all(x <= 0 for x in ch)
    print("done", L, flush=True)
for k in ("Y2", "Y3"):
    D = pd.concat(recd[k], axis=1).fillna(0).mean(axis=1); v = D.values; P = float((v[rng.integers(0, len(v), (20000, len(v)))].sum(1) >= 0).mean()); t3 = D.sort_values().iloc[3:].sum()
    ok = cells_ok[k] and P < .0125 and t3 < 0 and ext_ok[k]
    print(f"\n== {k} ({'farm' if k == 'Y3' else 'pooled'} beta) vs V23"); print("\n".join(lines[k]))
    print(f"{k}: 24 cells pass-2 better {cells_ok[k]} | record P {P:.4f} | drop-top3 {t3:+.2f} | changed {(D.abs() > 1e-12).sum()} harmed {(D > 1e-6).sum()} | EXT10 guard {ext_ok[k]} -> {'ADOPT' if ok else 'FAIL'}")
    print("   top gains:", (-D).sort_values(ascending=False).head(5).round(2).to_dict(), "| worst:", (-D).sort_values().head(4).round(2).to_dict())
