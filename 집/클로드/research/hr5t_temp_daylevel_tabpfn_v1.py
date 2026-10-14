# -*- coding: utf-8 -*-
"""HR5-T (2026-10-14 집 클로드; 사용자: '2번부터 진행해봐').  사전 고정(커밋 후 실행).  온도.  일반 방법(자기 기록 입력 + 기본 예측만).
근거: 15회차 EC HR5(TabPFN 하루 수준 보정, 자기 입력 0..h 요약) 리더보드 −9.1%.  온도 오차의 74% 가 하루 수준(TX3) — 지금은 앵커(규칙성)만 고침.
설계: 기준 GV = W40G-S + V3 10%(일반 구성, 6.893 섞기 규칙).  기준 시각 h' ∈ {0,3,…,21} 마다 TabPFN v2(GPU, n_est 4, random_state 0):
  특징 = [입력 14종 + 기본 예측]의 0..h' 누적 평균, 0..min(h',7) 평균(h' > 7 일 때만), 9..h' 평균(h' >= 9 일 때만) (정의상 같은 열 제외 — 6.895);
  목표 = 하루 수준 잔차 d = 실제 하루 평균 − 기본 하루 평균;  학습 행 = DIAG10 배치 OOF 의 GV(4셀 평균) 학습 정답 기록(거친 날 제외);
  검증 폴드마다 숨김 = 그 폴드 기록 + 같은 농장 ±1 번호 + 다른 농장 ±3 번호.  시각 h 예측 = GV_h + 0.5 · d̂_{h'}, h' = h 이하 최대 기준 시각 (λ .5 = EC HR5 값, 다시 안 고름).
판정(미리 정함, 단일 안 α .025): DIAG10·EL1·F·G·H·I·K 모든 셀 전체 행 GVH < GV & 합친(DIAG10+F+G+H+I+K) 5일 블록 P(악화) < .025 & EXT10·EXT12 안전(셀 평균 악화 & 나음 비율 < .025 이면 FAIL).
서술: 2차 행, 지금 구성 CV(사슬 + v7 + TNA8 + V3) 대비 — GV→CV 격차 중 HR5-T 가 메운 비율.
[한계: 학습 행 기본은 DIAG10 OOF(다른 배치 셀과 분포 차이 가능, EC HR5 와 같은 설계)]
"""
import os, sys
import env  # noqa: F401
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, "tb2_temp_agree_any_p_v1.py"), encoding="utf-8").read(); __file__ = os.path.join(HERE, "tb2_temp_agree_any_p_v1.py")
exec(src[:src.index("DD = pd.read_pickle")])
TNA8M = dict(ARMS["TNA8"][1])
src = open(os.path.join(HERE, "ext1_combo_b4_tn8_window_v1.py"), encoding="utf-8").read(); __file__ = os.path.join(HERE, "ext1_combo_b4_tn8_window_v1.py")
exec(src[:src.index("res = {k: []")])
__file__ = os.path.join(HERE, "tv3_v3_gru_on_v23_v1.py"); rng = np.random.default_rng(20261224)
LV = pd.concat([L7f, pd.DataFrame(dict(rec=list(TNA8M), prev=list(TNA8M.values()), p=1.0, prev_is_labelled=True, anchor_st23=[float(ST[idx[p], 23]) for p in TNA8M.values()], dT_air=0.0, steps_to_anchor=1))], ignore_index=True)
def v3_for(L, M):
    if "V3" in M.columns: return M.V3.values
    if L in ("DIAG10", "EL1", "EXT10", "EXT12"):
        d = os.path.join(env.LOCAL, "ts7b_ckpt"); X = pd.concat([pd.read_csv(os.path.join(d, f)) for f in os.listdir(d) if f.startswith(L + "_")]); X = X[X.validator == L]
    else:
        d = os.path.join(env.LOCAL, "ts7_ckpt"); X = pd.concat([pd.read_csv(os.path.join(d, f)) for f in os.listdir(d) if f.startswith(L + "_")])
    return M[["row_id"]].merge(X[["row_id", "V3"]].drop_duplicates("row_id"), on="row_id", how="left").V3.values

from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion
import torch
DEV = "cuda" if torch.cuda.is_available() else "cpu"
USEI = ["out_temp", "out_hum", "out_rad", "out_wspd", "in_temp", "in_hum", "in_co2", "act_vent", "act_shade", "act_thermal", "act_heating", "act_circfan", "act_co2", "act_fog"]
HS = (0, 3, 6, 9, 12, 15, 18, 21); OTH = {"F13": "F47", "F47": "F13"}; LAM = .5
_X2 = pd.read_csv(os.path.join(env.DATA, "train_X.csv")); _X2 = _X2[_X2.row_id.str[:3].isin(["F13", "F47"])]; _X2["rec"] = _X2.row_id.str[:7]; _X2["hh"] = _X2.row_id.str[8:].astype(int)
INP = {r: g.sort_values("hh")[USEI].values.astype(float) for r, g in _X2.groupby("rec")}
def keep(h): return np.r_[np.arange(0, 15), np.arange(15, 30) if h > 7 else np.arange(0), np.arange(30, 45) if h >= 9 else np.arange(0)].astype(int)
def feat(inp, base, h):
    Z = np.concatenate([inp, base[:, None]], 1)
    with np.errstate(all="ignore"):
        cm = np.nanmean(Z[:h + 1], 0); nt = np.nanmean(Z[:min(h, 7) + 1], 0); dy = np.nanmean(Z[9:h + 1], 0) if h >= 9 else cm
    return np.concatenate([cm, nt, dy])[keep(h)]
def gv_cells(L):
    d, pre, bs, cx = T.LAY[L]; M = T.layout_frame(L, CT); y = M.sub_temp.values; v3 = v3_for(L, M); miss = np.isnan(v3)
    g = np.where(M.in_temp.isna(), 1, np.clip((M.in_temp - 8) / 2, 0, 1))
    order = M.sort_values(["farm", "day", "hour"]).index; mm = M.loc[order].groupby(["farm", "day"]).in_temp.cummin().reindex(M.index)
    h = np.where(mm.isna(), 1, np.clip((mm - 10) / 2, 0, 1)); wgt = .1 * g * h * (~miss)
    GV, CV = [], []
    for bi, b in enumerate(bs):
        for fam in "AB":
            a = B.w40(M, b, cx, fam); a2 = a + wgt * (np.nan_to_num(v3) - a); GV.append(a2); CV.append(anch(M, C2.corrected(M, a2, D1, bi, fam, 7), LV))
    M = M.assign(rec=[f"{f}_{int(dd):03d}" for f, dd in zip(M.farm, M.day)]); return M, y, GV, CV
# training table from DIAG10 GV (cell mean)
MD, yD, GVD, _ = gv_cells("DIAG10"); MD = MD.assign(gv=np.mean(GVD, 0), y=yD)
TRT = {}
for r, g in MD.groupby("rec"):
    g = g.sort_values("hour")
    if len(g) == 24 and r in INP and np.isfinite(g.gv.values).all() and np.isfinite(g.y.values).sum() >= 12: TRT[r] = (g.gv.values, float(np.nanmean(g.y.values) - g.gv.mean()))
TRK = sorted(TRT); TF = {h: np.stack([feat(INP[r], TRT[r][0], h) for r in TRK]) for h in HS}; TD = np.array([TRT[r][1] for r in TRK])
for h in HS: TF[h] = np.where(np.isfinite(TF[h]), TF[h], np.nanmean(TF[h], 0))
tfarm = np.array([r[:3] for r in TRK]); tnum = np.array([int(r[4:]) for r in TRK]); print("corrector training records", len(TRK), "| d sd %.3f" % TD.std(), flush=True)
JUDGE = ["DIAG10", "EL1", "F", "G", "H", "I", "K"]; ok = True; diffs = []; lines = []; safety = {}; gapfill = {}
for L in JUDGE + ["EXT10", "EXT12"]:
    M, y, GV, CV = gv_cells(L); M = M.reset_index(drop=True); hrs = M.hour.values.astype(int); new = [x.copy() for x in GV]
    for k, Gk in M.groupby("fold"):
        val = set(Gk.rec); hide = np.zeros(len(TRK), bool)
        for r in val:
            f, n = r[:3], int(r[4:]); hide |= ((tfarm == f) & (np.abs(tnum - n) <= 1)) | ((tfarm == OTH[f]) & (np.abs(tnum - n) <= 3))
        models = {}
        for h in HS:
            m = TabPFNRegressor.create_default_for_version(ModelVersion.V2, device=DEV, n_estimators=4, random_state=0, ignore_pretraining_limits=True); m.fit(TF[h][~hide], TD[~hide]); models[h] = m
        recs = [r for r in sorted(val) if r in INP]
        for ci in range(len(GV)):
            ix = {r: Gk.index[Gk.rec.values == r] for r in recs}; ok_r = [r for r in recs if len(ix[r]) == 24]
            if not ok_r: continue
            DHc = {}
            for h in HS:
                Q = np.stack([feat(INP[r], GV[ci][ix[r]][np.argsort(hrs[ix[r]])], h) for r in ok_r]); Q = np.where(np.isfinite(Q), Q, TF[h].mean(0)[None, :])
                DHc[h] = dict(zip(ok_r, models[h].predict(Q)))
            for r in ok_r:
                ii = ix[r]; new[ci][ii] = GV[ci][ii] + LAM * np.array([DHc[max(x for x in HS if x <= hh)][r] for hh in hrs[ii]])
    p2 = M.day.values >= 179
    ch = [rm(n - y) / rm(b_ - y) - 1 for n, b_ in zip(new, GV)]; ch2 = [rm((n - y)[p2]) / rm((b_ - y)[p2]) - 1 for n, b_ in zip(new, GV)]
    gf = [(rm(b_ - y) - rm(n - y)) / (rm(b_ - y) - rm(c - y)) for n, b_, c in zip(new, GV, CV)]; gapfill[L] = np.mean(gf)
    lines.append(f"{L:6s} GVH vs GV all {' '.join('%+.2f' % (100 * x) for x in ch)} | p2 {' '.join('%+.2f' % (100 * x) for x in ch2)} | RMSE GV {np.mean([rm(b_ - y) for b_ in GV]):.4f} GVH {np.mean([rm(n - y) for n in new]):.4f} CV {np.mean([rm(c - y) for c in CV]):.4f} | gap filled {100 * np.mean(gf):.0f}%")
    key = (M.farm + "_" + (M.day // 5).astype(str)).values
    if L in JUDGE:
        ok &= all(x < 0 for x in ch)
        if L != "EL1": diffs.append(pd.Series((new[0] - y) ** 2 - (GV[0] - y) ** 2).groupby(key).mean())
    else:
        pw = []
        for n, b_ in zip(new, GV):
            dd = pd.Series((n - y) ** 2 - (b_ - y) ** 2).groupby(key).mean().values; pw.append(float((dd[rng.integers(0, len(dd), (20000, len(dd)))].mean(1) < 0).mean()))
        safety[L] = not (np.mean(ch) > 0 and min(pw) < .025); ok &= safety[L]
    print("done", L, flush=True)
S = pd.concat(diffs, axis=1).mean(axis=1).values; P = float((S[rng.integers(0, len(S), (20000, len(S)))].mean(1) >= 0).mean())
print("\n".join(lines)); print(f"\npooled P(worse) {P:.4f} | cold safety {safety}")
print("HR5-T:", "ADOPT" if (ok and P < .025) else "FAIL")
