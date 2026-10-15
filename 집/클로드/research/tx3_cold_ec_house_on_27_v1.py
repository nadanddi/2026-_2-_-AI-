# -*- coding: utf-8 -*-
"""TX3 (2026-10-15 집 클로드; 사용자: '특정 부분에서만 빛난 실험 떼어 쓰기' 1번).  사전 고정(커밋 후 실행).  온도.  학습 없음.
TX2 COLD(6.799: 44칸 모두 개선, EXT10 P .035 근소 FAIL)를 27번 온도 구성 위에서 재판정.
  기준 = W40G-S 셀(사슬 없음) + FL 어제 온도 앵커(fl2_cv_FL, p ≥ .8·정답 전날·|dT| ≤ 1.5; AT1 v2 의 K8 과 같음).
  후보 = 기준 + .25 · w_h · g_h,  w_h = clip((cmEC_h − .8)/.4, 0, 1),  g_h = clip((10 − in_temp_h)/2, 0, 1)(결측 0)  — TX2 와 같은 상수(새로 맞추지 않음).
  cmEC_h = 27번 EC 기본 NH(nh1 DIAG10 OOF, 3시드 평균) 0..h 누적평균(대상 정답 미사용; 없는 기록은 w = 0).
5점 분류: F (자기 기록 입력으로 만든 EC 예측 + 학습 정답으로 잰 물리 차이 + 0시 이후 현재 실내온도).
판정(단일 안 α .025, 온도 범위 규칙 COLD): 주 EXT10 4칸 모두 개선 & 농장×5일 블록 P(악화) < .025;
  나머지 판(DIAG10·EL1·F~L·EXT12)은 유의한 악화 없음(칸 평균 악화 & P(나음) < .025 이면 FAIL).
한계: 같은 배치·같은 상수(TX2 에서 본 판) — 새 증거는 '사슬 없음 + FL 앵커 + NH 게이트' 구성에서도 재현되는지뿐.
Run:  cd 집/클로드/research && PYTHONPATH="" python -u tx3_cold_ec_house_on_27_v1.py
"""
import os, sys, importlib.util
import env  # noqa: F401
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, "tv3_v3_gru_on_v23_v1.py"), encoding="utf-8").read(); g = {"__file__": os.path.join(HERE, "tv3_v3_gru_on_v23_v1.py"), "__name__": "tv3at1"}
exec(src[:src.index("JUDGE = [")], g)
T, B, CT, rmf, A7 = g["T"], g["B"], g["CT"], g["rm"], g["A7"]
if "L" not in T.LAY: T.LAY["L"] = ("tt27_ckpt", "L_", ("base_R_1414", "base_R_1515"), "codex_R")
_X = pd.concat([pd.read_csv(os.path.join(env.DATA, f)) for f in ("train_X.csv", "test_X.csv")]); _X = _X[_X.row_id.str[:3].isin(["F13", "F47"])]
_X["rec"] = _X.row_id.str[:7]; _X["h"] = _X.row_id.str[8:].astype(int); TIN = _X.pivot(index="rec", columns="h", values="in_temp")
O = pd.read_csv(os.path.join(env.LOCAL, "fl2_cv_FL.csv"), encoding="utf-8-sig"); O["dT_air"] = [TIN.at[r, 0] - TIN.at[p, 23] if p in TIN.index else np.nan for r, p in zip(O.rec, O.prev)]
O = O[O.prev_is_labelled.astype(bool) & (O.dT_air.abs() <= 1.5) & O.anchor_st23.notna()].set_index("rec")
def corr(M, q, pmin, soft):
    out = q.copy(); h = M.hour.values; key = (M.farm + "_" + M.day.map("{:03d}".format)).values; own = set(key)
    sel = O[O.p >= pmin]; q0 = pd.Series(q[h == 0], index=key[h == 0]); used = set()
    for rec in sel.index.intersection(q0.index):
        gap = np.clip(sel.at[rec, "anchor_st23"] - .36 - q0[rec], -2, 2) * (sel.at[rec, "p"] if soft else 1.0)
        m = key == rec; out[m] = q[m] + A7.BETA[h[m]] * gap; used.add(rec)
    return out, used
import pickle
_R = pd.read_csv(os.path.join(HERE, "nh1_rows_v1.csv.gz")); _R = _R[_R.validator == "DIAG10"]
_MT = pickle.load(open(os.path.join(env.LOCAL, "nh1_mt1_v1.pkl"), "rb")); _mt = np.array([_MT[(v, k)][r][h] for v, k, r, h in zip(_R.validator, _R.fold, _R.rec, _R.hour)])
_R["NH"] = np.mean([(_R[f"A1_{s}"].values - .1 * _mt) / .9 for s in (1357, 2468, 3579)], axis=0)
Eh = _R.groupby(["rec", "hour"]).NH.mean().reset_index().sort_values(["rec", "hour"]); Eh["cm"] = Eh.groupby("rec").NH.transform(lambda s: s.expanding().mean())
CM = {(r, int(h)): c for r, h, c in zip(Eh.rec, Eh.hour, Eh.cm)}; print("NH gate records", Eh.rec.nunique(), flush=True)
LAYS = [L for L in ["DIAG10", "EL1", "F", "G", "H", "I", "J", "K", "L", "EXT10", "EXT12"] if L in T.LAY]; rng = np.random.default_rng(20270115); ok = True
for L in LAYS:
    d, pre, bs, cx = T.LAY[L]; M = T.layout_frame(L, CT); y = M.sub_temp.values; bk = (M.farm + "_" + (M.day // 5).astype(str)).values
    rk = (M.farm + "_" + M.day.map("{:03d}".format)).values; cm = np.array([CM.get((k, int(hh)), np.nan) for k, hh in zip(rk, M.hour.values)])
    w = np.clip((np.nan_to_num(cm, nan=0) - .8) / .4, 0, 1); g_ = np.nan_to_num(np.clip((10 - M.in_temp.values) / 2, 0, 1), nan=0.0); add = .25 * w * g_
    ch = []; dd0 = None
    for bi, b in enumerate(bs):
        for fam in "AB":
            base, _ = corr(M, B.w40(M, b, cx, fam), .8, False); new = base + add; ch.append(rmf(new - y) / rmf(base - y) - 1)
            if dd0 is None: dd0 = pd.Series((new - y) ** 2 - (base - y) ** 2).groupby(bk).mean().values
    m = dd0[rng.integers(0, len(dd0), (20000, len(dd0)))].mean(1); pw, pb = float((m >= 0).mean()), float((m < 0).mean())
    cold = (M.in_temp <= 8).values; nchg = int((add > 0).sum())
    if L == "EXT10":
        c = all(x < 0 for x in ch) and pw < .025; ok &= c; tag = f"[primary] P(worse) {pw:.4f} -> {'ok' if c else 'FAIL'}"
    else:
        bad = np.mean(ch) > 0 and pb < .025; ok &= not bad; tag = f"P(better) {pb:.3f}{' SIGNIF WORSE' if bad else ''}"
    print(f"{L:6s} TX3 vs base {' '.join('%+.3f' % (100 * x) for x in ch)} | rows changed {nchg} (cold rows {int(cold.sum())}) | {tag}", flush=True)
print("TX3:", "ADOPT" if ok else "FAIL")
