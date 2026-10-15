# -*- coding: utf-8 -*-
"""AT1 v2 (2026-10-15 집 클로드; v1 무효 수정: 사용자: '가장 쉬운 부분부터' → AC1·AC2(EC) 다음 온도).  사전 고정(커밋 후 실행).  온도.  학습 없음.
27번 온도 = NX(W40G-S) → (사슬 없음, FL4) → FL 어제 온도 앵커(TA7 꼴: gap = clip(A23 − .36 − q0, ±2) x BETA[h], p ≥ .8, 정답 전날, |dT_air| ≤ 1.5).
후보 S5: 앵커 조건을 p ≥ .5 로 넓히고 gap 에 p 를 곱함(확률만큼만 옮김; 미리 고정, 튜닝 없음).  링크 = FL2 거울 검증표 local/fl2_cv_FL.csv (FL2 온도 판정과 같은 표).
v1 은 '앵커 전날이 배치 검증 기록이면 사용 안 함'을 배치 전체(거의 모든 기록)에 걸어 대부분 판에서 앵커 0 → 무효.
v2: FL2 온도 판정과 같이 숨김은 거울 검증표(prev_is_labelled)가 맡음(같은 한계).  판정 규칙 그대로.
기본 q = B.w40 셀 출력(사슬 없음).  판 DIAG10·EL1·F·G·H·I + 있으면 J·K·L, 안전 EXT10·EXT12.
판정(α .025): 모든 판 모든 셀 S5 < K8 & 합친(EL1 제외, 셀 1, 농장 x 5일 블록) P(악화) < .025 & EXT 안전(셀 평균 악화 & '나음' 비율 < .025 이면 FAIL).
서술: 새로 앵커 걸린 기록만, 2차, 추운 행, K8 vs 앵커 없음.
Run:  cd 집/클로드/research && PYTHONPATH="" python -u at1_temp_anchor_coverage_v2.py
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
LAYS = [L for L in ["DIAG10", "EL1", "F", "G", "H", "I", "J", "K", "L"] if L in T.LAY]; rng = np.random.default_rng(20270114); ok = True; diffs = []; safety = {}
for L in LAYS + ["EXT10", "EXT12"]:
    d, pre, bs, cx = T.LAY[L]; M = T.layout_frame(L, CT); y = M.sub_temp.values; key = (M.farm + "_" + (M.day // 5).astype(str)).values
    rk = (M.farm + "_" + M.day.map("{:03d}".format)).values; P = {"NONE": [], "K8": [], "S5": []}; new = set()
    for bi, b in enumerate(bs):
        for fam in "AB":
            q = B.w40(M, b, cx, fam); k8, u8 = corr(M, q, .8, False); s5, u5 = corr(M, q, .5, True); P["NONE"].append(q); P["K8"].append(k8); P["S5"].append(s5); new = u5 - u8
    ch = [rmf(s - y) / rmf(k - y) - 1 for s, k in zip(P["S5"], P["K8"])]; nw = np.isin(rk, list(new)); p2 = M.day.values >= 179; cold = (M.in_temp <= 8).values
    f = lambda m: " ".join("%+.2f" % (100 * (rmf((s - y)[m]) / rmf((k - y)[m]) - 1)) for s, k in zip(P["S5"], P["K8"])) if m.any() else "-"
    k8n = " ".join("%+.2f" % (100 * (rmf(k - y) / rmf(n - y) - 1)) for k, n in zip(P["K8"], P["NONE"]))
    print(f"{L:6s} S5 vs K8 all {' '.join('%+.2f' % (100 * x) for x in ch)} | new recs({len(new)}) {f(nw)} | p2 {f(p2)} | cold {f(cold)} || K8 vs none {k8n}", flush=True)
    if L in ("EXT10", "EXT12"):
        dd = (P["S5"][0] - y) ** 2 - (P["K8"][0] - y) ** 2; safety[L] = not (np.mean(ch) > 0 and (dd < 0).mean() < .025); continue
    ok &= all(x < 0 for x in ch)
    if L != "EL1": diffs.append(pd.Series((P["S5"][0] - y) ** 2 - (P["K8"][0] - y) ** 2).groupby(key).mean())
S = pd.concat(diffs, axis=1).mean(axis=1).values; P_ = float((S[rng.integers(0, len(S), (20000, len(S)))].mean(1) >= 0).mean())
print(f"AT1 S5 vs K8: all cells better {ok} | pooled P {P_:.4f} | safety {safety} -> {'PASS' if ok and P_ < .025 and all(safety.values()) else 'FAIL'}")
