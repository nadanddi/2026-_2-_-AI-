# -*- coding: utf-8 -*-
"""VW2 (2026-10-15 집 클로드; 사용자: '자고 올 테니 뭐라도 마구 해봐').  사전 고정(커밋 후 실행).  온도.
V3L(6.912 ADOPT)의 섞기 비율 .1 과 날 문턱(g·h)은 옛 V3 기준값.  V3L 이 더 좋으니 비율·문턱을 다시 고른다(저장 예측만, 재학습 없음).
후보: 비율 w ∈ {.10, .15, .20, .30, .40} x 문턱 {GH = 지금(g·h), G = g 만(그날 최저 문턱 h 없음)} — 10개, 기준 = (.10, GH).
선택: DIAG10 + EL1 셀 평균 RMSE(v23 사슬·앵커 위) 가 가장 낮은 안 하나(선택 배치).
판정(선택된 안이 기준이 아니면, α .025): F·G·H·I 모든 셀에서 기준보다 개선 & 합친(F~I, 셀 1) 5일 블록 P(악화) < .025 & EXT10·EXT12 안전(기준 대비 셀 평균 악화 & '나음' 비율 < .025 이면 FAIL).
서술: 모든 후보의 판별 배치 결과(선택에는 안 씀).
Run:  cd 집/클로드/research && PYTHONPATH="" python -u vw2_v3l_weight_gate_sweep_v1.py
"""
import os, sys
import env  # noqa: F401
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, "tv3_v3_gru_on_v23_v1.py"), encoding="utf-8").read(); g = {"__file__": os.path.join(HERE, "tv3_v3_gru_on_v23_v1.py"), "__name__": "tv3vw2"}
exec(src[:src.index("JUDGE = [")], g)
T, B, C2, anch, LV, D1, CT, rmf = g["T"], g["B"], g["C2"], g["anch"], g["LV"], g["D1"], g["CT"], g["rm"]
CKL = os.path.join(env.LOCAL, "v3l_ckpt"); rng = np.random.default_rng(20261228)
def ld(L):
    fs = [f for f in os.listdir(CKL) if f.startswith(L + "_")]; return pd.concat([pd.read_csv(os.path.join(CKL, f)) for f in fs]).drop_duplicates("row_id")
CANDS = [(w, gt) for w in (.10, .15, .20, .30, .40) for gt in ("GH", "G")]; BASE = (.10, "GH")
OUT = {}
for L in ["DIAG10", "EL1", "F", "G", "H", "I", "EXT10", "EXT12"]:
    d, pre, bs, cx = T.LAY[L]; M = T.layout_frame(L, CT); y = M.sub_temp.values
    v = M[["row_id"]].merge(ld(L)[["row_id", "V3L"]], on="row_id", how="left").V3L.values; miss = np.isnan(v)
    g_ = np.where(M.in_temp.isna(), 1, np.clip((M.in_temp - 8) / 2, 0, 1)); order = M.sort_values(["farm", "day", "hour"]).index
    mm = M.loc[order].groupby(["farm", "day"]).in_temp.cummin().reindex(M.index); h_ = np.where(mm.isna(), 1, np.clip((mm - 10) / 2, 0, 1))
    key = (M.farm + "_" + (M.day // 5).astype(str)).values; preds = {c: [] for c in CANDS}
    for bi, b in enumerate(bs):
        for fam in "AB":
            a = B.w40(M, b, cx, fam)
            for (w, gt) in CANDS:
                wt = w * g_ * (h_ if gt == "GH" else 1) * (~miss); preds[(w, gt)].append(anch(M, C2.corrected(M, a + wt * (np.nan_to_num(v) - a), D1, bi, fam, 7), LV))
    OUT[L] = (y, key, preds); print("done", L, flush=True)
sel = {c: np.mean([np.mean([rmf(p - OUT[L][0]) for p in OUT[L][2][c]]) for L in ("DIAG10", "EL1")]) for c in CANDS}
best = min(sel, key=sel.get); print("selection (DIAG10+EL1 cell-mean RMSE):", {f"{w}/{gt}": round(s, 5) for (w, gt), s in sel.items()}, "-> chosen", best)
for c in CANDS:
    print(f"  {c}: " + " | ".join(f"{L} {100 * np.mean([rmf(p - OUT[L][0]) / rmf(q - OUT[L][0]) - 1 for p, q in zip(OUT[L][2][c], OUT[L][2][BASE])]):+.2f}" for L in ("F", "G", "H", "I", "EXT10", "EXT12")))
if best == BASE:
    print("VW2: chosen = baseline -> no change"); sys.exit(0)
ok = True; diffs = []; safety = {}
for L in ("F", "G", "H", "I", "EXT10", "EXT12"):
    y, key, P = OUT[L]; ch = [rmf(n - y) / rmf(b_ - y) - 1 for n, b_ in zip(P[best], P[BASE])]
    if L in ("F", "G", "H", "I"):
        ok &= all(x < 0 for x in ch); diffs.append(pd.Series((P[best][0] - y) ** 2 - (P[BASE][0] - y) ** 2).groupby(key).mean())
    else:
        pw = []
        for n, b_ in zip(P[best], P[BASE]):
            dd = pd.Series((n - y) ** 2 - (b_ - y) ** 2).groupby(key).mean().values; pw.append(float((dd[rng.integers(0, len(dd), (20000, len(dd)))].mean(1) < 0).mean()))
        safety[L] = not (np.mean(ch) > 0 and min(pw) < .025); ok &= safety[L]
S = pd.concat(diffs, axis=1).mean(axis=1).values; P_ = float((S[rng.integers(0, len(S), (20000, len(S)))].mean(1) >= 0).mean())
print(f"judge F~I: all cells better {ok and all(True for _ in [0])} | pooled P(worse) {P_:.4f} | safety {safety}\nVW2:", f"ADOPT {best}" if (ok and P_ < .025) else "FAIL (keep .10/GH)")
