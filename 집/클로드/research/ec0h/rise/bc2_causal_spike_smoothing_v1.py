# -*- coding: utf-8 -*-
"""BC2 (2026-10-12 집 클로드, 기초 점검·탐색): EC 예측의 한 시간짜리 튐을 인과적 후처리로 줄이면 좋아지나?
규정: 행 h 는 같은 기록 0..h 입력만 → 예측도 p_0..p_h 만 사용(뒤 시각 예측 사용 금지).
후보(사전 나열, 모두 보고): CLIP_c: p'_h = p'_{h-1} + clip(p_h − p'_{h-1}, ±c), c ∈ {.05, .1, .2}; MED3: median(p_{h-2}, p_{h-1}, p_h) (h<2 는 그대로);
EW_a: p'_h = a p_h + (1−a) p'_{h-1}, a ∈ {.5, .7}.  0시는 그대로.
기준 예측: EV EA2(3시드, 시드별), DIAG10 CUR sg(시드별).  채택 기준은 두지 않음(탐색) — 모든 시드×두 검증기 같은 방향이면 후속 사전 고정 시험 후보.
"""
import sys, os
sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env
import numpy as np, pandas as pd
H_ = os.path.join(env.ROOT, u"집", u"클로드", "research")
R = pd.read_csv(os.path.join(H_, "ev_validator", "ec", "out_v1", "v2_ev_ec_rows.csv")); R = R[(R.arm == "BASE") & (R.seed.astype(str) != "avg")]
O = pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "ec_rv1_preds_v1.csv")); O = O[(O.tag == "CUR") & (O.validator == "DIAG10")]; O["rec"] = O.row_id.str[:7]
def clip(P, c):
    Q = P.copy()
    for h in range(1, 24): Q[:, h] = Q[:, h - 1] + np.clip(P[:, h] - Q[:, h - 1], -c, c)
    return Q
def med3(P):
    Q = P.copy(); Q[:, 2:] = np.median(np.stack([P[:, :-2], P[:, 1:-1], P[:, 2:]]), 0); return Q
def ew(P, a):
    Q = P.copy()
    for h in range(1, 24): Q[:, h] = a * P[:, h] + (1 - a) * Q[:, h - 1]
    return Q
F = {"CLIP.05": lambda P: clip(P, .05), "CLIP.1": lambda P: clip(P, .1), "CLIP.2": lambda P: clip(P, .2), "MED3": med3, "EW.5": lambda P: ew(P, .5), "EW.7": lambda P: ew(P, .7)}
rm = lambda e: float(np.sqrt(np.nanmean(np.square(e))))
for tag, D, pc, yc in (("EV EA2", R, "EA2", "y"), ("DIAG10 sg", O, "sg", "sub_ec")):
    print(f"== {tag}")
    for sd, G in D.groupby("seed"):
        P = G.pivot_table(index="rec", columns="hour", values=pc); Y = G.groupby(["rec", "hour"])[yc].first().unstack().loc[P.index]
        p2 = np.array([int(r[4:7]) >= 179 for r in P.index]); b = rm((P - Y).values); b2 = rm((P - Y).values[p2])
        print(f"  seed {sd}: base {b:.4f} (p2 {b2:.4f}) | " + " | ".join(f"{k} {100*(rm(f(P.values)-Y.values)/b-1):+.2f}% (p2 {100*(rm((f(P.values)-Y.values)[p2])/b2-1):+.2f}%)" for k, f in F.items()))
