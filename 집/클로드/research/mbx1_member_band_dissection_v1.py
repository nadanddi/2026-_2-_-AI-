# -*- coding: utf-8 -*-
"""MBX1 (2026-10-15 집 클로드; 사용자 '실험했던 것도 부품 단위로 다시 쪼개기' — 기본 모델 W40G-S 구성원).  서술 분해(판정 없음), 실행 전 커밋.  온도.
W40G-S 셀 = .4·BASE + (.2 + .4·(1−g))·CODEX + .4·g·PFN,  g = clip((in_temp − 8)/2) — 추운 행(g=0)은 .4 BASE + .6 CODEX.
배치 CV8(추운 42기록)·CK2·N·EL1 에서 실내온도 구간(≤ 6, 6~8, 8~10, 10~12, ≥ 12)별로 각 구성원(BASE 2시드 평균, CODEX, PFN A·B 평균)과 W40 혼합의 RMSE·평균 편향,
그리고 구간별 '최선 볼록 혼합(BASE/CODEX/PFN 0.1 격자)' 의 RMSE(사후 상한, 서술용).
Run:  cd 집/클로드/research && PYTHONPATH="" python -u mbx1_member_band_dissection_v1.py
"""
import os, sys, itertools
import env  # noqa: F401
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
D = {"CV8": os.path.join(env.LOCAL, "cv8_base_ckpt"), "CK2": os.path.join(env.LOCAL, "os1_base_ckpt"), "N": os.path.join(env.LOCAL, "os1_base_ckpt"), "EL1": os.path.join(env.LOCAL, "yl1_ref_ckpt")}
RD = set(map(tuple, pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "rw1_rough_days_v1.csv"))[["farm", "day"]].values))
rm = lambda e: float(np.sqrt(np.mean(np.square(e))))
BANDS = [("<=6", -99, 6), ("6-8", 6, 8), ("8-10", 8, 10), ("10-12", 10, 12), (">=12", 12, 99)]
W = list(itertools.product(np.round(np.arange(0, 1.01, .1), 1), repeat=2))
rows = []
for L, d in D.items():
    M = pd.concat([pd.read_csv(os.path.join(d, f)) for f in sorted(os.listdir(d)) if f.startswith(L + "_")], ignore_index=True)
    M = M[np.array([(f, x) not in RD for f, x in zip(M.farm, M.day)])].reset_index(drop=True); y = M.sub_temp.values; t = M.in_temp.values
    base = M[[c for c in M.columns if c.startswith("base_R_")]].mean(1).values; cx = M.codex_R.values; pf = (M.pfn_A.values + M.pfn_B.values) / 2
    g = np.where(np.isnan(t), 1, np.clip((t - 8) / 2, 0, 1)); w40 = .4 * base + (.2 + .4 * (1 - g)) * cx + .4 * g * pf
    for nm, lo, hi in BANDS:
        m = (t > lo) & (t <= hi) if lo > -99 else (t <= hi)
        if m.sum() < 10: continue
        best = min(((a, b) for a, b in W if a + b <= 1.0001), key=lambda ab: rm((ab[0] * base + ab[1] * cx + (1 - ab[0] - ab[1]) * pf - y)[m]))
        rows.append(dict(set=L, band=nm, rows=int(m.sum()), W40=rm((w40 - y)[m]), BASE=rm((base - y)[m]), CODEX=rm((cx - y)[m]), PFN=rm((pf - y)[m]),
                         bias_W40=float((w40 - y)[m].mean()), bias_BASE=float((base - y)[m].mean()), bias_CODEX=float((cx - y)[m].mean()), bias_PFN=float((pf - y)[m].mean()),
                         best_mix=f"B{best[0]:.1f}/C{best[1]:.1f}/P{1 - best[0] - best[1]:.1f}", best_rmse=rm((best[0] * base + best[1] * cx + (1 - best[0] - best[1]) * pf - y)[m])))
pd.set_option("display.width", 240); print(pd.DataFrame(rows).round(3).to_string(index=False))
