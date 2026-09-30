# -*- coding: utf-8 -*-
"""재분석 12 (2026-10-01, 집 클로드): 2차 구간(일차>=179)에서 이웃 날 정보가 하루 수준을 알려 주는가 — 탐색.
(1) G_C2 DIAG10 하루 오프셋의 이웃 상관: 1차 구간 vs 2차 구간 (k=1,2)
(2) 라벨 하루 평균 자체의 이웃 상관 (sub_temp; sub_ec는 참고) 1차 vs 2차
(3) 2차 구간에서 앞뒤 학습일 라벨 평균(보간)으로 하루 평균을 맞히면 오차 얼마 — 모델 하루 평균과 비교
결과 local/re12_pass2_neighbors.txt
"""
import env  # noqa
import os, numpy as np, pandas as pd
from harness import load
_, lab, _ = load()
z = np.load(env.LOCAL + "/temp_mask_v1_oof.npz", allow_pickle=True)
t = lab.in_temp.values; g = np.where(np.isnan(t), 1, np.clip((t - 8) / 2, 0, 1)); s = "DIAG10"
base = np.nanmean([z[f"{s}__MASK__7"], z[f"{s}__MASK__101"]], 0)
cx = np.nanmean([z[f"{s}__CODEX__726"], z[f"{s}__CODEX__727"]], 0)
pfn = np.load(env.LOCAL + f"/web_tabpfn_v2_temp_{s}.npy").mean(0)
lab = lab.assign(pred=(0.6 - 0.2 * (1 - g)) * base + (0.2 + 0.4 * (1 - g)) * cx + 0.2 * g * pfn)
D = lab.groupby(["farm", "day"]).agg(y=("sub_temp", "mean"), p=("pred", "mean"), ec=("sub_ec", "mean"),
                                     tin=("in_temp", "mean")).reset_index()
D["e"] = D.p - D.y
out = open(env.LOCAL + "/re12_pass2_neighbors.txt", "w", encoding="utf-8")
def P(x): print(x); out.write(x + "\n")
idx = D.set_index(["farm", "day"])
def pair(k, cond):
    a = D[cond(D.day)].copy()
    a["key"] = list(zip(a.farm, a.day - k))
    a = a[a.key.isin(idx.index)]
    b = idx.loc[a.key.tolist()]
    return a, b
for seg, cond in [("1차(<179)", lambda d: d < 179), ("2차(>=179)", lambda d: d >= 179)]:
    for k in (1, 2):
        a, b = pair(k, cond)
        P(f"{seg} k={k} n={len(a)}: 오프셋 r {np.corrcoef(a.e, b.e.values)[0,1]:+.3f} | "
          f"sub_temp 하루평균 r {np.corrcoef(a.y, b.y.values)[0,1]:+.3f} | "
          f"(sub_temp − in_temp) 하루평균 r {np.corrcoef(a.y-a.tin, (b.y-b.tin).values)[0,1]:+.3f} | "
          f"sub_ec r {np.corrcoef(a.ec, b.ec.values)[0,1]:+.3f}")
# (3) 2차 구간: 앞뒤 가장 가까운 학습일(자기 제외, ±1..3)의 (y - tin) 평균 + 자기 tin 으로 하루 평균 추정
P("\n2차 구간 하루 평균 추정 RMSE (자기 날 제외, 이웃 ±1~3일 중 가까운 것):")
S = D[D.day >= 179].copy()
est = []
for r in S.itertuples():
    nb = D[(D.farm == r.farm) & (D.day != r.day) & ((D.day - r.day).abs() <= 3) & (D.day >= 179)]
    est.append(np.nan if nb.empty else r.tin + (nb.y - nb.tin).mean())
S["est_nb"] = est
m = S.est_nb.notna()
P(f"  n={m.sum()}  이웃 차이 보간: {np.sqrt(((S.est_nb-S.y)[m]**2).mean()):.3f} | G_C2 하루평균: {np.sqrt(((S.p-S.y)[m]**2).mean()):.3f} | "
  f"두 추정 평균: {np.sqrt((((S.est_nb+S.p)/2-S.y)[m]**2).mean()):.3f}")
