# -*- coding: utf-8 -*-
"""재분석 17b (2026-10-01, 집 클로드): 큰 하루 오차가 연속 며칠로 뭉치는가 (에피소드성) — 탐색.
G_C2 DIAG10 하루 오프셋. |e|>0.7 날의 이웃(k=1,2) 날이 같은 부호로 |e|>0.5일 확률 vs 기준 비율(무작위 날).
또 자정 0시 오차와 하루 오프셋 관계, 시각별 오차가 하루 안에서 줄어드는가(초기 상태 기억 가설).
결과 local/re17b_episodes.txt
"""
import env  # noqa
import numpy as np, pandas as pd
from harness import load
_, lab, _ = load()
z = np.load(env.LOCAL + "/temp_mask_v1_oof.npz", allow_pickle=True)
t = lab.in_temp.values; g = np.where(np.isnan(t), 1, np.clip((t - 8) / 2, 0, 1)); s = "DIAG10"
base = np.nanmean([z[f"{s}__MASK__7"], z[f"{s}__MASK__101"]], 0)
cx = np.nanmean([z[f"{s}__CODEX__726"], z[f"{s}__CODEX__727"]], 0)
pfn = np.load(env.LOCAL + f"/web_tabpfn_v2_temp_{s}.npy").mean(0)
lab = lab.assign(e=(0.6 - 0.2 * (1 - g)) * base + (0.2 + 0.4 * (1 - g)) * cx + 0.2 * g * pfn - lab.sub_temp)
D = lab.groupby(["farm", "day"]).e.mean()
out = open(env.LOCAL + "/re17b_episodes.txt", "w", encoding="utf-8")
def p(x): print(x); out.write(x + "\n")
big = D[D.abs() > 0.7]
p(f"|e|>0.7 날 {len(big)}/{len(D)}")
rng = np.random.default_rng(0)
for k in (1, 2):
    hit, n = 0, 0
    for (f, d), e in big.items():
        for dd in (d - k, d + k):
            if (f, dd) in D.index:
                n += 1; hit += (np.sign(D[(f, dd)]) == np.sign(e)) and abs(D[(f, dd)]) > 0.5
    # 기준: 무작위 날 짝
    allk = [(f, d) for (f, d) in D.index]
    base_hit = np.mean([abs(D[x]) > 0.5 for x in allk]) / 2
    p(f"  k={k}: 큰 오차일 이웃이 같은 부호·|e|>0.5 인 비율 {hit}/{n} = {hit/n:.2f} (무작위 기준 ≈ {base_hit:.2f})")
# 시각별: 하루 오프셋 부호 기준으로 큰 날의 시각별 평균 오차 (초기 상태 기억이면 0시 큼 → 감소)
lab["e_day"] = lab.groupby(["farm", "day"]).e.transform("mean")
bg = lab[lab.e_day.abs() > 0.7].copy()
bg["se"] = bg.e * np.sign(bg.e_day)
p("큰 오차일(|e_day|>0.7) 시각별 평균 부호정렬 오차 (0시→23시):")
p("  " + " ".join(f"{v:.2f}" for v in bg.groupby("hour").se.mean().values))
p("전체 날 시각별 |e| 평균:")
p("  " + " ".join(f"{v:.2f}" for v in lab.groupby("hour").e.apply(lambda q: q.abs().mean()).values))
