# -*- coding: utf-8 -*-
"""재분석 10 (2026-10-01, 집 클로드): 구간 오프셋 보정의 상한 (진단, 오라클).
G_C2 DIAG10 OOF 잔차에서 구간 평균을 빼면 RMSE가 얼마까지 내려가는가.
구간: 온실 전체(2), 온실×평가형 블록(5·10·10·5일 연속 → 8), 온실×날(60 상당).
평가형 블록은 학습 400일을 온실별로 30일씩 잘라 5/10/10/5로 나눈 것(여러 창 평균).
결과 local/re10_segment_ceiling.txt
"""
import env  # noqa
import os, numpy as np, pandas as pd
from harness import load
_, lab, _ = load()
z = np.load(env.LOCAL + "/temp_mask_v1_oof.npz", allow_pickle=True)
t = lab.in_temp.values; g = np.where(np.isnan(t), 1, np.clip((t - 8) / 2, 0, 1))
s = "DIAG10"
base = np.nanmean([z[f"{s}__MASK__7"], z[f"{s}__MASK__101"]], 0)
cx = np.nanmean([z[f"{s}__CODEX__726"], z[f"{s}__CODEX__727"]], 0)
pfn = np.load(env.LOCAL + f"/web_tabpfn_v2_temp_{s}.npy").mean(0)
lab = lab.assign(e=(0.6 - 0.2 * (1 - g)) * base + (0.2 + 0.4 * (1 - g)) * cx + 0.2 * g * pfn - lab.sub_temp)
out = open(env.LOCAL + "/re10_segment_ceiling.txt", "w", encoding="utf-8")
def p(x): print(x); out.write(x + "\n")
res = {k: [] for k in ["원래", "온실 전체(2)", "온실×블록(8)", "온실×날(60)"]}
for f0 in range(0, 170, 10):                      # 30일 창을 여러 위치에서
    parts = []
    for fm in ("F13", "F47"):
        days = sorted(lab[lab.farm == fm].day.unique())[f0:f0 + 30]
        if len(days) < 30: continue
        d = lab[(lab.farm == fm) & lab.day.isin(days)].copy()
        blk = np.repeat([0, 1, 2, 3], [5, 10, 10, 5])
        d["blk"] = d.day.map(dict(zip(days, blk)))
        parts.append(d)
    if len(parts) < 2: continue
    d = pd.concat(parts)
    r = lambda e: np.sqrt((e ** 2).mean())
    res["원래"].append(r(d.e))
    res["온실 전체(2)"].append(r(d.e - d.groupby("farm").e.transform("mean")))
    res["온실×블록(8)"].append(r(d.e - d.groupby(["farm", "blk"]).e.transform("mean")))
    res["온실×날(60)"].append(r(d.e - d.groupby(["farm", "day"]).e.transform("mean")))
p(f"30일×2온실 창 {len(res['원래'])}개, G_C2 DIAG10 잔차 (평균 [최소, 최대])")
b = np.mean(res["원래"])
for k, v in res.items():
    p(f"  {k:14s} RMSE {np.mean(v):.4f} [{np.min(v):.3f},{np.max(v):.3f}]  원래 대비 {100*(np.mean(v)/b-1):+.1f}%  → 평가 환산 ≈ {0.5251*np.mean(v)/b:.3f}")
