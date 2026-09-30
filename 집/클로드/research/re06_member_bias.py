# -*- coding: utf-8 -*-
"""재분석 6 (2026-10-01, 집 클로드): EXT10·EXT12 외삽 편향이 어느 멤버에서 오는가 (진단).
멤버: MASK(기준 블렌드 res/ridge/nys, 시드 7·101 평균), CODEX(726·727), TabPFN v2.
날 오프셋과 heat_mean·in_temp_mean 스피어만, 멤버별 RMSE·편향. 결과 local/re06_member_bias.txt
"""
import env  # noqa
import os, numpy as np, pandas as pd
from scipy.stats import spearmanr
from harness import load
_, lab, _ = load()
z = np.load(env.LOCAL + "/temp_mask_v1_oof.npz", allow_pickle=True)
out = open(env.LOCAL + "/re06_member_bias.txt", "w", encoding="utf-8")
def p(s): print(s); out.write(s + "\n")
for s in ("EXT10", "EXT12", "DIAG10"):
    M = {"MASK": np.nanmean([z[f"{s}__MASK__7"], z[f"{s}__MASK__101"]], 0),
         "CODEX": np.nanmean([z[f"{s}__CODEX__726"], z[f"{s}__CODEX__727"]], 0),
         "TABPFN": np.load(env.LOCAL + f"/web_tabpfn_v2_temp_{s}.npy").mean(0)}
    ok = ~np.isnan(M["MASK"])
    d = lab.loc[ok, ["farm", "day", "hour", "in_temp", "act_heating", "sub_temp"]].copy()
    p(f"[{s}] 날 {d.groupby(['farm','day']).ngroups}")
    for k, v in M.items():
        d["e"] = v[ok] - d.sub_temp
        day = d.groupby(["farm", "day"]).agg(e=("e", "mean"), heat=("act_heating", "mean"), tin=("in_temp", "mean"))
        cold = d.in_temp < 8
        p(f"  {k:7s} RMSE {np.sqrt((d.e**2).mean()):.4f} 편향 {d.e.mean():+.3f} | in<8 편향 {d.e[cold].mean():+.3f} (n {cold.sum()}) "
          f"| 오프셋~heat ρ {spearmanr(day.heat, day.e)[0]:+.3f} ~in_temp ρ {spearmanr(day.tin, day.e)[0]:+.3f}")
    for a in M:
        for b in M:
            if a < b:
                ea, eb = M[a][ok] - d.sub_temp.values, M[b][ok] - d.sub_temp.values
                p(f"  오차 상관 {a}-{b}: {np.corrcoef(ea, eb)[0,1]:.3f}")
