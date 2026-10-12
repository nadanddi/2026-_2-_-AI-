# -*- coding: utf-8 -*-
"""CH2b (CH2 사후 서술): 2단계 대상 28기록을 진짜 같은 동 사슬(진단용) 여부로 나눠 기록 RMSE 합 비교 (시드 7)."""
import sys, os; sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))); import env  # noqa
import re, numpy as np, pandas as pd
L = open("ch2_v1.log", encoding="utf-8", errors="ignore").read().splitlines()
i = [n for n, s in enumerate(L) if s.strip().startswith("fold") and "true_chain" in s][0]
rows = [re.split(r"\s+", s.strip()) for s in L[i + 2:] if s.strip()]
T = pd.DataFrame(rows, columns=["rec", "fold", "p1", "p2", "p_r", "p_p1", "true_chain", "A_pred", "A_lab", "y", "EA2", "C2P", "C2L", "rmse_EA2", "rmse_C2P", "rmse_C2L"])
for c in T.columns[4:]:
    if c != "true_chain": T[c] = T[c].astype(float)
for k, g in T.groupby("true_chain"):
    print(f"true_chain={k}: n {len(g)} | mean record RMSE  EA2 {g.rmse_EA2.mean():.3f}  C2P {g.rmse_C2P.mean():.3f}  C2L {g.rmse_C2L.mean():.3f} | improved C2L {(g.rmse_C2L < g.rmse_EA2).sum()}/{len(g)} | SSE-ish EA2 {(g.rmse_EA2**2).sum():.3f} vs C2L {(g.rmse_C2L**2).sum():.3f}")
