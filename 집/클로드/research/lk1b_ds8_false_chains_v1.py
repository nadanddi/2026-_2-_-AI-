# -*- coding: utf-8 -*-
"""LK1B (2026-10-13 집 클로드; LK1 후속, 서술).  DS8 EV 발동 11기록의 사슬을 단계별로: 짝이 진짜 전날인가(진단 연결), 그 날의 실제 상태(하루 평균 EC)."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import env  # noqa: F401
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(HERE, "linker_v4")); from feat4 import EC, FARM, NUM, N, TRAIN, S3  # noqa: E402
rid = lambda i: f"{FARM[i]}_{NUM[i]:03d}"; idx = {rid(i): i for i in range(N)}
with np.errstate(all="ignore"): DM = np.nanmean(EC, 1)
TL = pd.read_pickle(os.path.join(S3, "truth.pkl")); CAL = pd.read_csv(os.path.join(HERE, "ec0h", "rise", "dm11_calendar_v1.csv")); D11 = dict(zip(CAL.rec, CAL.day))
CLEAN = {rid(int(b)): rid(int(a)) for a, b in zip(TL.a1, TL.b) if D11.get(rid(int(b)), np.nan) - D11.get(rid(int(a)), np.nan) == 1}
PE = pd.read_csv(os.path.join(env.LOCAL, "linker_v10", "v15r15_ev.csv")); PK = {r: (pv, p0, int(f)) for r, pv, p0, f in zip(PE.rec, PE.prev0, PE.p0, PE.fold)}
FO = pd.read_csv(os.path.join(HERE, "ev_validator", "ev_folds_v1.csv")); hidEV = {k: set(g.rec[g.role.isin(["val", "del"])]) for k, g in FO.groupby("fold")}
for r in ["F47_130", "F13_160", "F47_128", "F47_210", "F47_132", "F47_134", "F47_143", "F47_141", "F13_146", "F47_138", "F47_139"]:
    hs = hidEV[PK[r][2]]; cur, P, out = r, 1.0, [f"{r}(y {DM[idx[r]]:.2f})"]
    for _ in range(6):
        pv, p0, _f = PK[cur]; P *= p0
        if P < .3: out.append("| stop P<.3"); break
        ok = "ok" if CLEAN.get(cur) == pv else ("WRONG" if cur in CLEAN else "?")
        vis = TRAIN[idx[pv]] and pv not in hs
        out.append(f"<-[{ok} p{p0:.2f}] {pv}(y {DM[idx[pv]]:.2f}{' LABEL' if vis else ''})")
        if vis: break
        cur = pv
    print(" ".join(out))
