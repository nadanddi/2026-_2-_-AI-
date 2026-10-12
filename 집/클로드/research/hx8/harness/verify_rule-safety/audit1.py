# -*- coding: utf-8 -*-
# 규정 감사(읽기 전용 점검). DM11 은 감사 대조로만 읽음.
import sys, os
sys.path.insert(0, r"C:\work\farmai\집\클로드\research"); import env
sys.path.insert(0, r"C:\work\farmai\집\클로드\research\linker_v4")
from feat4 import X, EC, FARM, NUM, N, TRAIN
import numpy as np, pandas as pd
rid = lambda i: f"{FARM[i]}_{NUM[i]:03d}"; idx = {rid(i): i for i in range(N)}
CAL = pd.read_csv(os.path.join(env.LOCAL, "dmt3e_E0_days.csv"))
print("cal rows", len(CAL), "cols", CAL.columns.tolist())
print("cal recs not in idx", [r for r in CAL.rec if r not in idx][:5])
print("cal recs that are EVAL:", [r for r in CAL.rec if r in idx and not TRAIN[idx[r]]])
SD = {r: (int(s), int(d)) for r, s, d in zip(CAL.rec, CAL.seg, CAL.day)}
DM = pd.read_csv(os.path.join(env.ROOT, "집", "클로드", "research", "ec0h", "rise", "dm11_calendar_v1.csv")); DAY = dict(zip(DM.rec, DM.day))
for f, lo, hi in (("F13", 214, 240), ("F47", 209, 234)):
    s = []
    for n in range(lo, hi):
        r = f"{f}_{n:03d}"
        if r not in idx: s.append(f"{n}:X(dm{DAY.get(r)})"); continue
        i = idx[r]
        s.append(f"{n}:{'T' if TRAIN[i] else 'E'}{SD.get(r,'')}(dm{DAY.get(r)})")
    print(f, " ".join(s))
# F13_237 same-date: full-day vs causal prefix
Xf = X.reshape(N, -1); TR = np.where(TRAIN)[0]
i = idx["F13_237"]
full = (np.abs(Xf[TR] - Xf[i]) < 1e-6).mean(1)
print("F13_237 full-day >0.7 matches:", [(rid(j), round(float(np.nanmean(EC[j])),3)) for j in TR[full > 0.7]])
for h in (0, 1, 3, 6, 12, 23):
    pre = (np.abs(X[TR, :h+1, :] - X[i, :h+1, :]) < 1e-6).reshape(len(TR), -1).mean(1)
    m = TR[pre >= 0.999]
    print(f"  prefix 0..{h}h exact-match training recs: {len(m)}", [rid(j) for j in m][:8])
# NaN check of X for F13_237 hour 0
print("F13_237 X[0]", X[i, 0])
S13 = pd.read_csv(r"C:\work\farmai\제출\13회차_2026-10-12(나)\submission_22.csv"); S13["rec"] = S13.row_id.str[:7]; P13 = S13.groupby("rec").sub_ec.mean()
print({r: round(P13[r], 3) for r in ["F13_221","F13_223","F13_225","F13_237","F47_221","F47_223","F47_225","F47_227"]})
print("anchors F13_217 23h %.3f, F47_229 0h %.3f" % (EC[idx["F13_217"], 23], EC[idx["F47_229"], 0]))
print("E4p F47_227 = %.3f" % (P13["F47_227"] + 0.85 * (EC[idx["F47_229"], 0] - P13["F47_227"])))
