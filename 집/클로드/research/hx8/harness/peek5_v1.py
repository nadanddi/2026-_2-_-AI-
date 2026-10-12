# -*- coding: utf-8 -*-
# peek5 (HX3 준비): 평가 8기록 주변 번호 배열. 학습 전용 달력(E0) + 13회차 예측 + 학습 OOF q. DM11 day 는 서술 대조만.
import sys, os
sys.path.insert(0, r"C:\work\farmai\집\클로드\research"); import env
sys.path.insert(0, r"C:\work\farmai\집\클로드\research\linker_v4")
from feat4 import *
import numpy as np, pandas as pd
rid = lambda i: f"{FARM[i]}_{NUM[i]:03d}"; idx = {rid(i): i for i in range(N)}
E0 = pd.read_csv(os.path.join(env.LOCAL, "dmt3e_E0_days.csv")); E0d = dict(zip(E0.rec, zip(E0.seg, E0.day)))
DM = pd.read_csv(os.path.join(env.ROOT, "집", "클로드", "research", "ec0h", "rise", "dm11_calendar_v1.csv")); DAY = dict(zip(DM.rec, DM.day))
S = pd.read_csv(r"C:\work\farmai\제출\13회차_2026-10-12(나)\submission_22.csv"); S["rec"] = S.row_id.str[:7]; P = S.groupby("rec").sub_ec.mean()
O = pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "ec_rv1_preds_v1.csv"), usecols=["validator", "tag", "row_id", "sg"])
O = O[(O.tag == "CUR") & (O.validator == "DIAG10")]; O["rec"] = O.row_id.str[:7]; Q = O.groupby("rec").sg.mean()
for f, lo, hi in (("F13", 212, 242), ("F47", 208, 236)):
    for n in range(lo, hi):
        r = f"{f}_{n:03d}"
        if r not in idx: print(r, "MISSING", "dm11", DAY.get(r)); continue
        i = idx[r]
        if TRAIN[i]:
            print(r, "TRAIN E0", E0d.get(r), "dm11", DAY.get(r), "y %.3f y0 %.3f y23 %.3f oofq %s" % (np.nanmean(EC[i]), EC[i, 0], EC[i, 23], ("%.3f" % Q[r]) if r in Q.index else "-"))
        else:
            print(r, "EVAL  dm11", DAY.get(r), "pred13 %.3f" % P.get(r, np.nan))
