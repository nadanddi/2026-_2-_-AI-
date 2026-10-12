# -*- coding: utf-8 -*-
"""INC0 (2026-10-13 집 클로드, hx8/increment): 자료 배치 확인(서술). 순수 2차 창 기록 목록, 학습 고EC 에피소드(진단 연결 a1->b), 하루 안 변화(23시-0시)와 자정 연속."""
import sys, os
sys.path.insert(0, r"C:\work\farmai\집\클로드\research"); sys.path.insert(0, r"C:\work\farmai\집\클로드\research\linker_v4")
import env
from feat4 import X, IN_, AC, EC, ST, FARM, NUM, N, TRAIN, S3
import numpy as np, pandas as pd
pd.set_option("display.width", 220); pd.set_option("display.max_rows", 300)
rid = lambda i: f"{FARM[i]}_{NUM[i]:03d}"; idx = {rid(i): i for i in range(N)}
CAL = pd.read_csv(os.path.join(env.LOCAL, "dmt3e_E0_days.csv")); DAY = dict(zip(CAL.rec, CAL.day))
TL = pd.read_pickle(os.path.join(S3, "truth.pkl"))
print(TL.cert.value_counts(dropna=False).head())
pre = {rid(int(b)): (rid(int(a)), c) for a, b, c in zip(TL.a1, TL.b, TL.cert)}
O = pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "ec_rv1_preds_v1.csv"))
O = O[O.tag == "CUR"]; O["rec"] = O.row_id.str[:7]
print(O.groupby("validator").agg(n=("sg", "size"), seeds=("seed", "nunique"), recs=("rec", "nunique")))
S = pd.read_csv(os.path.join(env.ROOT, u"제출", u"13회차_2026-10-12(나)", "submission_22.csv")); S["rec"] = S.row_id.str[:7]
s13 = S.groupby("rec").sub_ec.mean()
rows = []
for i in range(N):
    r = rid(i)
    rows.append(dict(rec=r, split="tr" if TRAIN[i] else "TE", day=DAY.get(r, np.nan), y=np.nanmean(EC[i]) if TRAIN[i] else np.nan,
                     e0=EC[i, 0], e23=EC[i, 23], p13=s13.get(r, np.nan), pre=pre.get(r, (None, None))[0], cert=pre.get(r, (None, None))[1]))
D = pd.DataFrame(rows).set_index("rec")
win = D[((D.index.str[:3] == "F13") & pd.Series(D.index.str[4:].astype(int), index=D.index).between(210, 245)) | ((D.index.str[:3] == "F47") & pd.Series(D.index.str[4:].astype(int), index=D.index).between(205, 245))]
print(win.round(3).to_string())
# high-EC chains in training
print("\n--- high-EC training records (y>=1) with predecessor info")
H = D[(D.split == "tr") & (D.y >= 1)].copy()
H["pre_y"] = [D.y.get(p, np.nan) if p else np.nan for p in H.pre]
H["pre_day"] = [D.day.get(p, np.nan) if p else np.nan for p in H.pre]
H["jump"] = [EC[idx[r], 0] - EC[idx[p], 23] if p in idx and TRAIN[idx[p]] else np.nan for r, p in zip(H.index, H.pre)]
H["W"] = H.e23 - H.e0
print(H.sort_values("day").round(3).to_string())
