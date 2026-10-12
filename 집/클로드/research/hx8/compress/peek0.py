import sys, os
sys.path.insert(0, r"C:\work\farmai\집\클로드\research"); import env
sys.path.insert(0, r"C:\work\farmai\집\클로드\research\linker_v4")
from feat4 import *
import pandas as pd, numpy as np
T = pd.read_pickle(os.path.join(S3, "truth.pkl")); print(T.columns.tolist()); print(T.head()); print(len(T))
D = pd.read_csv(os.path.join(env.LOCAL, "dmt3e_E0_days.csv")); print(D.head()); print(len(D), D.seg.value_counts())
print(env.DATA, env.LOCAL)
print(X.shape, IN_.shape, AC.shape, EC.shape, ST.shape)
O = pd.read_csv(r"C:\work\farmai\연구실\클로드\results\ec_rv1_preds_v1.csv")
print(O.groupby(["validator","tag"]).agg(n=("row_id","size"), rec=("row_id", lambda s: s.str[:7].nunique()), seeds=("seed","nunique")))
S = pd.read_csv(r"C:\work\farmai\제출\13회차_2026-10-12(나)\submission_22.csv"); print(S.head(), len(S))
# window layout
for f, lo, hi in [("F13", 210, 244), ("F47", 205, 236)]:
    idx = [i for i in range(N) if FARM[i] == f and lo <= NUM[i] <= hi]
    print(f, [(NUM[i], "T" if TRAIN[i] else "e", round(float(np.nanmean(EC[i])), 2) if TRAIN[i] else None) for i in idx])
