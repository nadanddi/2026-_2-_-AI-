import sys, os
sys.path.insert(0, r"C:\work\farmai\집\클로드\research"); import env
sys.path.insert(0, r"C:\work\farmai\집\클로드\research\linker_v4")
from feat4 import *
import pandas as pd, numpy as np
t=pd.read_pickle(os.path.join(S3,"truth.pkl"))
pd.set_option('display.width',250); pd.set_option('display.max_columns',30)
print(t.iloc[100:115])
print(t.cert.describe()); print((t.cert>0.5).sum()); print(t.same_farm.value_counts(), t.sameG.value_counts())
d=pd.read_csv(os.path.join(env.LOCAL,"dmt3e_E0_days.csv")); print(d[d.rec.isin(['F13_217','F47_229','F47_231','F13_216','F47_211','F13_238','F47_132','F13_213'])])
print(N, TRAIN.sum())
names=np.array([f"{FARM[i]}_{NUM[i]:03d}" for i in range(N)])
print(names[:5], names[TRAIN][:5])
ev=[n for n in names[~TRAIN]]; print(len(ev), ev)
