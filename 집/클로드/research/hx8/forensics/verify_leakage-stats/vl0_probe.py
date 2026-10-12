# -*- coding: utf-8 -*-
import sys, os
sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env
sys.path.insert(0, r"C:\work\farmai\집\클로드\research\linker_v4")
import numpy as np, pandas as pd
from feat4 import X, IN_, AC, EC, ST, FARM, NUM, N, TRAIN, S3
rid = np.array([f"{FARM[i]}_{NUM[i]:03d}" for i in range(N)]); ix = {r: i for i, r in enumerate(rid)}
tr = np.where(TRAIN)[0]
print("N", N, "train", len(tr), "X shape", X.shape, "nan in train X recs", int(np.isnan(X[tr]).any(axis=(1,2)).sum()))
for r in ["F47_230","F47_231","F47_132","F47_133","F13_128","F13_129","F13_233","F47_229"]:
    i = ix[r]; print(r, "train", bool(TRAIN[i]), "EC0 %.3f EC23 %.3f mean %.3f" % (EC[i,0], EC[i,23], np.nanmean(EC[i])) if TRAIN[i] else "", "nanX", int(np.isnan(X[i]).sum()))
for a, b in [("F47_132","F47_133"),("F47_132","F13_128"),("F47_230","F47_231"),("F13_128","F13_129")]:
    print(a, b, "max|dX|", np.nanmax(np.abs(X[ix[a]]-X[ix[b]])))
t = pd.read_pickle(S3 + r"\truth.pkl")
for r in ["F47_132","F47_133","F13_128","F13_129"]:
    s = t[t.b == ix[r]]
    for _, q in s.iterrows(): print("b", r, "a1", rid[int(q.a1)], "a2", rid[int(q.a2)], "c1 %.3f c2 %.3f" % (q.c1, q.c2))
for r in ["F47_230","F47_231"]:
    s = t[t.a1 == ix[r]]
    for _, q in s.iterrows(): print("a1", r, "-> b", rid[int(q.b)], "c1 %.3f c2 %.3f" % (q.c1, q.c2))
