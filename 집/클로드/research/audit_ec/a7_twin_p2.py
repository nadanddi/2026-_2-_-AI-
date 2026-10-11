# -*- coding: utf-8 -*-
"""EC audit 7: eval records without pass-1 twins - do they have exact weather twins among training pass-2 records (any farm)?
If so, the model's training-side season for that twin vs the eval interpolated season."""
import sys, os
sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env  # noqa
PK = r"C:\Users\aozks\AppData\Local\Temp\claude\C--work-farmai\0038d811-b5c9-4a0d-913f-084fb31e14ab\scratchpad\build22\팜모니_정형데이터_재현패키지_22\EC"
sys.path.insert(0, PK)
import json, numpy as np, pandas as pd
from season import vectors, W
x = pd.read_csv(os.path.join(PK, "data", "train_X.csv")); tx = pd.read_csv(os.path.join(PK, "data", "test_X.csv"))
vec = vectors(pd.concat([x[["row_id"] + W], tx[["row_id"] + W]]))
n = json.load(open(os.path.join(PK, "season_transform.json"), encoding="utf-8"))["standardization"]
mu, sd = np.array(n["mu"]), np.array(n["sd"])
ts = pd.read_csv(os.path.join(PK, "training_day_to_season.csv")); ts = {(f, d): s for f, d, s in ts.itertuples(index=False)}
E = pd.read_csv("a1_eval_season.csv")
tk = [k for k in vec if k in ts and k[1] >= 179]
ref = np.stack([(vec[k] - mu[:, None]) / sd[:, None] for k in tk])
for f, d, s in E[E.n_twin == 0][["farm", "day", "season_interp"]].itertuples(index=False):
    qz = (vec[(f, d)] - mu[:, None]) / sd[:, None]
    dist = np.sqrt(np.nanmean((ref - qz) ** 2, axis=(1, 2))); j = np.argsort(dist)[:2]
    print(f, d, "interp %.1f | nearest training pass-2: %s dist %.3f season %.1f ; 2nd %s %.3f" % (s, tk[j[0]], dist[j[0]], ts[tk[j[0]]], tk[j[1]], dist[j[1]]))
