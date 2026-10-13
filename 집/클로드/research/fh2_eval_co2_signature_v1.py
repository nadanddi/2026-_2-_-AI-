import os, sys
sys.path.insert(0, r"C:\work\farmai\집\클로드\research"); import env, pandas as pd, numpy as np
X = pd.read_csv(os.path.join(env.DATA, "test_X.csv")); X["rec"] = X.row_id.str[:7]; X["h"] = X.row_id.str[8:].astype(int)
T = pd.read_csv(os.path.join(env.DATA, "train_X.csv")); T = T[T.row_id.str[:3].isin(["F13","F47"])]; T["rec"] = T.row_id.str[:7]; T["h"] = T.row_id.str[8:].astype(int)
A = pd.concat([X, T])
d = A[(A.h >= 9) & (A.h <= 16)].groupby("rec")[["act_co2", "act_thermal", "act_shade", "act_vent", "in_temp", "out_temp"]].mean().round(1)
n = A[(A.h <= 5) | (A.h >= 20)].groupby("rec")[["act_heating", "in_temp"]].mean().round(1).add_suffix("_n")
D = d.join(n)
P12 = pd.read_csv(r"C:\work\farmai\제출\12회차_2026-10-11(나)\submission_19.csv"); P12["rec"] = P12.row_id.str[:7]; pm = P12.groupby("rec").sub_ec.mean().round(2)
D["pred12"] = pm
ev = D.loc[D.index.isin(X.rec.unique())]
print("eval records with heavy CO2 dosing (act_co2_d >= 25):"); print(ev[ev.act_co2 >= 25].sort_values("act_co2", ascending=False).to_string())
print("\nchain F13_229..238 & train neighbours:"); print(D.loc[[r for r in D.index if r.startswith("F13_2") and 226 <= int(r[4:]) <= 238]].to_string())
