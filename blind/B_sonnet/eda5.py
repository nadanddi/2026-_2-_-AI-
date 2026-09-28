# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, r"C:\Users\aozks\OneDrive\바탕 화면\2026_2학기_농업AI경진대회\blind")
import boot
import pandas as pd
import numpy as np
import os

train_X = pd.read_csv(os.path.join(boot.DATA, "train_X.csv"))
train_y = pd.read_csv(os.path.join(boot.DATA, "train_y.csv"))

def parse_rid(df):
    parts = df["row_id"].str.split("_", expand=True)
    df = df.copy()
    df["gh"] = parts[0]
    df["day"] = parts[1].astype(int)
    df["hour"] = parts[2].astype(int)
    df["t"] = df["day"] * 24 + df["hour"]
    return df

train_y_p = parse_rid(train_y)

out = []
def p(s=""):
    out.append(str(s))

for target in ["sub_temp", "sub_ec"]:
    for gh in ["F13", "F47"]:
        sub = train_y_p[(train_y_p.gh == gh) & train_y_p[target].notna()].sort_values('t')
        t = sub['t'].values
        y = sub[target].values
        idx = {tt: i for i, tt in enumerate(t)}
        p(f"=== {target} / {gh}: gap-k diff stats (n={len(t)}) ===")
        for k in [1, 6, 12, 24, 48, 72, 120]:
            diffs = []
            tset = set(t)
            for i, tt in enumerate(t):
                if (tt - k) in idx:
                    diffs.append(y[i] - y[idx[tt-k]])
            diffs = np.array(diffs)
            if len(diffs) > 0:
                p(f"  k={k:>4}h  n_pairs={len(diffs):>5}  mean_abs={np.abs(diffs).mean():.4f}  std={diffs.std():.4f}  RMSE_if_persistence={np.sqrt((diffs**2).mean()):.4f}")
        overall_std = y.std()
        p(f"  overall std of {target} for {gh}: {overall_std:.4f}")
        p()

with open("eda5_output.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(out))
print("done")
