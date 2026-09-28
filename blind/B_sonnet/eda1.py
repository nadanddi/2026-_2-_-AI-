# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, r"C:\Users\aozks\OneDrive\바탕 화면\2026_2학기_농업AI경진대회\blind")
import boot
import pandas as pd
import numpy as np
import os

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)

train_X = pd.read_csv(os.path.join(boot.DATA, "train_X.csv"))
train_y = pd.read_csv(os.path.join(boot.DATA, "train_y.csv"))
test_X = pd.read_csv(os.path.join(boot.DATA, "test_X.csv"))
sample_sub = pd.read_csv(os.path.join(boot.DATA, "sample_submission.csv"))

out = []
def p(s=""):
    out.append(str(s))

p(f"train_X shape: {train_X.shape}")
p(f"train_y shape: {train_y.shape}")
p(f"test_X shape: {test_X.shape}")
p(f"sample_sub shape: {sample_sub.shape}")
p()
p("train_X columns: " + ", ".join(train_X.columns))
p()

# parse row_id
def parse_rid(df):
    parts = df["row_id"].str.split("_", expand=True)
    df = df.copy()
    df["gh"] = parts[0]
    df["day"] = parts[1].astype(int)
    df["hour"] = parts[2].astype(int)
    return df

train_X_p = parse_rid(train_X)
test_X_p = parse_rid(test_X)

p(f"Number of unique greenhouses in train_X: {train_X_p['gh'].nunique()}")
p(f"Greenhouses in train_X: {sorted(train_X_p['gh'].unique())}")
p()
p(f"Number of unique greenhouses in test_X: {test_X_p['gh'].nunique()}")
p(f"Greenhouses in test_X: {sorted(test_X_p['gh'].unique())}")
p()

# test greenhouses in train?
test_ghs = set(test_X_p['gh'].unique())
train_ghs = set(train_X_p['gh'].unique())
p(f"Test greenhouses also in train_X: {test_ghs & train_ghs}")
p(f"Test greenhouses NOT in train_X: {test_ghs - train_ghs}")
p()

# per test greenhouse day range
for gh in sorted(test_ghs):
    sub = test_X_p[test_X_p['gh']==gh]
    p(f"test gh={gh}: n={len(sub)}, day range {sub['day'].min()}-{sub['day'].max()}, hour range {sub['hour'].min()}-{sub['hour'].max()}")
p()
for gh in sorted(test_ghs):
    sub = train_X_p[train_X_p['gh']==gh]
    p(f"train gh={gh} (if exists): n={len(sub)}, day range {sub['day'].min() if len(sub) else None}-{sub['day'].max() if len(sub) else None}")
p()

# row_id overlap between train_X and test_X
overlap_ids = set(train_X['row_id']) & set(test_X['row_id'])
p(f"row_id overlap between train_X and test_X: {len(overlap_ids)}")
p()

# train_X vs train_y row_id match
p(f"train_X row_id set == train_y row_id subset check")
p(f"train_y row_ids all in train_X: {set(train_y['row_id']).issubset(set(train_X['row_id']))}")
missing_y = set(train_X['row_id']) - set(train_y['row_id'])
p(f"train_X rows without any train_y entry: {len(missing_y)}")
p()

# NaN in train_y targets
p("train_y NaN counts:")
p(train_y.isna().sum())
p()

# missingness in train_X and test_X
p("train_X missing value counts (top):")
p(train_X.isna().sum())
p()
p("test_X missing value counts (top):")
p(test_X.isna().sum())
p()

with open("eda1_output.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(out))
print("done")
