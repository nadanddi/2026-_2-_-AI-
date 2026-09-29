# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, r"C:\Users\aozks\OneDrive\바탕 화면\2026_2학기_농업AI경진대회\blind")
import boot
import pandas as pd
import numpy as np
import os

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)
pd.set_option("display.max_rows", 400)

train_X = pd.read_csv(os.path.join(boot.DATA, "train_X.csv"))
train_y = pd.read_csv(os.path.join(boot.DATA, "train_y.csv"))
test_X = pd.read_csv(os.path.join(boot.DATA, "test_X.csv"))

def parse_rid(df):
    parts = df["row_id"].str.split("_", expand=True)
    df = df.copy()
    df["gh"] = parts[0]
    df["day"] = parts[1].astype(int)
    df["hour"] = parts[2].astype(int)
    df["t"] = df["day"] * 24 + df["hour"]
    return df

train_X_p = parse_rid(train_X)
test_X_p = parse_rid(test_X)
train_y_p = parse_rid(train_y)

out = []
def p(s=""):
    out.append(str(s))

for gh in ["F13", "F47"]:
    tr = train_X_p[train_X_p.gh == gh]
    te = test_X_p[test_X_p.gh == gh]
    tr_days = sorted(tr.day.unique())
    te_days = sorted(te.day.unique())
    all_days_in_span = set(range(min(te_days), max(te_days)+1))
    accounted = set(tr_days) | set(te_days)
    missing_days = sorted(all_days_in_span - accounted)
    p(f"=== {gh} ===")
    p(f"test days full list: {te_days}")
    p(f"train days within test span: {[d for d in tr_days if min(te_days)<=d<=max(te_days)]}")
    p(f"days in span accounted neither train nor test: {missing_days}")
    p()

# merge train_y onto train_X_p to check which "within-range" train rows have labels
merged = train_X_p.merge(train_y_p[['row_id','sub_temp','sub_ec']], on='row_id', how='left')
for gh in ["F13","F47"]:
    sub = merged[merged.gh==gh]
    te = test_X_p[test_X_p.gh==gh]
    te_days=sorted(te.day.unique())
    within = sub[(sub.day>=min(te_days)) & (sub.day<=max(te_days))]
    p(f"{gh}: within-range train rows={len(within)}, sub_temp present={within.sub_temp.notna().sum()}, sub_ec present={within.sub_ec.notna().sum()}")

p()
# Look at actual time series of sub_temp and sub_ec around test boundaries for both gh
for gh in ["F13","F47"]:
    sub = merged[merged.gh==gh].sort_values('t')
    te = test_X_p[test_X_p.gh==gh]
    te_days=sorted(te.day.unique())
    test_start_t = min(te.t)
    test_end_t = max(te.t)
    p(f"=== {gh}: time series near test window [{test_start_t},{test_end_t}] ===")
    window = sub[(sub.t >= test_start_t-48) & (sub.t <= test_start_t+48)]
    p(window[['row_id','t','sub_temp','sub_ec']].to_string())
    p()

with open("eda3_output.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(out))
print("done")
