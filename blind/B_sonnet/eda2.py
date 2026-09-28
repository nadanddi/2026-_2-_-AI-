# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, r"C:\Users\aozks\OneDrive\바탕 화면\2026_2학기_농업AI경진대회\blind")
import boot
import pandas as pd
import numpy as np
import os

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)
pd.set_option("display.max_rows", 200)

train_X = pd.read_csv(os.path.join(boot.DATA, "train_X.csv"))
train_y = pd.read_csv(os.path.join(boot.DATA, "train_y.csv"))
test_X = pd.read_csv(os.path.join(boot.DATA, "test_X.csv"))

def parse_rid(df):
    parts = df["row_id"].str.split("_", expand=True)
    df = df.copy()
    df["gh"] = parts[0]
    df["day"] = parts[1].astype(int)
    df["hour"] = parts[2].astype(int)
    return df

train_X_p = parse_rid(train_X)
test_X_p = parse_rid(test_X)
train_y_p = parse_rid(train_y)

out = []
def p(s=""):
    out.append(str(s))

# 1. per-greenhouse missingness for F13, F47 vs their test counterparts
cols = [c for c in train_X.columns if c not in ("row_id",)]
for gh in ["F13", "F47"]:
    sub_tr = train_X_p[train_X_p.gh == gh]
    sub_te = test_X_p[test_X_p.gh == gh]
    p(f"=== Greenhouse {gh} ===")
    p(f"train rows: {len(sub_tr)}, test rows: {len(sub_te)}")
    miss_tr = sub_tr[cols].isna().mean().round(3)
    miss_te = sub_te[cols].isna().mean().round(3)
    cmp = pd.DataFrame({"train_missing_frac": miss_tr, "test_missing_frac": miss_te})
    p(cmp.to_string())
    p()

# 2. day coverage for F13, F47 in train vs test
for gh in ["F13", "F47"]:
    sub_tr = train_X_p[train_X_p.gh == gh]
    sub_te = test_X_p[test_X_p.gh == gh]
    tr_days = sorted(sub_tr.day.unique())
    te_days = sorted(sub_te.day.unique())
    p(f"=== {gh} day coverage ===")
    p(f"train day min/max: {min(tr_days)}/{max(tr_days)}, n_unique_days={len(tr_days)}")
    p(f"test day min/max: {min(te_days)}/{max(te_days)}, n_unique_days={len(te_days)}")
    # gap analysis: train days immediately before test start
    test_start = min(te_days)
    test_end = max(te_days)
    days_before = [d for d in tr_days if d < test_start]
    days_after = [d for d in tr_days if d > test_end]
    days_within = [d for d in tr_days if test_start <= d <= test_end]
    p(f"train days before test_start({test_start}): count={len(days_before)}, last 10: {days_before[-10:]}")
    p(f"train days after test_end({test_end}): count={len(days_after)}, first 10: {days_after[:10]}")
    p(f"train days WITHIN test range [{test_start},{test_end}]: count={len(days_within)} -> {days_within[:20]}")
    p()

# 3. sub_ec availability overall and for F13/F47
p("=== sub_ec label availability ===")
n_total = len(train_y_p)
n_ec = train_y_p['sub_ec'].notna().sum()
p(f"Total train_y rows: {n_total}, sub_ec present: {n_ec} ({n_ec/n_total*100:.2f}%)")

ec_by_gh = train_y_p.groupby('gh')['sub_ec'].agg(['count', lambda s: s.notna().sum()])
ec_by_gh.columns = ['n_rows', 'n_ec_present']
ec_by_gh['frac'] = ec_by_gh['n_ec_present'] / ec_by_gh['n_rows']
p(ec_by_gh.sort_values('n_ec_present', ascending=False).to_string())
p()

for gh in ["F13", "F47"]:
    sub = train_y_p[train_y_p.gh == gh]
    n_ec_gh = sub['sub_ec'].notna().sum()
    p(f"{gh}: train_y rows={len(sub)}, sub_ec present={n_ec_gh}")
    if n_ec_gh > 0:
        p(sub[sub['sub_ec'].notna()][['day','hour','sub_ec']].describe().to_string())

with open("eda2_output.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(out))
print("done")
