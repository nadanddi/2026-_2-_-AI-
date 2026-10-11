# -*- coding: utf-8 -*-
"""EV (2026-10-12 집 클로드; user: "평가같이 앞뒤 하루씩 빼고 검증"): standard EVALUATION-GEOMETRY validator.
Evaluation facts: F13 chunks 185-189, 200-209, 220-229, 235-239; F47 = F13 numbers - 2 (183-187, 198-207, 218-227, 233-237);
the record right before and right after each chunk is DELETED (no inputs, no labels); both IDs' chunks are hidden together.
EV folds: F13 10-record chunks starting at s = lo, lo+10, ... ; paired with the F47 mirror chunk s-2 .. s+7 (validated together).
  role 'val'  : chunk records (scored if they are labelled training records)
  role 'del'  : F13 s-1, s+10 and F47 s-3, s+8 -> inputs AND labels must be removed for this fold
  all other records: training (labels usable, inputs usable).
Feature rule for val rows (as in evaluation): own inputs up to hour h, same-ID earlier records' inputs (incl. earlier val records of
the same ID), all training records; never the other ID's val records' inputs, never 'del' records.
Groups of chunks -> 8 folds (round robin over chunk index) to keep compute manageable.  Output: ev_folds_v1.csv (rec, farm, num,
fold, role) with one row per (fold, record) for val/del records; records not listed for a fold are training in that fold.
"""
import os, sys
sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env
import pandas as pd
Y = pd.read_csv(os.path.join(env.DATA, "train_y.csv"), usecols=["row_id"]); X = pd.read_csv(os.path.join(env.DATA, "train_X.csv"), usecols=["row_id"])
T = pd.read_csv(os.path.join(env.DATA, "test_X.csv"), usecols=["row_id"])
recs = lambda D: set(D.row_id.str[:7][D.row_id.str[:3].isin(["F13", "F47"])])
lab = recs(Y); allr = recs(X) | recs(T); test = recs(T)
num = lambda f: sorted(int(r[4:]) for r in allr if r[:3] == f)
lo, hi = min(num("F13")), max(num("F13"))
rows = []; k = 0
for s in range(lo, hi + 1, 10):
    fold = k % 8; k += 1
    spec = [("F13", range(s, s + 10), (s - 1, s + 10)), ("F47", range(s - 2, s + 8), (s - 3, s + 8))]
    for f, vr, dl in spec:
        for n in vr:
            r = f"{f}_{n:03d}"
            if r in lab: rows.append(dict(rec=r, farm=f, num=n, fold=fold, chunk=s, role="val"))
        for n in dl:
            r = f"{f}_{n:03d}"
            if r in allr and r not in test: rows.append(dict(rec=r, farm=f, num=n, fold=fold, chunk=s, role="del"))
F = pd.DataFrame(rows)
F.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ev_folds_v1.csv"), index=False)
V = F[F.role == "val"]
print("val records", len(V), "unique", V.rec.nunique(), "| pass-2 val", (V.num >= 179).sum(), "| del rows", (F.role == "del").sum())
print(V.groupby("fold").agg(n=("rec", "size"), p2=("num", lambda s: (s >= 179).sum())).T)
