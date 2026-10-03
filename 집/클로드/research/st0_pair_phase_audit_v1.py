# -*- coding: utf-8 -*-
"""ST0 (structure audit; 2026-10-03 집 클로드).  Assumption used everywhere:
record-day parity = source (동), and the two sources of one calendar date are
neighbouring record days.  Check from train_X inputs only (no labels):
for each farm, neighbouring record days (d, d+1) whose 24h outdoor weather
(out_temp/out_hum/out_rad/out_wspd) is identical (max abs diff < 1e-9 over all
non-missing hours, >= 20 hours) are 'same-date pairs'.  Report, along the record
index, whether pairs start on an even d (phase 0) or odd d (phase 1), runs of
phase, breaks, and days with no twin.  A phase flip means parity <-> source
mapping flips there."""
import env  # noqa: F401
import os
import numpy as np
import pandas as pd

V = ["out_temp", "out_hum", "out_rad", "out_wspd"]
X = pd.read_csv(os.path.join(env.DATA, "train_X.csv"), usecols=["row_id"] + V)
X = X[X.row_id.str[:3].isin(["F13", "F47"])].copy()
X["farm"], X["day"], X["hour"] = X.row_id.str[:3], X.row_id.str[4:7].astype(int), X.row_id.str[8:10].astype(int)
for f in ("F13", "F47"):
    W = X[X.farm == f].pivot_table(index="day", columns="hour", values=V)
    days = sorted(W.index)
    rows = []
    for d in days:
        if d + 1 not in W.index:
            continue
        a, b = W.loc[d].values, W.loc[d + 1].values
        ok = ~np.isnan(a) & ~np.isnan(b)
        same = ok.sum() >= 80 and np.nanmax(np.abs(a[ok] - b[ok])) < 1e-9
        rows.append((d, same))
    R = pd.DataFrame(rows, columns=["d", "same"])
    P = R[R.same]
    print("\n%s: train days %d (range %d-%d), neighbour pairs checked %d, identical-weather pairs %d" % (
        f, len(days), min(days), max(days), len(R), len(P)))
    P = P.assign(phase=P.d % 2, pas=(P.d >= 179).astype(int))
    print(P.groupby(["pas", "phase"]).size().rename("pairs").to_string())
    # runs of phase along record index
    seq = P[["d", "phase"]].values
    runs, start = [], 0
    for k in range(1, len(seq) + 1):
        if k == len(seq) or seq[k][1] != seq[start][1]:
            runs.append((int(seq[start][0]), int(seq[k - 1][0]), int(seq[start][1]), k - start))
            start = k
    print("phase runs (first d, last d, phase, n pairs):")
    for r_ in runs:
        print("  ", r_)
    # overlapping pairs: day both in (d-1,d) and (d,d+1) identical -> 3 identical days
    s = set(P.d)
    tri = [d for d in s if d + 1 in s]
    print("triples (d,d+1,d+2 identical):", sorted(tri)[:30], "... n", len(tri))
