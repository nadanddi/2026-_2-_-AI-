# -*- coding: utf-8 -*-
"""ST2 (structure audit; 2026-10-03 집 클로드).  What is one record day?
Hypothesis H_pair: identical-weather neighbour pairs are the two 동 of one calendar
date, singletons are dates with one record (or with the twin elsewhere).
Test by weather continuity: |out_temp(d, 23h) - out_temp(next, 0h)|:
  pair-internal (d -> d+1 identical weather): if same date, 23h -> 0h of the SAME
  date = a wrap-around jump (large); following transitions (pair end -> next
  record) should be continuous (small) if records advance one date.
Also cross-farm: share of F13 record days whose 24h weather is identical to some
F47 record day (same site), and the record-index offset between them.
Uses train_X and test_X INPUTS for structure description only (no labels, nothing
feeds a model)."""
import env  # noqa: F401
import os
import numpy as np
import pandas as pd

V = ["out_temp", "out_hum", "out_rad", "out_wspd"]
A = pd.concat([pd.read_csv(os.path.join(env.DATA, f), usecols=["row_id"] + V).assign(src=s)
               for f, s in (("train_X.csv", "tr"), ("test_X.csv", "te"))])
A = A[A.row_id.str[:3].isin(["F13", "F47"])].copy()
A["farm"], A["day"], A["hour"] = A.row_id.str[:3], A.row_id.str[4:7].astype(int), A.row_id.str[8:10].astype(int)
src = A.groupby(["farm", "day"]).src.first()
for f in ("F13", "F47"):
    G = A[A.farm == f]
    W = G.pivot_table(index="day", columns="hour", values=V)
    days = sorted(W.index)
    print("\n%s all records %d (train %d, test %d), range %d-%d, missing numbers %s" % (
        f, len(days), (src[f] == "tr").sum(), (src[f] == "te").sum(), days[0], days[-1],
        sorted(set(range(days[0], days[-1] + 1)) - set(days))[:20]))
    same = {}
    for d in days:
        if d + 1 in W.index:
            a, b = W.loc[d].values, W.loc[d + 1].values
            ok = ~np.isnan(a) & ~np.isnan(b)
            same[d] = ok.sum() >= 80 and np.max(np.abs(a[ok] - b[ok])) < 1e-9
    ot = G.pivot_table(index="day", columns="hour", values="out_temp")
    jumps = {"pair-internal": [], "after pair": [], "single->next": []}
    role = {}
    k = 0
    while k < len(days):
        d = days[k]
        if same.get(d):
            role[d], role[d + 1] = "p1", "p2"; k += 2
        else:
            role[d] = "s"; k += 1
    for d in days:
        if d + 1 not in ot.index:
            continue
        j = abs(ot.loc[d, 23] - ot.loc[d + 1, 0])
        if role.get(d) == "p1":
            jumps["pair-internal"].append(j)
        elif role.get(d) == "p2":
            jumps["after pair"].append(j)
        else:
            jumps["single->next"].append(j)
    for k_, v in jumps.items():
        v = np.array(v, float); v = v[~np.isnan(v)]
        print("  23h->0h out_temp jump %-14s n %3d  median %.2f  mean %.2f" % (k_, len(v), np.median(v), v.mean()))
    nP = sum(1 for d in role if role[d] == "p1")
    nS = sum(1 for d in role if role[d] == "s")
    print("  records in pairs %d, singletons %d; pairs by pass: early %d late %d; singletons late %d" % (
        2 * nP, nS, sum(1 for d in role if role[d] == "p1" and d < 179), sum(1 for d in role if role[d] == "p1" and d >= 179),
        sum(1 for d in role if role[d] == "s" and d >= 179)))
    seq = "".join({"p1": "[", "p2": "]", "s": "."}[role[d]] for d in days)
    print("  role sequence (each char = record; [] pair, . single):")
    for i in range(0, len(seq), 80):
        print("   %3d %s" % (days[i], seq[i:i + 80]))
# cross-farm identical days
W13 = A[A.farm == "F13"].pivot_table(index="day", columns="hour", values=V)
W47 = A[A.farm == "F47"].pivot_table(index="day", columns="hour", values=V)
hits = []
for d in W13.index:
    a = W13.loc[d].values
    for e in range(d - 15, d + 16):
        if e in W47.index:
            b = W47.loc[e].values; ok = ~np.isnan(a) & ~np.isnan(b)
            if ok.sum() >= 80 and np.max(np.abs(a[ok] - b[ok])) < 1e-9:
                hits.append((d, e))
H = pd.DataFrame(hits, columns=["f13", "f47"])
print("\ncross-farm identical-weather record pairs (|offset| <= 15): %d, F13 days covered %d/%d" % (len(H), H.f13.nunique(), len(W13)))
print("offset (F47 day - F13 day) counts:", (H.f47 - H.f13).value_counts().sort_index().to_dict())
