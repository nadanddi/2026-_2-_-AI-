# -*- coding: utf-8 -*-
"""TA6 (2026-10-12 집 클로드; temperature 'day state' bottleneck).  Temperature sub-agent (6.567): when the 0 h error is large
(|r0| > 1.5 C) it persists through the day (beta 12 h .41, 23 h .32), while the submitted anchor only acts on hours 0-7 with a
+-1 clip.  TA6 adds a day-long persistence term ONLY when the debiased anchor gap is large.  Fixed before running.
  base  = round-11 temperature recipe + submitted anchor rule with linker v7 (TA5m: c .36, w 1.0, clip 1, gate 1.5, hours 0-7)
  gap   = A - c - q_0  (A = linker v7 confident 1-step previous-record 23 h substrate label, mirror-CV choices)
  extra = b * clip(gap, -3, 3) on hours 8..23, only if |gap| > 1.5
  b     = least-squares slope of (y - base) on clip(gap) over hours 8..23 of PASS-1 anchored records with |gap| > 1.5 (DIAG10)
Judgement (k = 1): clean DIAG10 pass-2 rows, all 4 cells better than base AND record-level bootstrap P(worse) < .025;
G / EL1 / EXT reported.  Records affected and b reported.
Run:  cd 집/클로드/research && PYTHONPATH="" py -3.12 -u ta6_anchor_persistence_v1.py
"""
import os, sys, importlib.util
import env  # noqa: F401
import numpy as np, pandas as pd
from scipy.stats import binomtest
HERE = os.path.dirname(os.path.abspath(__file__)); sys.argv = ["x"]
s = importlib.util.spec_from_file_location("ta2", os.path.join(HERE, "ta2_anchor_debiased_v1.py")); A2 = importlib.util.module_from_spec(s); s.loader.exec_module(A2)
A2.LK = os.path.join(env.LOCAL, "linker_v7", "cv_all400_choice_v7m.csv")
C, W, GAP = .36, 1.0, 1.5


def gaps(M, q, AN):
    g = np.full(len(M), np.nan)
    M = M.assign(q=q)
    for (f, d), idx in M.groupby(["farm", "day"]).groups.items():
        a = AN.get((f, d))
        if a is None:
            continue
        rows = M.loc[idx]; q0 = rows.loc[rows.hour == 0, "q"]
        if len(q0):
            g[[M.index.get_loc(i) for i in rows.index]] = a - C - float(q0.iloc[0])
    return g


def main():
    AN = A2.anchors(); print("anchors (gated): %d" % len(AN))
    R = A2.A1.frames(); rng = np.random.default_rng(20261201)
    M, cells = R["DIAG10"]; y = M.sub_temp.values; p1 = M.day.values < 179; late = M.hour.values >= 8
    xs, rs = [], []
    for q in cells:
        base = A2.apply(M, q, AN, W, C); g = gaps(M, q, AN)
        m = p1 & late & np.isfinite(g) & (np.abs(g) > GAP)
        xs.append(np.clip(g[m], -3, 3)); rs.append((y - base)[m])
    xs, rs = np.concatenate(xs), np.concatenate(rs)
    b = float((xs * rs).sum() / (xs * xs).sum()) if len(xs) else 0.0
    print("b (pass-1, %d rows over 4 cells) = %.3f" % (len(xs), b))
    for L in ("DIAG10", "G", "EL1", "EXT10", "EXT12"):
        M, cells = R[L]; y = M.sub_temp.values; m2 = M.day.values >= 179; rel = []; per = []
        for q in cells:
            base = A2.apply(M, q, AN, W, C); g = gaps(M, q, AN)
            add = np.where(np.isfinite(g) & (np.abs(g) > GAP) & (M.hour.values >= 8), b * np.clip(np.nan_to_num(g), -3, 3), 0.0)
            new = base + add
            rel.append(A2.rm((new - y)[m2]) / A2.rm((base - y)[m2]) - 1)
            per.append(pd.Series(((new - y) ** 2 - (base - y) ** 2)[m2]).groupby((M.farm + "_" + M.day.astype(str)).values[m2]).sum())
        P = pd.concat(per, axis=1).mean(axis=1); P = P[P.abs() > 1e-12]
        line = "  %-6s pass-2 cells %s | affected pass-2 records %d, improved %d" % (L, " ".join("%+.2f%%" % (100 * x) for x in rel), len(P), int((P < 0).sum()))
        if L == "DIAG10" and len(P):
            idx = rng.integers(0, len(P), (20000, len(P))); pw = float((P.values[idx].sum(1) >= 0).mean())
            ok = all(x < 0 for x in rel) and pw < .025
            line += " | record P(worse) %.4f, sign p %.3f -> %s | %s" % (pw, binomtest(int((P < 0).sum()), len(P), .5, alternative="greater").pvalue, "PASS" if ok else "FAIL", P.round(2).to_dict())
        print(line, flush=True)


if __name__ == "__main__":
    main()
