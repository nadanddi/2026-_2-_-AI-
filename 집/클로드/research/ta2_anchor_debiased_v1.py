# -*- coding: utf-8 -*-
"""TA2 (2026-10-11 집 클로드; TA1 critic 6.561 applied).  Linker v2 (hour-0 causal, scratch linker/cv_all400_choice_v2.csv).
Fixed before running:
  anchors: 1 step, prev labelled, p >= .8 (linker CV: record's 10-record chunk hidden, both neighbours deleted).
  bias:    c = median over PASS-1 anchored records of (A - y at 0 h)   (substrate keeps cooling after 23 h; TA1 bias +.37)
  gate:    anchor off if |dT_air| > 1.5 C (this record's 0 h in_temp - prev 23 h in_temp)
  step:    pred_h = q_h + w (1 - h/8) clip(A - c - q_0, -1, +1),  h = 0..7;  w chosen on PASS-1 DIAG10 rows (grid .1..1.0)
  q = round-11 temperature recipe (W40G-S + chain corrector).
Judgement (critic): only DIAG10 is clean (no anchored record's prev hidden in the same fold); G / EL1 reported.
  PASS iff on DIAG10 pass-2 rows all 4 cells better AND record-level (anchored pass-2 records, cell-mean squared-error change
  per record) bootstrap P(worse) < .025 AND sign test reported.  k = 1.  Rough days excluded.
Run:  cd 집/클로드/research && PYTHONPATH="" py -3.12 -u ta2_anchor_debiased_v1.py
"""
import os, sys, importlib.util
import env  # noqa: F401
import numpy as np, pandas as pd
from scipy.stats import binomtest
HERE = os.path.dirname(os.path.abspath(__file__)); sys.argv = ["x"]
s = importlib.util.spec_from_file_location("ta1", os.path.join(HERE, "ta1_linker_substrate_anchor_v1.py")); A1 = importlib.util.module_from_spec(s); s.loader.exec_module(A1)
LK = r"C:\Users\aozks\AppData\Local\Temp\claude\C--work-farmai\96fe1d15-2691-4cbd-a117-4a3e2d96bdf9\scratchpad\linker\cv_all400_choice_v2.csv"
rm = A1.rm


def anchors():
    A = pd.read_csv(LK)
    A = A[(A.prev_is_labelled.astype(bool)) & (A.p >= .8) & (A.steps_to_anchor == 1) & A.anchor_st23.notna() & (A.dT_air.abs() <= 1.5)]
    return {(r[:3], int(r[4:7])): float(a) for r, a in zip(A.rec, A.anchor_st23)}


def apply(M, q, AN, w, c):
    out = q.copy(); M = M.assign(q=q)
    for (f, d), idx in M.groupby(["farm", "day"]).groups.items():
        a = AN.get((f, d))
        if a is None:
            continue
        rows = M.loc[idx]; q0 = rows.loc[rows.hour == 0, "q"]
        if not len(q0):
            continue
        q0 = float(q0.iloc[0]); h = rows.hour.values; m = h <= 7
        j = np.array([M.index.get_loc(i) for i in rows.index])
        out[j[m]] = q[j[m]] + w * (1 - h[m] / 8) * np.clip(a - c - q0, -1, 1)
    return out


def main():
    AN = anchors(); print("anchors (gated): %d" % len(AN))
    ty = pd.read_csv(os.path.join(env.DATA, "train_y.csv")); y0 = ty[ty.row_id.str[8:10] == "00"].set_index(ty.row_id.str[:7][ty.row_id.str[8:10] == "00"]).sub_temp
    bias = [a - y0["%s_%03d" % k] for k, a in AN.items() if k[1] < 179 and "%s_%03d" % k in y0.index]
    c = float(np.median(bias)); print("bias c (pass-1, n %d) = %+.3f" % (len(bias), c))
    R = A1.frames(); M, cells = R["DIAG10"]; y = M.sub_temp.values; p1 = M.day.values < 179
    best = None
    for w in np.round(np.arange(.1, 1.01, .1), 2):
        sc = np.mean([rm((apply(M, q, AN, w, c) - y)[p1]) for q in cells]); best = (w, sc) if best is None or sc < best[1] else best
    w = best[0]; print("chosen w = %.1f (pass-1 DIAG10)" % w)
    rng = np.random.default_rng(20261124)
    for L in ("DIAG10", "G", "EL1", "EXT10", "EXT12"):
        M, cells = R[L]; y = M.sub_temp.values; m2 = M.day.values >= 179
        rel = []; per = []
        for q in cells:
            a = apply(M, q, AN, w, c); rel.append(rm((a - y)[m2]) / rm((q - y)[m2]) - 1)
            per.append(pd.Series(((a - y) ** 2 - (q - y) ** 2)[m2]).groupby((M.farm + "_" + M.day.astype(str)).values[m2]).sum())
        P = pd.concat(per, axis=1).mean(axis=1); P = P[P.abs() > 1e-12]
        line = "  %-6s pass-2 cells %s | anchored pass-2 records %d, improved %d" % (L, " ".join("%+.2f%%" % (100 * x) for x in rel), len(P), int((P < 0).sum()))
        if L == "DIAG10":
            idx = rng.integers(0, len(P), (20000, len(P))); pw = float((P.values[idx].sum(1) >= 0).mean())
            sg = binomtest(int((P < 0).sum()), len(P), .5, alternative="greater").pvalue
            ok = all(x < 0 for x in rel) and pw < .025
            line += " | record bootstrap P(worse) %.4f, sign-test p %.3f -> %s" % (pw, sg, "PASS" if ok else "FAIL")
        print(line)


if __name__ == "__main__":
    main()
