# -*- coding: utf-8 -*-
"""TA5 (2026-10-12 집 클로드): the SUBMITTED round-12 anchor rule with linker v7 (v5 + other-greenhouse TRAINING candidates) (cross-farm TRAINING calendar, user ruling
6.568; CV choices local/linker_v4/cv_all400_choice_v4.csv).  Fixed before running.  Constants NOT re-estimated: c = .36, w = 1.0,
clip +-1, gate |dT_air| <= 1.5, anchors = 1 step, prev labelled, p >= .8 (exactly as submitted in round 12).
Judgement as TA2 (clean layout DIAG10, pass-2 rows): PASS iff all 4 cells better than q (round-11 recipe) AND record-level
bootstrap P(worse) < .025; sign test, G / EL1 / EXT reported.  Also reported: v3 anchors (round-12 linker family) on the same
frame for comparison.  Caveat (linker agent): in CV the other farm's same-date records are always visible as training records,
at test time they may be evaluation records -> v4 CV hit rates are optimistic.
Run:  cd 집/클로드/research && PYTHONPATH="" py -3.12 -u ta3_anchor_linker_v4_v1.py
"""
import os, sys, importlib.util
import env  # noqa: F401
import numpy as np, pandas as pd
from scipy.stats import binomtest
HERE = os.path.dirname(os.path.abspath(__file__)); sys.argv = ["x"]
s = importlib.util.spec_from_file_location("ta2", os.path.join(HERE, "ta2_anchor_debiased_v1.py")); A2 = importlib.util.module_from_spec(s); s.loader.exec_module(A2)
C, W = .36, 1.0
FILES = {"v7": os.path.join(env.LOCAL, "linker_v7", "cv_all400_choice_v7.csv"), "v5": os.path.join(env.LOCAL, "linker_v5", "cv_all400_choice_v5.csv")}


def main():
    R = A2.A1.frames(); rng = np.random.default_rng(20261130)
    for tag, path in FILES.items():
        A2.LK = path; AN = A2.anchors(); print("\n==== linker %s: anchors (gated) %d" % (tag, len(AN)))
        for L in ("DIAG10", "G", "EL1", "EXT10", "EXT12"):
            M, cells = R[L]; y = M.sub_temp.values; m2 = M.day.values >= 179; rel = []; per = []
            for q in cells:
                a = A2.apply(M, q, AN, W, C); rel.append(A2.rm((a - y)[m2]) / A2.rm((q - y)[m2]) - 1)
                per.append(pd.Series(((a - y) ** 2 - (q - y) ** 2)[m2]).groupby((M.farm + "_" + M.day.astype(str)).values[m2]).sum())
            P = pd.concat(per, axis=1).mean(axis=1); P = P[P.abs() > 1e-12]
            line = "  %-6s pass-2 cells %s | anchored pass-2 records %d, improved %d" % (L, " ".join("%+.2f%%" % (100 * x) for x in rel), len(P), int((P < 0).sum()))
            if L == "DIAG10" and len(P):
                idx = rng.integers(0, len(P), (20000, len(P))); pw = float((P.values[idx].sum(1) >= 0).mean())
                sg = binomtest(int((P < 0).sum()), len(P), .5, alternative="greater").pvalue
                ok = all(x < 0 for x in rel) and pw < .025
                line += " | record bootstrap P(worse) %.4f, sign p %.3f -> %s" % (pw, sg, ("PASS" if ok else "FAIL") if tag == "v7" else "(reference)")
            print(line, flush=True)


if __name__ == "__main__":
    main()
