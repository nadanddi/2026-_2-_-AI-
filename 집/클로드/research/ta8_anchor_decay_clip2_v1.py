# -*- coding: utf-8 -*-
"""TA8 (2026-10-12 집 클로드; TA7 critic 6.588 applied).  Fixed before running:
  gap    = A - .36 - q_0   (the submitted constant bias; beta re-measured with THIS gap definition)
  beta_h = LS slope of (y_h - q_h) on gap per hour h, measured on PASS-1 confident (p >= .8, |dT_air| <= 1.5) records of DIAG10
  pred_h = q_h + beta_h * clip(gap, -2, +2),  h = 0..23   (clip 2: tie with inf on pass-1, chosen to avoid reliance on one outlier)
  q = round-11 recipe; anchors = linker v7 mirror-CV choices; p >= .8, gate 1.5 (submitted).
Judgement vs the submitted rule (lin8, clip 1) on DIAG10 pass-2: all 4 cells better AND record-level bootstrap P(worse) < .025
AND the same direction (seed-mean pass-2 RMSE better) after removing the single record with the largest improvement (jackknife-1).
G / EL1 / EXT reported (not independent: same anchored records).
Run:  cd 집/클로드/research && PYTHONPATH="" py -3.12 -u ta8_anchor_decay_clip2_v1.py
"""
import os, sys, importlib.util
import env  # noqa: F401
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.argv = ["x"]
s = importlib.util.spec_from_file_location("ta7", os.path.join(HERE, "ta7_anchor_sweep_v1.py")); A7 = importlib.util.module_from_spec(s); s.loader.exec_module(A7)
A2 = A7.A2; rm = A2.rm


def main():
    L = pd.read_csv(A7.LK); L = L[L.prev_is_labelled.astype(bool) & L.anchor_st23.notna() & (L.steps_to_anchor == 1)].copy()
    R = A2.A1.frames(); M, cells = R["DIAG10"]; key, A = A7.build(M, L)
    sel = A[(A.p >= .8) & (A.dT_air.abs() <= 1.5)]; h = M.hour.values; y = M.sub_temp.values
    # re-measure beta with the const gap on pass-1
    num = np.zeros(24); den = np.zeros(24)
    for q in cells:
        q0 = pd.Series(q[h == 0], index=key[h == 0])
        for rec in sel.index.intersection(q0.index):
            if int(rec[4:7]) >= 179:
                continue
            g = sel.at[rec, "anchor_st23"] - .36 - q0[rec]; m = key == rec
            num[h[m]] += g * (y[m] - q[m]); den[h[m]] += g * g
    beta = num / den; print("beta_h (const gap, pass-1): " + " ".join("%d:%.2f" % (i, b) for i, b in enumerate(beta)))
    A7.BETA = beta
    rng = np.random.default_rng(20261207)
    for Lname in ("DIAG10", "G", "EL1", "EXT10", "EXT12"):
        M, cells = R[Lname]; key, A = A7.build(M, L); y = M.sub_temp.values; m2 = M.day.values >= 179; rel = []; per = []
        for q in cells:
            a = A7.correct(M, q, key, A, .8, "const", "meas", 2, 1.5); b = A7.correct(M, q, key, A, .8, "const", "lin8", 1, 1.5)
            rel.append(rm((a - y)[m2]) / rm((b - y)[m2]) - 1)
            per.append(pd.Series(((a - y) ** 2 - (b - y) ** 2)[m2]).groupby(key[m2]).sum())
        P = pd.concat(per, axis=1).mean(axis=1); P = P[P.abs() > 1e-12]
        line = "  %-6s TA8 vs submitted, pass-2 cells %s | records %d improved %d" % (Lname, " ".join("%+.2f%%" % (100 * x) for x in rel), len(P), int((P < 0).sum()))
        if Lname == "DIAG10" and len(P):
            idx = rng.integers(0, len(P), (20000, len(P))); pw = float((P.values[idx].sum(1) >= 0).mean())
            top = P.idxmin(); jk = P.drop(top).sum() < 0
            ok = all(x < 0 for x in rel) and pw < .025 and jk
            line += " | record P(worse) %.4f | top record %s %.2f (share %.0f%%), jackknife-1 sum %.2f -> %s" % (pw, top, P.min(), 100 * P.min() / P.sum(), P.drop(top).sum(), "PASS" if ok else "FAIL")
        print(line, flush=True)
    np.save(os.path.join(env.LOCAL, "linker_v7", "ta8_beta.npy"), beta)


if __name__ == "__main__":
    main()
