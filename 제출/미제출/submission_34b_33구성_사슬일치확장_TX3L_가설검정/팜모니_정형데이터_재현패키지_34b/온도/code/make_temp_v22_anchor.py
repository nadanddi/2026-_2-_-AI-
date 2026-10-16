# -*- coding: utf-8 -*-
"""submission_22 temperature (2026-10-12 집 클로드, LEADERBOARD HYPOTHESIS TEST; catalog 6.583, 6.588, 6.589).
pred = round-11 temperature (temp_candidate_v18.csv) + linker v7 anchor with a MEASURED all-day decay:
    pred_h = q_h + BETA[h] * clip(A - C - q_0, -2, +2),  h = 0..23
  A = 23 h substrate label of the linker's confident previous record (온도/linker_v7 eval60_confident_1step_anchor_v7.csv:
      hour-0 decision, 1 step, previous record labelled, p >= .8), gate |dT_air| <= 1.5 C, C = .36 (as rounds 12 / 21).
  BETA = per-hour least-squares slope of the residual on the gap, measured on PASS-1 training records only (research
      ta8_anchor_decay_clip2_v1.py, linker v7 mirror-CV choices): fast midnight decay (thermal inertia) + an all-day remainder.
Local evidence (6.589): clean DIAG10 pass-2 all 4 cells -1.13..-1.18 % vs the round-12 rule, cold EXT10 -10.5 %, EXT12 -4.2 %,
  jackknife-1 same direction, but record-level P .093 > .025 -> FAIL on the local rule; submitted as a hypothesis test.
Run:  python make_temp_v22_anchor.py ../temp_candidate_v18.csv <eval60_confident_1step_anchor_v7.csv> ../temp_candidate_v22.csv
"""
import os, sys
import numpy as np, pandas as pd

C, CLIP = 0.36, 2.0
BETA = np.array([0.7648, 0.6232, 0.5525, 0.4844, 0.4091, 0.3453, 0.3025, 0.2576, 0.1866, 0.1264, 0.1570, 0.1661, 0.1655, 0.1566,
                 0.1999, 0.2032, 0.2233, 0.2263, 0.2145, 0.2008, 0.1897, 0.1721, 0.1991, 0.2355])


def main():
    q_path, a_path, out = sys.argv[1:4]
    assert not os.path.exists(out), "refusing to overwrite " + out
    T = pd.read_csv(q_path); A = pd.read_csv(a_path)
    A = A[A.dT_air.abs() <= 1.5]; AN = dict(zip(A.row, A.anchor_st23))
    k = T.row_id.str[:7]; h = T.row_id.str[8:10].astype(int)
    q0 = T[h == 0].set_index(k[h == 0]).sub_temp
    new = T.sub_temp.values.copy()
    for i, (kk, hh) in enumerate(zip(k, h)):
        if kk in AN:
            new[i] = T.sub_temp.values[i] + BETA[hh] * np.clip(AN[kk] - C - q0[kk], -CLIP, CLIP)
    S = T[["row_id"]].copy(); S["sub_temp"] = new
    assert len(S) == 1440 and np.isfinite(S.sub_temp).all()
    S.to_csv(out, index=False, float_format="%.6f")
    d = new - T.sub_temp.values
    print("wrote %s | anchored records %d, rows changed %d, RMS %.4f, mean %+.4f, min %+.3f, max %+.3f"
          % (out, len(AN), int((np.abs(d) > 1e-9).sum()), np.sqrt((d ** 2).mean()), d.mean(), d.min(), d.max()))


if __name__ == "__main__":
    main()
