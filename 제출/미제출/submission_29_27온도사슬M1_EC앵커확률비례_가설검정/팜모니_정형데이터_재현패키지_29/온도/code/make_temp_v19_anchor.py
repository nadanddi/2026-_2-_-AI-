# -*- coding: utf-8 -*-
"""submission_19 temperature (2026-10-11 집 클로드, LEADERBOARD HYPOTHESIS TEST; catalog 6.559-6.562).
pred = round-11 temperature (temp_candidate_v18.csv: W40G-S + chain-aware early-hour correction)
     + linker anchor on hours 0-7 for evaluation records whose same-house previous date was found with confidence:
       pred_h = q_h + W * (1 - h/8) * clip(A - C - q_0, -1, +1),   h = 0..7
  A   = 23 h substrate temperature (training label) of the linker's chosen previous record (온도/linker, ap_v2.py output
        eval60_confident_1step_anchor_v2.csv: hour-0 choice, 1 step, previous record labelled, p >= .8)
  gate: anchor off if |dT_air| > 1.5 C (record's 0 h in_temp - previous record's 23 h in_temp)
  C = +0.36 C (median bias A - y_0h over pass-1 anchored training records, research TA2), W = 1.0 (chosen on pass-1 DIAG10 rows)
Local evidence (TA2, 6.562): clean layout DIAG10 pass-2 all 4 cells -0.44..-0.53 %, but record-level 8 / 15 better, P .096 -> FAIL;
submitted only as a hypothesis test of 'same-house previous-date labels carry temperature information on the evaluation set'.
Run:  python make_temp_v19_anchor.py ../temp_candidate_v18.csv ../linker/work/eval60_confident_1step_anchor_v2.csv ../temp_candidate_v19.csv
"""
import os, sys
import numpy as np, pandas as pd

C, W = 0.36, 1.0


def main():
    q_path, a_path, out = sys.argv[1:4]
    assert not os.path.exists(out), "refusing to overwrite " + out
    T = pd.read_csv(q_path); A = pd.read_csv(a_path)
    A = A[A.dT_air.abs() <= 1.5]; AN = dict(zip(A.row, A.anchor_st23))
    k = T.row_id.str[:7]; h = T.row_id.str[8:10].astype(int)
    q0 = T[h == 0].set_index(k[h == 0]).sub_temp
    new = T.sub_temp.values.copy()
    for i, (kk, hh) in enumerate(zip(k, h)):
        if kk in AN and hh <= 7:
            new[i] = T.sub_temp.values[i] + W * (1 - hh / 8) * np.clip(AN[kk] - C - q0[kk], -1, 1)
    S = T[["row_id"]].copy(); S["sub_temp"] = new
    assert len(S) == 1440 and np.isfinite(S.sub_temp).all()
    S.to_csv(out, index=False, float_format="%.6f")
    d = new - T.sub_temp.values
    print("wrote %s | anchored records %d, rows changed %d, RMS %.4f, mean %+.4f, min %+.3f, max %+.3f"
          % (out, len(AN), int((np.abs(d) > 1e-9).sum()), np.sqrt((d ** 2).mean()), d.mean(), d.min(), d.max()))


if __name__ == "__main__":
    main()
