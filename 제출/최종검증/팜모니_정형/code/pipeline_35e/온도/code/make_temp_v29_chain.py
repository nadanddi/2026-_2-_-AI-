# -*- coding: utf-8 -*-
"""submission_29 temperature, last step (2026-10-16 집 클로드, LEADERBOARD HYPOTHESIS TEST; catalog 6.979-6.985, organizer answer 2026-10-16 item 2).
Time-decaying PREDICTION CHAIN (rule M1):  for an evaluation record r that has NO labelled-yesterday anchor, if its FL hour-0 previous
record p (온도/linker_v7/ap_fl.py eval -> fl_eval_links.csv) is
    a same-farm, earlier-numbered EVALUATION record (unlabelled), p0 >= .8, |indoor air 23h(p) -> 0h(r)| <= 1.5 C,
then  pred_h(r) += w(h) * BETA[h] * clip(A - .36 - x0, -2, +2),   w(h) = .9 * exp(-h / 8),   h = 0..23
    A  = p's FINAL submitted 23 h temperature (this script's output for p: anchor + TX3 + its own chain; records processed in row_id order),
    x0 = r's hour-0 NX base prediction (temp_candidate_nx27.csv, before anchors).
  BETA, .36, clip 2 = the round-27 FL anchor constants (make_temp_v22_anchor.py, measured on training labels).  .9 and 8 h were fitted on
  validation chain rows (PCH3 M1) and checked on other layouts with eval-condition mirror links (PCH5, HYB1), on the round-27 assembly
  (ASM3: EXT10/EXT12/DIAG10 all 12 cells better, P .0003) and externally on 48 of 49 other greenhouses with true previous days (PCH4).
Uses only the same farm's earlier-numbered evaluation records' computed predictions (inputs of those records only) and fixed constants;
no later inputs, no other-farm evaluation inputs, nothing refitted on evaluation data.  The branch 'labelled yesterday -> label anchor,
unlabelled -> predicted chain' depends on whether the previous record has a TRAINING LABEL, not on which records exist in the evaluation set.
Run:  python make_temp_v29_chain.py <temp_in (round-27 temperature)> <nx27 base> <fl_eval_links.csv> <fl_eval_anchor.csv> <temp_out>
"""
import os, sys
import numpy as np, pandas as pd

C, CLIP, S1, TAU = 0.36, 2.0, 0.9, 8.0
BETA = np.array([0.7648, 0.6232, 0.5525, 0.4844, 0.4091, 0.3453, 0.3025, 0.2576, 0.1866, 0.1264, 0.1570, 0.1661, 0.1655, 0.1566,
                 0.1999, 0.2032, 0.2233, 0.2263, 0.2145, 0.2008, 0.1897, 0.1721, 0.1991, 0.2355])
W = S1 * np.exp(-np.arange(24) / TAU)


def main():
    tp, bp, lp, ap, out = sys.argv[1:6]
    assert not os.path.exists(out), "refusing to overwrite " + out
    T = pd.read_csv(tp); Bx = pd.read_csv(bp).set_index("row_id").sub_temp
    L = pd.read_csv(lp, encoding="utf-8-sig"); A = pd.read_csv(ap, encoding="utf-8-sig"); anchored = set(A.row)
    rec = T.row_id.str[:7]; hr = T.row_id.str[8:10].astype(int)
    val = dict(zip(T.row_id, T.sub_temp.values.astype(float)))
    chains = {}
    for r, p, q, lab, dt in zip(L.rec, L.prev0, L.p0, L.prev_is_labelled.astype(bool), L.dT_air):
        if r in anchored or lab or not isinstance(p, str) or q < .8 or not np.isfinite(dt) or abs(dt) > 1.5: continue
        if p[:3] == r[:3] and int(p[4:7]) < int(r[4:7]) and (p + "_23") in val: chains[r] = p
    order = sorted(set(rec), key=lambda r: (r[:3], int(r[4:7])))
    for r in order:
        if r not in chains: continue
        A23 = val[chains[r] + "_23"]; x0 = float(Bx[r + "_00"]); g = np.clip(A23 - C - x0, -CLIP, CLIP)
        for h in range(24):
            k = "%s_%02d" % (r, h)
            if k in val: val[k] += W[h] * BETA[h] * g
    S = T[["row_id"]].copy(); S["sub_temp"] = S.row_id.map(val)
    assert len(S) == 1440 and np.isfinite(S.sub_temp).all() and S.row_id.tolist() == T.row_id.tolist()
    S.to_csv(out, index=False, float_format="%.6f")
    d = S.sub_temp.values - T.sub_temp.values
    print("wrote %s | chained records %d %s | rows changed %d, RMS %.4f, mean %+.4f, min %+.3f, max %+.3f"
          % (out, len(chains), sorted(chains.items()), int((np.abs(d) > 1e-9).sum()), np.sqrt((d ** 2).mean()), d.mean(), d.min(), d.max()))


if __name__ == "__main__":
    main()
