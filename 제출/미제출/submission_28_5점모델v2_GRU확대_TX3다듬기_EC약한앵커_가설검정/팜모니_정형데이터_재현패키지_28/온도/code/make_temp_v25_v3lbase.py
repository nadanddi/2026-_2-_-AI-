# -*- coding: utf-8 -*-
"""submission_25 temperature, stage 1 (2026-10-15 집 클로드, LEADERBOARD HYPOTHESIS TEST; '5-point / field-information' build; catalog 6.912, 6.918):
P9 (9th-round W40G-S, temp_candidate_v13.csv, unchanged) + the 2-layer GRU member 'V3L' on warm hours of not-yet-cold days:

    pred = P9 + 0.1 * g * h10 * (V3L - P9)       (same blend rule and gates as submission_24's V3)

V3L = V3 with one change: the 48-hour window is no longer 'the previous record numbers of the same farm' (a record-order regularity of the
competition files) but 'the SAME HOUSE's previous day (and the day before)', found by the v15R linker:
    window(row r, hour h) = last 48 steps of [prev2 day 0..23] + [prev1 day 0..23] + [r 0..h]     (missing parts -> NaN -> 0 + in_temp-missing flag)
    prev1 = linker prev0 of r if p0 >= .5;  prev2 = linker prev0 of prev1 if p0 >= .5.
  TRAINING sequences: links = v15r_mirror_pkg.py run on data whose test_X inputs are all NaN (MASK world -> training features never use
    test_X), sequences in the MASK world (every test_X input NaN), as for V3.
  EVALUATION sequences: links = v15r_eval_pkg.py picks (an evaluation record's link uses only its own hour-0 inputs, training records and
    same-farm earlier-numbered records); a linked day is used only if it is a TRAINING record or a SAME-FARM, EARLIER-NUMBERED evaluation record
    (otherwise that part of the window is empty).  Inputs of the evaluation record itself only up to the current hour -> causal.
GRU(17 -> 128, 2 layers) -> Linear(128 -> 1), Adam 2e-3, 25 epochs, batch 256, round-5 row weights, seeds 1 and 2, CPU, deterministic (as V3).
Local evidence (research, links from the v15R mirror table): v23 + V3L vs v23: DIAG10 -1.20%, EL1 -1.59%, F~I -1.19..-1.33% (pooled P .0001),
fresh layouts J/K/L -1.07..-1.29%; better than V3 (pooled P .03); cold validators within the safety rule.
Run:  cd code && python make_temp_v25_v3lbase.py <temp_candidate_v13.csv> <train_links_mirror_masked.csv> <eval_links_v15r.csv> <out.csv>
"""
import os
import sys
import time

import env  # noqa: F401  MUST be the first project import
import env_extra  # noqa: F401
import numpy as np
import pandas as pd

import train_flags_v6 as TF
import make_submission_v13_temp as V13
import make_submission_v17_temp as V17
from make_temp_v24_v3base import fit_predict_v3, W_GRU, GRU_SEEDS

P_MIN = .5


def day_table(tX, sX):
    a = pd.concat([tX, sX]); a = a[a.farm.isin(["F13", "F47"])].copy()
    a["rec"] = a.row_id.str[:7]; a["hs"], a["hc"] = np.sin(2 * np.pi * a.hour / 24), np.cos(2 * np.pi * a.hour / 24)
    cols = V17.USE + ["hs", "hc"]; D = {}
    for r, g in a.groupby("rec"):
        m = np.full((24, len(cols)), np.nan, np.float32); g = g.sort_values("hour"); m[g.hour.values.astype(int)] = g[cols].to_numpy(np.float32); D[r] = m
    return D, len(cols)


def sequences(row_ids, D, nc, link):
    S = np.full((len(row_ids), 48, nc), np.nan, np.float32); nan = np.full((24, nc), np.nan, np.float32)
    for j, rid in enumerate(row_ids):
        r, h = rid[:7], int(rid[8:10]); p1 = link.get(r); p2 = link.get(p1) if p1 else None
        seq = np.concatenate([D.get(p2, nan) if p2 else nan, D.get(p1, nan) if p1 else nan, D[r][:h + 1]]); S[j] = seq[-48:]
    return S


def main():
    p9path, ltr, lev, out = sys.argv[1:5]
    assert not os.path.exists(out), "refusing to overwrite %s" % out
    lab, test, ct, phc, sX = V13.build_frames(None)
    w = TF.row_weights(lab, V13.W_FLAG, w_noisy=V13.W_NOISY)
    P9 = pd.read_csv(p9path).set_index("row_id").sub_temp.reindex(test.row_id.values).to_numpy(float)
    assert np.isfinite(P9).all()
    g = V13.gate(test.in_temp.values); h = V17.day_gate(test)
    train_recs = set(lab.row_id.str[:7])
    LT = pd.read_csv(ltr); link_tr = {r: p for r, p, q in zip(LT.rec, LT.prev0, LT.p0) if isinstance(p, str) and q >= P_MIN and p != r}
    LE = pd.read_csv(lev); link_ev = dict(link_tr)
    def allowed(r, p):
        return p in train_recs or (p[:3] == r[:3] and int(p[4:7]) < int(r[4:7]))
    for r, p, q in zip(LE.rec, LE.prev0, LE.p0):
        if isinstance(p, str) and q >= P_MIN and p != r and allowed(r, p): link_ev[r] = p
    t0 = time.time()
    tXm, _, sXm = V13.masked_loader_from(V13.ORIG); Dm, nc = day_table(tXm, sXm)
    Str = sequences(lab.row_id.values, Dm, nc, link_tr)
    tXf, _, sXf = V13.ORIG(); Df, _ = day_table(tXf, sXf)
    evr = set(test.row_id.str[:7]); LKE = {r: link_ev[r] for r in evr if r in link_ev}
    # build evaluation link map so that sequences() follows r -> p1 -> p2 with p2 checked against r
    def seq_eval(row_ids):
        S = np.full((len(row_ids), 48, nc), np.nan, np.float32); nan = np.full((24, nc), np.nan, np.float32)
        for j, rid in enumerate(row_ids):
            r, hh = rid[:7], int(rid[8:10]); p1 = LKE.get(r); p2 = None
            if p1:
                c = link_tr.get(p1) if p1 in train_recs else link_ev.get(p1)
                if c and c != r and allowed(r, c): p2 = c
            seq = np.concatenate([Df.get(p2, nan) if p2 else nan, Df.get(p1, nan) if p1 else nan, Df[r][:hh + 1]]); S[j] = seq[-48:]
        return S
    Ste = seq_eval(test.row_id.values)
    v3l = np.mean([fit_predict_v3(Str, lab.sub_temp.values, w, Ste, s) for s in GRU_SEEDS], axis=0)
    pred = P9 + W_GRU * g * h * (v3l - P9)
    S = pd.DataFrame({"row_id": test.row_id.values, "sub_temp": pred})
    order = pd.read_csv(p9path)[["row_id"]]
    S = order.merge(S, on="row_id", how="left"); assert S.sub_temp.notna().all() and len(S) == 1440
    S.to_csv(out, index=False, float_format="%.6f")
    d = S.sub_temp.values - pd.read_csv(p9path).sub_temp.values
    print("wrote %s | V3L %.0fs | training links %d, evaluation records with a linked yesterday %d/60 | change vs P9: RMS %.4f, mean %+.4f | V3L used on %.0f%% of rows"
          % (out, time.time() - t0, len(link_tr), len(LKE), np.sqrt((d ** 2).mean()), d.mean(), 100 * (g * h > 0).mean()))


if __name__ == "__main__":
    main()
