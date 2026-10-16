# -*- coding: utf-8 -*-
"""submission_28 temperature stage (v28b = make_temp_v28_v3lf.py with the GRU training cache of gru_cache_v28.py for the rule checks; identical output) (2026-10-15 집 클로드; catalog 6.935 OS1 v2 C1, 6.938 ASM1).
V3LF member = make_temp_v26_v3lbase.py's 48-hour same-house GRU, with the links of the FIELD linker (ap_fl.py: training links fl_train_links.csv
from the MASK world, evaluation links fl_eval_links.csv), blended with the ORIGINAL V3 gates at weight 0.30 (CB3-T, catalog 6.954):
    pred = P + 0.30 * g * h * (V3LF - P),   g = clip((in_temp - 8)/2, 0, 1),  h = clip((min in_temp of the day so far - 10)/2, 0, 1)  (missing -> 1)
  (grid of weight .10-.30 x gates 8/10, 9/11, 10/12, chosen on layout N, judged on CK2 + EL1 (-1.0 / -1.6 %, P .0013) with cold-extrapolation safety EXT10/EXT12 passed).
P = the round-27 temperature BEFORE anchors (NX base, temp_candidate_nx27.csv).  Anchors (make_temp_v22_anchor.py) and TX3 follow this stage.
Run:  cd code && python make_temp_v28_v3lf.py <temp_candidate_nx27.csv> <fl_train_links.csv> <fl_eval_links.csv> <out.csv>
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
from make_temp_v24_v3base import W_GRU, GRU_SEEDS
from gru_cache_v28 import fit_predict_v3_cached as fit_predict_v3

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
    t_ = np.asarray(test.in_temp.values, float); g = np.where(np.isnan(t_), 1.0, np.clip((t_ - 8.0) / 2.0, 0.0, 1.0))
    tt = test[['farm', 'day', 'hour', 'in_temp']].copy(); mm = tt.sort_values(['farm', 'day', 'hour']).groupby(['farm', 'day']).in_temp.cummin().reindex(tt.index).to_numpy(float)
    h = np.where(np.isnan(mm), 1.0, np.clip((mm - 10.0) / 2.0, 0.0, 1.0))
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
    wl = np.full(len(test), 0.30)
    pred = P9 + wl * g * h * (v3l - P9)
    S = pd.DataFrame({"row_id": test.row_id.values, "sub_temp": pred})
    order = pd.read_csv(p9path)[["row_id"]]
    S = order.merge(S, on="row_id", how="left"); assert S.sub_temp.notna().all() and len(S) == 1440
    S.to_csv(out, index=False, float_format="%.6f")
    d = S.sub_temp.values - pd.read_csv(p9path).sub_temp.values
    print("wrote %s | V3L %.0fs | training links %d, evaluation records with a linked yesterday %d/60 | change vs P9: RMS %.4f, mean %+.4f | V3L used on %.0f%% of rows"
          % (out, time.time() - t0, len(link_tr), len(LKE), np.sqrt((d ** 2).mean()), d.mean(), 100 * (g * h > 0).mean()))


if __name__ == "__main__":
    main()
