# -*- coding: utf-8 -*-
"""submission_24 temperature, stage 1 (2026-10-14 집 클로드, LEADERBOARD HYPOTHESIS TEST; catalog 6.531, 6.532, 6.893):
P9 (9th-round W40G-S, temp_candidate_v13.csv, unchanged) + the 2-layer GRU sequence member 'V3' on warm hours of not-yet-cold days:

    pred = P9 + 0.1 * g * h10 * (V3 - P9)        (same blend rule as submission_17, only the GRU is the 2-layer V3)
    g   = clip((in_temp_now - 8) / 2, 0, 1)        (NaN -> 1)
    h10 = clip((running min of in_temp over hours 0..t of the record - 10) / 2, 0, 1)   (NaN -> 1)

V3: input = the same greenhouse's last 48 hourly steps (current included, record order) of the 14 usable inputs + hour sin/cos;
per-channel z-score from the training sequences; NaN -> 0 plus an in_temp-missing flag.  GRU(17 -> 128, 2 layers) -> Linear(128 -> 1),
Adam lr 2e-3, 25 epochs, batch 256, MSE weighted by the round-5 row weights, seeds 1 and 2 averaged, CPU, deterministic.
Training sequences in the MASK world (every test_X input NaN); evaluation sequences from the full inputs up to the current hour of the
same greenhouse (earlier records only) -> causal.
The output then goes through the unchanged round-13/14 chain (make_temp_v18_chain.py -> make_temp_v22_anchor.py -> make_temp_v23_tna8.py).
Local evidence: TS7 screen (F/G/H/I) and TT26 layout K PASS (9th base); TV3 re-judge on the round-14 configuration (v23 = chain + v7 +
TNA8 anchors): 28 cells (DIAG10/EL1/F/G/H/I/K) all better, -0.83..-1.19%, pooled P .0003, cold validators safe.
Run:  cd code && python make_temp_v24_v3base.py <temp_candidate_v13.csv> <out.csv>      (data from env.DATA / AGRI_DATA)
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

W_GRU = 0.1
GRU_SEEDS = (1, 2)


def fit_predict_v3(Str, ytr, wtr, Ste, seed):
    import torch
    torch.manual_seed(seed); np.random.seed(seed); torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    mu = np.nanmean(Str.reshape(-1, Str.shape[2]), 0); sd = np.nanstd(Str.reshape(-1, Str.shape[2]), 0) + 1e-6

    def prep(S):
        Z = (S - mu) / sd; m = np.isnan(Z[..., 4:5]).astype(np.float32); Z = np.nan_to_num(Z, nan=0.0)
        return torch.tensor(np.concatenate([Z, m], 2))
    Xt, Xv = prep(Str), prep(Ste); ym = float(ytr.mean())
    yt = torch.tensor((ytr - ym).astype(np.float32)); wt = torch.tensor(wtr.astype(np.float32))

    class Net(torch.nn.Module):
        def __init__(s, d):
            super().__init__(); s.g = torch.nn.GRU(d, 128, num_layers=2, batch_first=True); s.o = torch.nn.Linear(128, 1)

        def forward(s, x):
            h, _ = s.g(x); return s.o(h[:, -1]).squeeze(-1)
    net = Net(Xt.shape[2]); opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    gen = torch.Generator().manual_seed(seed); n = len(yt)
    for _ in range(25):
        perm = torch.randperm(n, generator=gen)
        for b in range(0, n, 256):
            ii = perm[b:b + 256]; opt.zero_grad()
            loss = (wt[ii] * (net(Xt[ii]) - yt[ii]) ** 2).sum() / wt[ii].sum(); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        return np.concatenate([net(Xv[b:b + 2048]).numpy() for b in range(0, len(Xv), 2048)]) + ym


def main():
    p9path, out = sys.argv[1:3]
    assert not os.path.exists(out), "refusing to overwrite %s" % out
    lab, test, ct, phc, sX = V13.build_frames(None)
    w = TF.row_weights(lab, V13.W_FLAG, w_noisy=V13.W_NOISY)
    P9 = pd.read_csv(p9path).set_index("row_id").sub_temp.reindex(test.row_id.values).to_numpy(float)
    assert np.isfinite(P9).all()
    g = V13.gate(test.in_temp.values); h = V17.day_gate(test)
    t0 = time.time()
    Str = V17.gru_sequences(None, lab.row_id.values, mask_test=True)
    Ste = V17.gru_sequences(None, test.row_id.values, mask_test=False)
    v3 = np.mean([fit_predict_v3(Str, lab.sub_temp.values, w, Ste, s) for s in GRU_SEEDS], axis=0)
    pred = P9 + W_GRU * g * h * (v3 - P9)
    S = pd.DataFrame({"row_id": test.row_id.values, "sub_temp": pred})
    order = pd.read_csv(p9path)[["row_id"]]
    S = order.merge(S, on="row_id", how="left"); assert S.sub_temp.notna().all() and len(S) == 1440
    S.to_csv(out, index=False, float_format="%.6f")
    d = S.sub_temp.values - pd.read_csv(p9path).sub_temp.values
    print("wrote %s | V3 %.0fs | change vs P9: RMS %.4f, mean %+.4f, min %+.3f, max %+.3f | V3 used (g*h>0) on %.0f%% of rows"
          % (out, time.time() - t0, np.sqrt((d ** 2).mean()), d.mean(), d.min(), d.max(), 100 * (g * h > 0).mean()))


if __name__ == "__main__":
    main()
