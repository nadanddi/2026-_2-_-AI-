# -*- coding: utf-8 -*-
"""submission_17 temperature = 9th-round W40G-S (make_submission_v13_temp, unchanged) + a GRU sequence member used only on
warm hours of days that have not been cold so far:

    pred = P9 + 0.1 * g * h10 * (GRU - P9)
    P9   = 0.4 * BASE + (0.2 + 0.4(1-g)) * CODEX + 0.4 g * TABPFN          (W40G-S, round 9 = 0.5127)
    g    = clip((in_temp_now - 8) / 2, 0, 1)                               (NaN -> 1)
    h10  = clip((m_t - 10) / 2, 0, 1),  m_t = min of in_temp over hours 0..t of the same record day   (NaN -> 1)

GRU: input = the same greenhouse's last 48 hourly steps (current included, record order t) of the 14 usable inputs + hour
sin/cos; per-channel z-score from the training rows; NaN -> 0 plus one in_temp-missing flag channel.  GRU(17 -> 64, 1 layer)
-> Linear(64 -> 1), Adam lr 2e-3, 25 epochs, batch 256, MSE weighted by the round-5 row weights, seeds 1 and 2 averaged, CPU.
Training sequences are built in the MASK world (every test_X input NaN), test sequences from the full inputs up to the
current hour of the same greenhouse.

Local evidence (research, 집 클로드 2026-10-10/11, catalog 6.526-6.528): fresh layouts F/G/H/I -0.70..-0.76% all rows,
pass-2 -1.6..-2.0%; final pre-registered layout J -0.61..-0.63% (all 4 cells, both farms better, pass-2 -1.5%) but block
bootstrap P .029 > .025 -> NOT adopted; this file is a leaderboard HYPOTHESIS TEST (user decision 2026-10-11).

Run:  cd code && python make_submission_v17_temp.py ../output
"""
import json
import os
import platform
import sys
import time

import env  # noqa: F401  MUST be the first project import
import env_extra  # noqa: F401
import numpy as np
import pandas as pd

import common
import train_flags_v6 as TF
import make_submission_v13_temp as V13
from make_submission_v3 import sha256

USE = ["out_temp", "out_hum", "out_rad", "out_wspd", "in_temp", "in_hum", "in_co2", "act_vent", "act_shade", "act_thermal",
       "act_heating", "act_circfan", "act_co2", "act_fog"]
T_STEPS = 48
GRU_SEEDS = (1, 2)
W_GRU = 0.1
T_DAY = 10.0
HERE = os.path.dirname(os.path.abspath(__file__))
NAME = "temp_candidate_v17"


def gru_sequences(loader, row_ids, mask_test):
    loader = loader or V13.ORIG
    tX, _, sX = V13.masked_loader_from(loader) if mask_test else loader()
    a = pd.concat([tX, sX])
    a = a[a.farm.isin(common.TARGET_FARMS)].sort_values(["farm", "t"]).reset_index(drop=True)
    a["hs"], a["hc"] = np.sin(2 * np.pi * a.hour / 24), np.cos(2 * np.pi * a.hour / 24)
    F = a[USE + ["hs", "hc"]].to_numpy(np.float32)
    pos = {rid: i for i, rid in enumerate(a.row_id)}
    farm = a.farm.values
    S = np.full((len(row_ids), T_STEPS, F.shape[1]), np.nan, np.float32)
    for j, rid in enumerate(row_ids):
        i = pos[rid]; lo = max(0, i - T_STEPS + 1)
        seg = F[lo:i + 1]; seg = seg[farm[lo:i + 1] == farm[i]]
        S[j, T_STEPS - len(seg):] = seg
    return S


def fit_predict_gru(Str, ytr, wtr, Ste, seed):
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
            super().__init__(); s.g = torch.nn.GRU(d, 64, batch_first=True); s.o = torch.nn.Linear(64, 1)

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


def day_gate(test):
    t = test[["farm", "day", "hour", "in_temp"]].copy()
    m = t.sort_values(["farm", "day", "hour"]).groupby(["farm", "day"]).in_temp.cummin().reindex(t.index).to_numpy(float)
    return np.where(np.isnan(m), 1.0, np.clip((m - T_DAY) / 2.0, 0.0, 1.0))


def final_prediction(loader=None, pfn_samples=None, log=print):
    if pfn_samples is not None:
        V13.PFN_SAMPLES = tuple(pfn_samples)
    lab, test, ct, phc, sX = V13.build_frames(loader)
    w = TF.row_weights(lab, V13.W_FLAG, w_noisy=V13.W_NOISY)
    base, codex, pfn_all = V13.predict_members(lab, test, ct, phc, w, log=log)
    g = V13.gate(test.in_temp.values)
    p9 = V13.combine(base, codex, pfn_all.mean(0), g)
    t0 = time.time()
    Str = gru_sequences(loader, lab.row_id.values, mask_test=True)
    Ste = gru_sequences(loader, test.row_id.values, mask_test=False)
    gru = np.mean([fit_predict_gru(Str, lab.sub_temp.values, w, Ste, s) for s in GRU_SEEDS], axis=0)
    log("  gru done (%.0fs)" % (time.time() - t0))
    h = day_gate(test)
    pred = p9 + W_GRU * g * h * (gru - p9)
    return dict(lab=lab, test=test, sX=sX, ct=ct, phc=phc, w=w, base=base, codex=codex, pfn_all=pfn_all, g=g, h=h, p9=p9, gru=gru, pred=pred)


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "output")
    os.makedirs(out, exist_ok=True)
    path, npz, mpath = (os.path.join(out, NAME + e) for e in (".csv", "_members.npz", ".json"))
    for p in (path, npz, mpath):
        assert not os.path.exists(p), "refusing to overwrite %s" % p
    R = final_prediction(log=lambda s: print(s, flush=True))
    sX, test = R["sX"], R["test"]
    assert test.row_id.tolist() == sX.row_id.tolist()
    sub04 = pd.read_csv(os.path.join(HERE, "submissions", "submission_04.csv"))
    sub = pd.DataFrame({"row_id": sX.row_id.values, "sub_temp": R["pred"], "sub_ec": sub04.sub_ec.values})
    assert len(sub) == 1440 and sub.row_id.is_unique and np.isfinite(sub[["sub_temp", "sub_ec"]].to_numpy(float)).all()
    sub.to_csv(path, index=False, encoding="utf-8", float_format="%.6f")
    np.savez(npz, row_id=sX.row_id.values, base=R["base"], codex=R["codex"], pfn_samples=R["pfn_all"], gate=R["g"], day_gate=R["h"],
             gru=R["gru"], pred_9th_recipe=R["p9"], pred=R["pred"])
    d = R["pred"] - R["p9"]; g, h = R["g"], R["h"]
    print("\nwrote %s" % path)
    print("v17 - same-run 9th recipe: RMS %.4f, mean %+.4f, min %+.3f, max %+.3f | GRU used (g*h>0) on %.0f%% of rows"
          % (np.sqrt((d ** 2).mean()), d.mean(), d.min(), d.max(), 100 * (g * h > 0).mean()))
    for nm, m in (("in_temp<=8 (g=0)", g == 0), ("0<g<1", (g > 0) & (g < 1)), ("g=1", g == 1)):
        print("  %-16s rows %4d  mean change %+.4f  RMS %.4f" % (nm, m.sum(), d[m].mean(), np.sqrt((d[m] ** 2).mean())))
    man = dict(name=NAME, python=platform.python_version(), base_generator="make_submission_v13_temp.py (unchanged)",
               formula="P9 + 0.1*g*h10*(GRU-P9); P9 = W40G-S; g = clip((in_temp-8)/2,0,1); h10 = clip((daily running min in_temp - 10)/2,0,1)",
               gru=dict(steps=T_STEPS, inputs=USE + ["hour_sin", "hour_cos", "in_temp_missing_flag"], hidden=64, epochs=25, lr=2e-3, batch=256,
                        seeds=list(GRU_SEEDS), device="cpu", training_sequences="MASK world", test_sequences="full inputs <= current hour, same greenhouse"),
               inputs={n: sha256(os.path.join(common.DATA, n)) for n in ["train_X.csv", "train_y.csv", "test_X.csv"]},
               output_sha256=sha256(path), status="hypothesis test; pre-registered layout-J confirmation failed by P .029 > .025")
    with open(mpath, "w", encoding="utf-8") as f:
        json.dump(man, f, ensure_ascii=False, indent=2)
    print("wrote %s" % os.path.basename(mpath))


if __name__ == "__main__":
    main()
