# -*- coding: utf-8 -*-
"""GRU training cache for the rule checks (2026-10-15 집 클로드).  fit_predict_v3_cached = make_temp_v24_v3base.fit_predict_v3 (same code, same
seeds, same deterministic CPU training) + an optional cache of the TRAINED weights keyed by a SHA-256 of (training sequences, targets, weights, seed).
The training sequences are built from the MASK world (every test_X input NaN), so they cannot depend on evaluation inputs; if a check perturbs
evaluation inputs and the training sequences change anyway (= a leak), the key changes and the model is retrained -> the checks stay strict.
Enabled only when the environment variable GRU_CACHE names a folder; without it this is exactly fit_predict_v3."""
import os, hashlib
import numpy as np


def fit_predict_v3_cached(Str, ytr, wtr, Ste, seed):
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
    net = Net(Xt.shape[2]); cache = os.environ.get("GRU_CACHE"); path = None
    if cache:
        hsh = hashlib.sha256(); [hsh.update(np.ascontiguousarray(a).tobytes()) for a in (Str, ytr, wtr)]; hsh.update(str(seed).encode())
        path = os.path.join(cache, "gru_%s.pt" % hsh.hexdigest()[:32])
    if path and os.path.exists(path):
        net.load_state_dict(torch.load(path)); print("GRU cache HIT", os.path.basename(path), flush=True)
    else:
        opt = torch.optim.Adam(net.parameters(), lr=2e-3)
        gen = torch.Generator().manual_seed(seed); n = len(yt)
        for _ in range(25):
            perm = torch.randperm(n, generator=gen)
            for b in range(0, n, 256):
                ii = perm[b:b + 256]; opt.zero_grad()
                loss = (wt[ii] * (net(Xt[ii]) - yt[ii]) ** 2).sum() / wt[ii].sum(); loss.backward(); opt.step()
        if path:
            os.makedirs(cache, exist_ok=True); tmp = path + ".tmp%d" % os.getpid(); torch.save(net.state_dict(), tmp); os.replace(tmp, path)
            print("GRU cache MISS -> trained and saved", os.path.basename(path), flush=True)
    net.eval()
    with torch.no_grad():
        return np.concatenate([net(Xv[b:b + 2048]).numpy() for b in range(0, len(Xv), 2048)]) + ym
