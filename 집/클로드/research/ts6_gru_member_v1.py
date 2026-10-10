# -*- coding: utf-8 -*-
"""TS6: a SEQUENCE neural member (GRU over the last 48 h of raw inputs) added at 10% (2026-10-10 집 클로드, user: "걔네가
생각했을 만한 거 전부 다 해봐").  Never tried for temperature.  Fixed before running.
Inputs per row: the same greenhouse's last 48 hourly steps (current included, record order t) of 14 usable inputs + hour
  sin/cos, in the MASK world (test_X inputs NaN, as for every training-row feature); per-channel z-score from the fold's
  training rows; NaN -> 0 plus one missing-flag channel.  Target: sub_temp at the last step.
Model: GRU(16 -> 64, 1 layer) -> Linear(64 -> 1); Adam lr 2e-3, 25 epochs, batch 256, MSE weighted by the row weights wb,
  seeds 1 / 2 averaged; GPU.
Candidate: 0.9 * W40G-S + 0.1 * GRU  (k = 1, alpha .025).  Scope ALL -> primary DIAG10 AND EXT10: both BASE seeds x both PFN
  families better, block bootstrap P(worse) < .025 on both; EXT12/EL1 fail iff seed-mean worse and share(better) < .025.
  Rough days excluded from scoring.  (Fresh-layout guard skipped for cost; a pass would need a fresh-layout confirmation.)
Run:  cd 집/클로드/research && PYTHONPATH="" py -3.12 -u ts6_gru_member_v1.py
"""
import os, sys
import env  # noqa: F401
import numpy as np, pandas as pd
sys.argv = ["x"]
sys.path.insert(0, os.path.join(env.ROOT, u"집", u"코덱스", "analysis", "temp_tk_season_20261003_v1"))
import world as W  # noqa: E402
import common  # noqa: E402
CK = os.path.join(env.LOCAL, "ts6_ckpt"); CK1 = os.path.join(env.LOCAL, "tt1_ckpt")
RD = set(map(tuple, pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "rw1_rough_days_v1.csv"))[["farm", "day"]].values))
USE = ["out_temp", "out_hum", "out_rad", "out_wspd", "in_temp", "in_hum", "in_co2", "act_vent", "act_shade", "act_thermal",
       "act_heating", "act_circfan", "act_co2", "act_fog"]
T = 48; ALPHA = .025
r = lambda e: float(np.sqrt(np.mean(np.square(e))))


def sequences(lab):
    tX, ty, sX = W.TM.masked_loader()
    a = pd.concat([tX, sX])
    a = a[a.farm.isin(["F13", "F47"])].sort_values(["farm", "t"]).reset_index(drop=True)
    a["hs"], a["hc"] = np.sin(2 * np.pi * a.hour / 24), np.cos(2 * np.pi * a.hour / 24)
    F = a[USE + ["hs", "hc"]].to_numpy(np.float32)
    pos = {rid: i for i, rid in enumerate(a.row_id)}
    farm = a.farm.values
    idx = np.array([pos[x] for x in lab.row_id])
    S = np.full((len(lab), T, F.shape[1]), np.nan, np.float32)
    for j, i in enumerate(idx):
        lo = max(0, i - T + 1)
        seg = F[lo:i + 1]; seg = seg[farm[lo:i + 1] == farm[i]]
        S[j, T - len(seg):] = seg
    return S


def fit_predict(Str, ytr, wtr, Sva, seed, dev):
    import torch
    torch.manual_seed(seed); np.random.seed(seed)
    mu = np.nanmean(Str.reshape(-1, Str.shape[2]), 0); sd = np.nanstd(Str.reshape(-1, Str.shape[2]), 0) + 1e-6
    def prep(S):
        Z = (S - mu) / sd; m = np.isnan(Z[..., 4:5]).astype(np.float32); Z = np.nan_to_num(Z, nan=0.0)
        return torch.tensor(np.concatenate([Z, m], 2), device=dev)
    Xt, Xv = prep(Str), prep(Sva); ym = ytr.mean()
    yt = torch.tensor((ytr - ym).astype(np.float32), device=dev); wt = torch.tensor(wtr.astype(np.float32), device=dev)
    class Net(torch.nn.Module):
        def __init__(s, d):
            super().__init__(); s.g = torch.nn.GRU(d, 64, batch_first=True); s.o = torch.nn.Linear(64, 1)
        def forward(s, x):
            h, _ = s.g(x); return s.o(h[:, -1]).squeeze(-1)
    net = Net(Xt.shape[2]).to(dev); opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    n = len(yt)
    for ep in range(25):
        perm = torch.randperm(n, device=dev)
        for b in range(0, n, 256):
            ii = perm[b:b + 256]; opt.zero_grad()
            loss = (wt[ii] * (net(Xt[ii]) - yt[ii]) ** 2).sum() / wt[ii].sum(); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        return np.concatenate([net(Xv[b:b + 2048]).cpu().numpy() for b in range(0, len(Xv), 2048)]) + ym


def run():
    os.makedirs(CK, exist_ok=True)
    import env_extra_gpu  # noqa: F401
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    lab, pfn, ct, phc, wb, wp, wv, sets, z = W.worlds(); wb = np.asarray(wb, float)
    S = sequences(lab); y = lab.sub_temp.values
    print("sequences", S.shape, "device", dev, flush=True)
    for name, folds in sets:
        for k, fd in enumerate(folds):
            path = os.path.join(CK, "%s_%d.csv" % (name, k))
            if os.path.exists(path):
                continue
            tm, vm = common.split_mask(lab, fd)
            if not vm.sum():
                continue
            p = np.mean([fit_predict(S[tm], y[tm], wb[tm], S[vm], sd, dev) for sd in (1, 2)], 0)
            pd.DataFrame({"validator": name, "row_id": lab.row_id[vm].values, "GRU": p}).to_csv(path, index=False)
            print("%s/%d done (GRU alone %.3f)" % (name, k, r(p - y[vm])), flush=True)


def boot(X, a, b, rng):
    cl = (X.farm + "_" + (X.day // 5).astype(str)).values
    dd = pd.Series((b - X.sub_temp.values) ** 2 - (a - X.sub_temp.values) ** 2).groupby(cl).agg(["sum", "count"])
    idx = rng.integers(0, len(dd), (20000, len(dd)))
    return float(((dd["sum"].values[idx].sum(1) / dd["count"].values[idx].sum(1)) >= 0).mean())


def summarize():
    rng = np.random.default_rng(20261026)
    C = pd.concat([pd.read_csv(os.path.join(CK, f)) for f in sorted(os.listdir(CK))])
    G = pd.concat([pd.read_csv(os.path.join(CK1, f)) for f in sorted(os.listdir(CK1))], ignore_index=True).merge(C, on=["validator", "row_id"])
    G = G[[(f, d) not in RD for f, d in zip(G.farm, G.day)]]
    ok = True
    for v in ("DIAG10", "EXT10", "EXT12", "EL1"):
        X = G[G.validator == v]; g = np.where(X.in_temp.isna(), 1, np.clip((X.in_temp - 8) / 2, 0, 1)); y = X.sub_temp.values; rel, pw = [], []
        for s in (7, 101):
            for f in "AB":
                a = (.4 * X["base_REF_%d" % s] + (.2 + .4 * (1 - g)) * X.codex_REF + .4 * g * X["pfn_%s" % f]).values; b = .9 * a + .1 * X.GRU.values
                rel.append(r(b - y) / r(a - y) - 1); pw.append(boot(X, a, b, rng))
        m = np.mean(rel)
        o = (all(x < 0 for x in rel) and max(pw) < ALPHA) if v in ("DIAG10", "EXT10") else not (m > 0 and (1 - min(pw)) < ALPHA)
        ok &= o
        print("  %-6s mean %+.2f%% [%+.2f, %+.2f] P max %.4f | GRU alone %.3f, corr(GRU err, W40G-S err) %.2f -> %s" % (
            v, 100 * m, 100 * min(rel), 100 * max(rel), max(pw), r(X.GRU - X.sub_temp), np.corrcoef(X.GRU - y, a - y)[0, 1], "ok" if o else "FAIL"))
    print("VERDICT GRU10: %s" % ("PASS" if ok else "FAIL"))


if __name__ == "__main__":
    if os.environ.get("TS6_SUM") != "1":
        run()
    summarize()
