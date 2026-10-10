# -*- coding: utf-8 -*-
"""TS7 (SCREEN): sequence-model variants vs the current GRU member (2026-10-11 집 클로드, user: "그동안 다른 작업 해볼 만한 거
더 해봐").  The day-gated GRU (W40G-S + 0.1*g*h10*(GRU - W40G-S)) is the only change that held on fresh layouts.  Fixed before
running.  Variants (k = 3), each 2 seeds (1, 2), same training recipe as TS6 (Adam 2e-3, 25 epochs, batch 256, weighted MSE):
  V1 GRU96 : GRU(64) over 96 h instead of 48 h
  V2 CNN   : Conv1d(17->64,k5)-ReLU-Conv1d(64->64,k5,dil2)-ReLU-mean over last 12 steps -> Linear  (48 h)
  V3 GRU2L : 2-layer GRU, hidden 128 (48 h)
Compared on layouts F/G/H/I (stored BASE/CODEX/PFN; stored GRU OOF from TS6b/TS6d as the reference member), all rows,
rough days excluded, same day gate.  PROMOTE iff a variant beats the GRU blend on all four layouts (seed-mean over BASE seed x
PFN family) and pooled block bootstrap P(variant blend worse than GRU blend) < .0125.  Promotion needs a new-layout check.
Run:  cd 집/클로드/research && PYTHONPATH="" py -3.12 -u ts7_sequence_variants_v1.py
"""
import os, sys, importlib.util
import env  # noqa: F401
import numpy as np, pandas as pd
sys.argv = ["x"]
HERE = os.path.dirname(os.path.abspath(__file__))


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, file)); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


S6 = load("s6", "ts6_gru_member_v1.py"); T12 = load("t12", "tt12_confirm_noh0_noheat_fresh_v1.py"); T13 = load("t13", "tt13_pass2_rule_confirm_v1.py")
T17 = load("t17", "tt17_p1_second_confirm_v1.py"); T23 = load("t23", "tt23_pw1_confirm_v1.py")
W = S6.W; RD = S6.RD; ALPHA = .0125
CK = os.path.join(env.LOCAL, "ts7_ckpt")
LAY = {"F": (T12.fresh_folds, T12.masks, "tt12_ckpt", "fresh_", (202, 303), "base_%d", "ts6b_ckpt"),
       "G": (T13.fresh_folds, T13.masks, "tt13_ckpt", "G_", (404, 505), "base_R_%d", "ts6b_ckpt"),
       "H": (T17.fresh_folds, T17.masks, "tt17_ckpt", "H_", (606, 707), "base_R_%d", "ts6b_ckpt"),
       "I": (T23.fresh_folds, T23.masks, "tt23_ckpt", "I_", (808, 909), "base_R_%d", "ts6d_ckpt")}
r = lambda e: float(np.sqrt(np.mean(np.square(e))))


def fit_predict(Str, ytr, wtr, Ste, seed, dev, kind):
    import torch
    torch.manual_seed(seed); np.random.seed(seed)
    mu = np.nanmean(Str.reshape(-1, Str.shape[2]), 0); sd = np.nanstd(Str.reshape(-1, Str.shape[2]), 0) + 1e-6
    def prep(S):
        Z = (S - mu) / sd; m = np.isnan(Z[..., 4:5]).astype(np.float32); Z = np.nan_to_num(Z, nan=0.0)
        return torch.tensor(np.concatenate([Z, m], 2), device=dev)
    Xt, Xv = prep(Str), prep(Ste); ym = ytr.mean()
    yt = torch.tensor((ytr - ym).astype(np.float32), device=dev); wt = torch.tensor(wtr.astype(np.float32), device=dev)
    d = Xt.shape[2]
    class G(torch.nn.Module):
        def __init__(s, layers, hid):
            super().__init__(); s.g = torch.nn.GRU(d, hid, num_layers=layers, batch_first=True); s.o = torch.nn.Linear(hid, 1)
        def forward(s, x):
            h, _ = s.g(x); return s.o(h[:, -1]).squeeze(-1)
    class C(torch.nn.Module):
        def __init__(s):
            super().__init__(); s.c1 = torch.nn.Conv1d(d, 64, 5, padding=4); s.c2 = torch.nn.Conv1d(64, 64, 5, dilation=2, padding=8); s.o = torch.nn.Linear(64, 1)
        def forward(s, x):
            z = x.transpose(1, 2); L = z.shape[2]
            z = torch.relu(s.c1(z))[:, :, :L]; z = torch.relu(s.c2(z))[:, :, :L]   # causal-style trimming (left padding kept)
            return s.o(z[:, :, -12:].mean(2)).squeeze(-1)
    net = (G(1, 64) if kind == "V1" else G(2, 128) if kind == "V3" else C()).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=2e-3); n = len(yt)
    for _ in range(25):
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
    lab, pfn, ct, phc, wb, wp, wv, sets, z = W.worlds(); wb = np.asarray(wb, float); y = lab.sub_temp.values
    S48 = S6.sequences(lab)
    S6.T = 96; S96 = S6.sequences(lab); S6.T = 48
    for L, (ff, mk, *_) in LAY.items():
        for k, fd in enumerate(ff(lab)):
            path = os.path.join(CK, "%s_%d.csv" % (L, k))
            if os.path.exists(path):
                continue
            tm, vm = mk(lab, fd); out = pd.DataFrame({"row_id": lab.row_id[vm].values})
            out["V1"] = np.mean([fit_predict(S96[tm], y[tm], wb[tm], S96[vm], s, dev, "V1") for s in (1, 2)], 0)
            out["V2"] = np.mean([fit_predict(S48[tm], y[tm], wb[tm], S48[vm], s, dev, "V2") for s in (1, 2)], 0)
            out["V3"] = np.mean([fit_predict(S48[tm], y[tm], wb[tm], S48[vm], s, dev, "V3") for s in (1, 2)], 0)
            out.to_csv(path, index=False); print("%s/%d done | alone V1 %.3f V2 %.3f V3 %.3f" % (L, k, *(r(out[c] - y[vm]) for c in ("V1", "V2", "V3"))), flush=True)


def summarize():
    rng = np.random.default_rng(20261101)
    for v in ("V1", "V2", "V3"):
        print("\n==== %s" % v); ok = True; parts = []
        for L, (_, _, ck, pre, seeds, bc, gck) in LAY.items():
            ld = lambda d, p: pd.concat([pd.read_csv(os.path.join(env.LOCAL, d, f)) for f in sorted(os.listdir(os.path.join(env.LOCAL, d))) if f.startswith(p)], ignore_index=True)
            M = ld(ck, pre).merge(ld(gck, L + "_"), on="row_id").merge(ld("ts7_ckpt", L + "_"), on="row_id")
            M = M[[(f, d) not in RD for f, d in zip(M.farm, M.day)]].reset_index(drop=True)
            g = np.where(M.in_temp.isna(), 1, np.clip((M.in_temp - 8) / 2, 0, 1))
            mm = M.sort_values(["farm", "day", "hour"]).groupby(["farm", "day"]).in_temp.cummin().reindex(M.index)
            h = np.where(mm.isna(), 1, np.clip((mm - 10) / 2, 0, 1)); y = M.sub_temp.values; rg, rv = [], []
            for s in seeds:
                for f in "AB":
                    a = (.4 * M[bc % s] + (.2 + .4 * (1 - g)) * M.codex_R + .4 * g * M["pfn_%s" % f]).values
                    bg = a + .1 * g * h * (M.GRU.values - a); bv = a + .1 * g * h * (M[v].values - a)
                    rg.append(r(bg - y) / r(a - y) - 1); rv.append(r(bv - y) / r(a - y) - 1)
                    if s == seeds[0] and f == "A":
                        parts.append((M, bg, bv))
            ok &= np.mean(rv) < np.mean(rg)
            print("  layout %s: vs W40G-S  GRU %+.2f%%  %s %+.2f%% | alone GRU %.3f %s %.3f" % (L, 100 * np.mean(rg), v, 100 * np.mean(rv), r(M.GRU - M.sub_temp), v, r(M[v] - M.sub_temp)))
        sm, n = [], []
        for X, a, b in parts:
            cl = (X.farm + "_" + (X.day // 5).astype(str)).values
            dd = pd.Series((b - X.sub_temp.values) ** 2 - (a - X.sub_temp.values) ** 2).groupby(cl).agg(["sum", "count"]); sm.append(dd["sum"].values); n.append(dd["count"].values)
        sm, n = np.concatenate(sm), np.concatenate(n); idx = rng.integers(0, len(sm), (20000, len(sm)))
        p = float(((sm[idx].sum(1) / n[idx].sum(1)) >= 0).mean())
        print("  pooled P(%s blend worse than GRU blend) %.4f -> %s" % (v, p, "PROMOTE" if ok and p < ALPHA else "stop"))


if __name__ == "__main__":
    if os.environ.get("TS7_SUM") != "1":
        run()
    summarize()
