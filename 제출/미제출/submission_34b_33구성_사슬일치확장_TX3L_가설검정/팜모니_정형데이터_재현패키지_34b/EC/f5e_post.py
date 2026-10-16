# -*- coding: utf-8 -*-
"""submission_25 EC (2026-10-15 집 클로드, LEADERBOARD HYPOTHESIS TEST; '5-point / field-information' build; catalog 6.901, 6.921, NH2):
    F5E = 0.9 * NH + 0.1 * EV3L,     NH = pre + 0.5 * HR5(pre)
  pre  = round-9 EC members BEFORE the SG2 step (= submission_14 R3 x3 seeds + TabPFN x4 contexts, season index from the date estimate, DP1
         own-day operation features): clip(shrink(0.8 R3 + 0.2 PFN)) -- read from output_14/prediction_details.npz exactly as w2only_post.py.
         Removed vs round 15: SG2 (reference-calendar kNN level), W2 (record number - 2 rule), B5 (same-date high-EC exclusion) -- structure rules
         with no field counterpart.
  HR5  = the submission_24 day-level TabPFN corrector (same code/settings as hr5_post.py) but with base = pre and the training base =
         hr5_train_base_nosg_oof.csv (stored DIAG10 out-of-fold prediction of the same pre-SG2 stage, 360 records).
  EV3L = 2-layer GRU(128) over a 48-hour window [same house's previous day 0..23] + [today 0..23], outputs at today's positions (unidirectional
         -> hour h sees yesterday + today 0..h).  Previous day = v15R linker prev0 (p0 >= .5): TRAINING sequences use the MASK-world mirror links
         (v15r_mirror_pkg.py on data with all test_X inputs NaN) and MASK-world inputs; EVALUATION records use v15r_eval_pkg.py picks, kept only if
         the previous day is a training record or a same-farm earlier-numbered evaluation record.  Inputs = 14 usable + hour sin/cos, z-scored on
         training sequences, NaN -> 0 + in_temp-missing flag; target = sub_ec per hour (training mean removed); Adam 2e-3, 25 epochs, batch 16
         records, seeds 1 and 2 averaged, CPU, deterministic.
Local evidence: EV3L vs own-day GRU on top of sg: P2LOO P .008 (ADOPT); NH2 (independent SZ2 bases): F5E vs the SG2 structure stage 6/6 main cells
better (EL1 -7.3..-8.3%, P2LOO -5.7..-6.2%) but P2LOO P .27 -> HYPOTHESIS TEST.
Run:  python f5e_post.py <train_links_mirror_masked.csv> <eval_links_v15r.csv>   (writes output_25/ec_pre.csv, ec_nh.csv, ec_f5e.csv)
"""
import os, sys, subprocess
if os.environ.get("PYTHONHASHSEED") != "0":
    sys.exit(subprocess.call([sys.executable] + sys.argv, env=dict(os.environ, PYTHONHASHSEED="0")))
import random
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[_k] = "1"
for _k in ("HF_HUB_OFFLINE", "TABPFN_DISABLE_TELEMETRY", "HF_HUB_DISABLE_TELEMETRY"): os.environ[_k] = "1"
import env  # noqa: F401
import env_extra  # noqa: F401
import numpy as np, pandas as pd
import torch
from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion
import c18post as C
from hr5_post import feat, keep, cube, USE, HS, LAM
torch.set_num_threads(1); torch.use_deterministic_algorithms(True); torch.manual_seed(0)   # interop threads already fixed by hr5_post import
random.seed(0); np.random.seed(0)
HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, "data"); P_MIN = .5; W_EV = .1


def stage_pre(out_dir):
    z = np.load(os.path.join(HERE, "output_14", "prediction_details.npz")); rid = z["row_id"]
    F = pd.DataFrame({"row_id": rid, "farm": [r[:3] for r in rid], "day": [int(r[4:7]) for r in rid], "hour": [int(r[8:10]) for r in rid]})
    ty = pd.read_csv(os.path.join(DATA, "train_y.csv")); ty = ty[ty.row_id.str[:3].isin(["F13", "F47"])]
    lo, hi = float(ty.sub_ec.min()), float(ty.sub_ec.max())
    r3, pfn = z["raw_r3_by_seed"].mean(0), z["raw_pfn_by_context"].mean(0)
    F["sub_ec"] = np.clip(C.shrink(.8 * r3 + .2 * pfn, F), lo, hi)
    S = pd.read_csv(os.path.join(DATA, "sample_submission.csv"))[["row_id"]].merge(F[["row_id", "sub_ec"]], on="row_id", how="left")
    assert S.sub_ec.notna().all(); S.to_csv(os.path.join(out_dir, "ec_pre.csv"), index=False, float_format="%.6f"); return S


def stage_hr5(P, out_dir):
    TI = cube(pd.read_csv(os.path.join(DATA, "train_X.csv"))); EI = cube(pd.read_csv(os.path.join(DATA, "test_X.csv")))
    ty = pd.read_csv(os.path.join(DATA, "train_y.csv")).set_index("row_id").sub_ec
    B = pd.read_csv(os.path.join(HERE, "hr5_train_base_nosg_oof.csv")); B["rec"] = B.row_id.str[:7]; B["y"] = ty.reindex(B.row_id).values
    recs = sorted(B.rec.unique()); BB = {r: g.sort_values("row_id") for r, g in B.groupby("rec")}
    TD = np.array([float(BB[r].y.mean() - BB[r].base.mean()) for r in recs]); TF = {}
    for h in HS:
        F = np.stack([feat(TI[r], BB[r].base.values, h) for r in recs])[:, keep(h)]; TF[h] = np.where(np.isfinite(F), F, np.nanmean(F, 0))
    ck = os.path.join(HERE, "tabpfn-v2-regressor.ckpt"); models = {}
    for h in HS:
        m = TabPFNRegressor.create_default_for_version(ModelVersion.V2, device="cpu", model_path=ck, n_estimators=4, random_state=0,
                                                       ignore_pretraining_limits=True, inference_precision=torch.float32, n_preprocessing_jobs=1)
        m.fit(TF[h], TD); models[h] = m
    P = P.copy(); P["rec"] = P.row_id.str[:7]; P["h"] = P.row_id.str[8:10].astype(int); new = P.sub_ec.values.copy()
    G = {r: g.sort_values("h") for r, g in P[P.rec.str[:3].isin(["F13", "F47"])].groupby("rec")}; er = sorted(G); DH = {}
    for h in HS:
        Q = np.stack([feat(EI[r], G[r].sub_ec.values, h) for r in er])[:, keep(h)]; Q = np.where(np.isfinite(Q), Q, TF[h].mean(0)[None, :])
        DH[h] = dict(zip(er, models[h].predict(Q)))
    for r in er:
        g = G[r]; new[g.index.values] = g.sub_ec.values + LAM * np.array([DH[max(x for x in HS if x <= hh)][r] for hh in g.h.values])
    P["sub_ec"] = new; P[["row_id", "sub_ec"]].to_csv(os.path.join(out_dir, "ec_nh.csv"), index=False, float_format="%.6f"); return P[["row_id", "sub_ec"]]


def day_cube(X, mask_test_rows):
    X = X[X.row_id.str[:3].isin(["F13", "F47"])].copy(); X["rec"] = X.row_id.str[:7]; X["h"] = X.row_id.str[8:10].astype(int)
    if mask_test_rows is not None: X.loc[X.rec.isin(mask_test_rows), USE] = np.nan
    X["hs"], X["hc"] = np.sin(2 * np.pi * X.h / 24), np.cos(2 * np.pi * X.h / 24); cols = USE + ["hs", "hc"]; D = {}
    for r, g in X.groupby("rec"):
        gg = g.sort_values("h"); m = np.full((24, len(cols)), np.nan, np.float32); m[gg.h.values] = gg[cols].to_numpy(np.float32); D[r] = m
    return D, len(cols)


def stage_ev3l(ltr, lev):
    tX = pd.read_csv(os.path.join(DATA, "train_X.csv")); sX = pd.read_csv(os.path.join(DATA, "test_X.csv")); ty = pd.read_csv(os.path.join(DATA, "train_y.csv"))
    ty = ty[ty.row_id.str[:3].isin(["F13", "F47"])].copy(); ty["rec"] = ty.row_id.str[:7]; ty["h"] = ty.row_id.str[8:10].astype(int)
    test_recs = set(sX.row_id.str[:7]); train_recs = set(ty.rec)
    allX = pd.concat([tX, sX]); Dm, nc = day_cube(allX, test_recs); Df, _ = day_cube(allX, None)
    Y = {r: np.full(24, np.nan, np.float32) for r in train_recs}
    for r, g in ty.groupby("rec"): Y[r][g.h.values] = g.sub_ec.values
    lab = sorted(r for r in train_recs if np.isfinite(Y[r]).any())
    LT = pd.read_csv(ltr); link_tr = {r: p for r, p, q in zip(LT.rec, LT.prev0, LT.p0) if isinstance(p, str) and q >= P_MIN and p != r}
    LE = pd.read_csv(lev); allowed = lambda r, p: p in train_recs or (p[:3] == r[:3] and int(p[4:7]) < int(r[4:7]))
    link_ev = {r: p for r, p, q in zip(LE.rec, LE.prev0, LE.p0) if isinstance(p, str) and q >= P_MIN and p != r and allowed(r, p)}
    nan = np.full((24, nc), np.nan, np.float32)
    Xt = np.stack([np.concatenate([Dm.get(link_tr.get(r), nan) if link_tr.get(r) else nan, Dm[r]]) for r in lab])
    er = sorted(r for r in test_recs if r[:3] in ("F13", "F47"))
    Xe = np.stack([np.concatenate([Df.get(link_ev.get(r), nan) if link_ev.get(r) else nan, Df[r]]) for r in er])
    Yt = np.stack([Y[r] for r in lab])
    mu = np.nanmean(Xt.reshape(-1, nc), 0); sd = np.nanstd(Xt.reshape(-1, nc), 0) + 1e-6
    def prep(A):
        Z = (A - mu) / sd; m = np.isnan(A[..., 4:5]).astype(np.float32); return torch.tensor(np.concatenate([np.nan_to_num(Z), m], 2).astype(np.float32))
    xt, xe = prep(Xt), prep(Xe); ym = float(np.nanmean(Yt)); yt = torch.tensor(np.nan_to_num(Yt - ym).astype(np.float32)); mt = torch.tensor(np.isfinite(Yt).astype(np.float32))
    class NetA(torch.nn.Module):
        def __init__(s, d):
            super().__init__(); s.g = torch.nn.GRU(d, 128, num_layers=2, batch_first=True); s.o = torch.nn.Linear(128, 1)
        def forward(s, x): return s.o(s.g(x)[0]).squeeze(-1)
    out = []
    for seed in (1, 2):
        torch.manual_seed(seed); net = NetA(xt.shape[2]); opt = torch.optim.Adam(net.parameters(), lr=2e-3); g = torch.Generator().manual_seed(seed)
        for ep in range(25):
            perm = torch.randperm(len(xt), generator=g)
            for b in range(0, len(xt), 16):
                j = perm[b:b + 16]; e = net(xt[j])[:, 24:]; mj = mt[j]
                loss = (((e - yt[j]) ** 2) * mj).sum() / mj.sum().clamp(min=1); opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad(): out.append(net(xe)[:, 24:].numpy() + ym)
    P = np.mean(out, 0)
    rows = [(f"{r}_{h:02d}", float(P[i, h])) for i, r in enumerate(er) for h in range(24)]
    return pd.DataFrame(rows, columns=["row_id", "ev3l"]), len(link_tr), sum(r in link_ev for r in er)


def main():
    ltr, lev = sys.argv[1:3]
    out_dir = os.path.join(HERE, "output_25"); os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "ec_f5e.csv"); assert not os.path.exists(out), "refusing to overwrite " + out
    pre = stage_pre(out_dir); nh = stage_hr5(pre, out_dir); ev, ntr, nev = stage_ev3l(ltr, lev)
    F = nh.merge(ev, on="row_id", how="left"); assert F.ev3l.notna().all() and len(F) == 1440
    F["sub_ec"] = (1 - W_EV) * F.sub_ec + W_EV * F.ev3l; F[["row_id", "sub_ec"]].to_csv(out, index=False, float_format="%.6f"); F[["row_id", "ev3l"]].to_csv(os.path.join(out_dir, "ev3l_member.csv"), index=False, float_format="%.6f")
    d1 = nh.sub_ec.values - pre.sub_ec.values; d2 = F.sub_ec.values - nh.sub_ec.values
    print("wrote %s | HR5 change RMS %.4f mean %+.4f | EV3L blend change RMS %.4f mean %+.4f | training links %d, evaluation records with yesterday %d/60"
          % (out, np.sqrt((d1 ** 2).mean()), d1.mean(), np.sqrt((d2 ** 2).mean()), d2.mean(), ntr, nev))


if __name__ == "__main__":
    main()
