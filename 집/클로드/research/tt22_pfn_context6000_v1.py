# -*- coding: utf-8 -*-
"""TT22: temperature TabPFN with a LARGER context (2026-10-10 집 클로드, user: "뭐라도 해봐 진짜").  The temperature TabPFN
member (TK2: 2,000 weighted-sampled training rows per context, n_estimators 4, contexts 1-8) was sized for CPU; with the GPU a
6,000-row context is affordable.  EC tried 6,000 and got worse (6.384) - never tested for temperature.  Fixed before running.
Candidate PFN6K: same TK2 recipe (season features, weighted sampling without replacement, float32, n_estimators 4) but
  context 6,000 rows, NEW context seeds 101..103, mean of the 3.  Replaces the PFN term of W40G-S (PFN family A = TK2 seeds
  1-8 as reference); BASE (7/101) and CODEX unchanged.
Scope WARM (PFN acts only where g > 0) -> primary DIAG10 (user temperature scope rule): both BASE seeds better and farm x
  5-day block bootstrap P(worse) < .025.  EL1 reported (fail iff seed-mean worse and share(better) < .025), plus pass-2 rows.
  Rough days excluded from scoring.  Validators run: DIAG10, EL1 (EXT10/EXT12 only if DIAG10 passes - separate run).
Run:  cd 집/클로드/research && PYTHONPATH="" py -3.12 -u tt22_pfn_context6000_v1.py
"""
import os, sys, pathlib, gc, time
import env  # noqa: F401
import numpy as np, pandas as pd
sys.argv = ["x"]
sys.path.insert(0, os.path.join(env.ROOT, u"집", u"코덱스", "analysis", "temp_tk_season_20261003_v1"))
import world as W  # noqa: E402
import common  # noqa: E402
CK1 = os.path.join(env.LOCAL, "tt1_ckpt"); CK = os.path.join(env.LOCAL, "tt22_ckpt")
W.OUT = pathlib.Path(env.LOCAL) / "tt22_season"
RD = set(map(tuple, pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "rw1_rough_days_v1.csv"))[["farm", "day"]].values))
N_CTX = 6000; SEEDS = (101, 102, 103); ALPHA = .025
r = lambda e: float(np.sqrt(np.mean(np.square(e))))


def run():
    os.makedirs(CK, exist_ok=True); os.makedirs(W.OUT, exist_ok=True)
    lab, pfn, ct, phc, wb, wp, wv, sets, z = W.worlds()
    wp = np.asarray(wp, float)
    import env_extra_gpu  # noqa: F401
    import torch
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion
    torch.set_num_threads(4); assert torch.cuda.is_available()
    X = pfn[list(W.FEATURE_COLUMNS)].to_numpy(np.float32); y = pfn.sub_temp.to_numpy(); ixday = list(W.FEATURE_COLUMNS).index("day")
    t0 = time.time()
    for name, folds in sets:
        if name not in ("DIAG10", "EL1"):
            continue
        for k, fd in enumerate(folds):
            path = os.path.join(CK, "%s_%d.csv" % (name, k))
            if os.path.exists(path):
                continue
            tm, vm = common.split_mask(lab, fd)
            if not vm.sum():
                continue
            tr, va = W.season_fold(pfn[tm], pfn[vm], wv, "tt22_" + name, k)
            XS = X[tm].copy(); VS = X[vm].copy(); XS[:, ixday] = tr.season; VS[:, ixday] = va.season
            ps = []
            for seed in SEEDS:
                n = min(N_CTX, int(tm.sum()))
                idx = np.random.default_rng(seed).choice(int(tm.sum()), n, replace=False, p=wp[tm] / wp[tm].sum())
                m = TabPFNRegressor.create_default_for_version(ModelVersion.V2, device="cuda", n_estimators=4, random_state=seed,
                                                               ignore_pretraining_limits=True, inference_precision=torch.float32)
                m.fit(XS[idx], y[tm][idx]); ps.append(m.predict(VS)); del m; gc.collect(); torch.cuda.empty_cache()
            pd.DataFrame({"validator": name, "row_id": lab.row_id[vm].values, "pfn6k": np.mean(ps, 0)}).to_csv(path, index=False)
            print("%s/%d done (%.0fs)" % (name, k, time.time() - t0), flush=True)


def boot(X, a, b, rng):
    cl = (X.farm + "_" + (X.day // 5).astype(str)).values
    dd = pd.Series((b - X.sub_temp.values) ** 2 - (a - X.sub_temp.values) ** 2).groupby(cl).agg(["sum", "count"])
    idx = rng.integers(0, len(dd), (20000, len(dd)))
    return float(((dd["sum"].values[idx].sum(1) / dd["count"].values[idx].sum(1)) >= 0).mean())


def summarize():
    rng = np.random.default_rng(20261020)
    G = pd.concat([pd.read_csv(os.path.join(CK1, f)) for f in sorted(os.listdir(CK1))], ignore_index=True)
    G = G.merge(pd.concat([pd.read_csv(os.path.join(CK, f)) for f in sorted(os.listdir(CK))]), on=["validator", "row_id"], how="inner")
    G = G[[(f, d) not in RD for f, d in zip(G.farm, G.day)]].copy()
    g = np.where(G.in_temp.isna(), 1, np.clip((G.in_temp - 8) / 2, 0, 1)); G["g"] = g
    ok = True
    for v in ("DIAG10", "EL1"):
        X = G[G.validator == v]; y = X.sub_temp.values; rel, pw = [], []
        for s in (7, 101):
            bl = lambda p: (.4 * X["base_REF_%d" % s] + (.2 + .4 * (1 - X.g)) * X.codex_REF + .4 * X.g * X[p]).values
            a, b = bl("pfn_A"), bl("pfn6k"); p = boot(X, a, b, rng)
            rel.append(r(b - y) / r(a - y) - 1); pw.append(p)
            L = X.day.values >= 179
            print("  %-6s seed %3d: %.4f -> %.4f (%+.2f%%) P %.4f | 2차 %+.2f%% | PFN 단독 %.3f -> %.3f" % (
                v, s, r(a - y), r(b - y), 100 * rel[-1], p, 100 * (r((b - y)[L]) / r((a - y)[L]) - 1), r(X.pfn_A - X.sub_temp), r(X.pfn6k - X.sub_temp)))
        if v == "DIAG10":
            o = all(x < 0 for x in rel) and max(pw) < ALPHA
        else:
            o = not (np.mean(rel) > 0 and (1 - min(pw)) < ALPHA)
        ok &= o; print("  %-6s -> %s" % (v, "pass" if o else "fail"))
    print("VERDICT PFN6K (DIAG10+EL1 stage): %s" % ("PASS - run EXT10/EXT12 next" if ok else "FAIL"))


if __name__ == "__main__":
    if os.environ.get("TT22_SUM") != "1":
        run()
    summarize()
