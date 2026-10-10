# -*- coding: utf-8 -*-
"""TT25: fresh-layout confirmation of PFN6K (TabPFN context 6,000) on layout J (2026-10-11 집 클로드).  TT22/TT22b passed the
user's WARM-scope rule on the OLD validators (DIAG10 -0.98%, P .017; EL1/EXT10/EXT12 not significantly worse) but today's
lesson is that old-DIAG10 gains often vanish on new layouts.  Layout J members are stored (TT24: BASE 1010/1111, CODEX R,
PFN 2,000-context families A 49-52 / B 53-56).  Fixed before running.
Candidate: PFN term replaced by PFN6K (context 6,000 rows, weighted sampling, NEW seeds 201..203, mean of 3) on layout J.
PASS iff on layout J all rows (rough days excluded): all 4 cells (BASE seed x reference PFN family) better and farm x 6-day
  block bootstrap P(worse) < .025 (blocks aligned with J's 6-day chunks, critic 10-11).  Pass-2 reported.
Run:  cd 집/클로드/research && PYTHONPATH="" py -3.12 -u tt25_pfn6000_layoutJ_v1.py
"""
import os, sys, pathlib, gc, importlib.util
import env  # noqa: F401
import numpy as np, pandas as pd
sys.argv = ["x"]
sys.path.insert(0, os.path.join(env.ROOT, u"집", u"코덱스", "analysis", "temp_tk_season_20261003_v1"))
import world as W  # noqa: E402
HERE = os.path.dirname(os.path.abspath(__file__))
_sp = importlib.util.spec_from_file_location("t24", os.path.join(HERE, "tt24_gru_daygate_confirm_v1.py")); T24 = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(T24)
W.OUT = pathlib.Path(env.LOCAL) / "tt25_season"
CK = os.path.join(env.LOCAL, "tt25_ckpt"); CK24 = os.path.join(env.LOCAL, "tt24_ckpt"); RD = T24.RD
r = lambda e: float(np.sqrt(np.mean(np.square(e))))


def run():
    os.makedirs(CK, exist_ok=True); os.makedirs(W.OUT, exist_ok=True)
    lab, pfn, ct, phc, wb, wp, wv, sets, z = W.worlds(); wp = np.asarray(wp, float)
    import env_extra_gpu  # noqa: F401
    import torch
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion
    torch.set_num_threads(4)
    X = pfn[list(W.FEATURE_COLUMNS)].to_numpy(np.float32); y = pfn.sub_temp.to_numpy(); ixday = list(W.FEATURE_COLUMNS).index("day")
    for k, fd in enumerate(T24.fresh_folds(lab)):
        path = os.path.join(CK, "J_%d.csv" % k)
        if os.path.exists(path):
            continue
        tm, vm = T24.masks(lab, fd)
        tr, va = W.season_fold(pfn[tm], pfn[vm], wv, "tt25_J", k)
        XS = X[tm].copy(); VS = X[vm].copy(); XS[:, ixday] = tr.season; VS[:, ixday] = va.season
        ps = []
        for seed in (201, 202, 203):
            idx = np.random.default_rng(seed).choice(int(tm.sum()), min(6000, int(tm.sum())), replace=False, p=wp[tm] / wp[tm].sum())
            m = TabPFNRegressor.create_default_for_version(ModelVersion.V2, device="cuda", n_estimators=4, random_state=seed,
                                                           ignore_pretraining_limits=True, inference_precision=torch.float32)
            m.fit(XS[idx], y[tm][idx]); ps.append(m.predict(VS)); del m; gc.collect(); torch.cuda.empty_cache()
        pd.DataFrame({"row_id": lab.row_id[vm].values, "pfn6k": np.mean(ps, 0)}).to_csv(path, index=False)
        print("J/%d done" % k, flush=True)


def summarize():
    rng = np.random.default_rng(20261031)
    M = pd.concat([pd.read_csv(os.path.join(CK24, f)) for f in sorted(os.listdir(CK24)) if f.startswith("J_")], ignore_index=True)
    M = M.merge(pd.concat([pd.read_csv(os.path.join(CK, f)) for f in sorted(os.listdir(CK))]), on="row_id")
    M = M[[(f, d) not in RD for f, d in zip(M.farm, M.day)]].reset_index(drop=True)
    g = np.where(M.in_temp.isna(), 1, np.clip((M.in_temp - 8) / 2, 0, 1)); y = M.sub_temp.values; rel, pw, r2 = [], [], []
    cl = (M.farm + "_" + (M.day // 6).astype(str)).values
    for s in (1010, 1111):
        for f in "AB":
            a = (.4 * M["base_R_%d" % s] + (.2 + .4 * (1 - g)) * M.codex_R + .4 * g * M["pfn_%s" % f]).values
            b = (.4 * M["base_R_%d" % s] + (.2 + .4 * (1 - g)) * M.codex_R + .4 * g * M.pfn6k).values
            dd = pd.Series((b - y) ** 2 - (a - y) ** 2).groupby(cl).agg(["sum", "count"]); idx = rng.integers(0, len(dd), (20000, len(dd)))
            p = float(((dd["sum"].values[idx].sum(1) / dd["count"].values[idx].sum(1)) >= 0).mean())
            rel.append(r(b - y) / r(a - y) - 1); pw.append(p); L2 = M.day.values >= 179; r2.append(r((b - y)[L2]) / r((a - y)[L2]) - 1)
            print("  seed %d ref PFN %s: %+.2f%% P %.4f | 2차 %+.2f%%" % (s, f, 100 * rel[-1], p, 100 * r2[-1]))
    print("  PFN alone: 2000ctx A %.3f / 6000ctx %.3f" % (r(M.pfn_A - M.sub_temp), r(M.pfn6k - M.sub_temp)))
    print("VERDICT PFN6K layout J: %s" % ("PASS" if all(x < 0 for x in rel) and max(pw) < .025 else "FAIL"))


if __name__ == "__main__":
    if os.environ.get("TT25_SUM") != "1":
        run()
    summarize()
