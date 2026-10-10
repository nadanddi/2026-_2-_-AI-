# -*- coding: utf-8 -*-
"""TS3: CODEX residual LightGBM with row/feature subsampling, 5 seeds averaged (variance reduction)  (2026-10-10 집 클로드,
user: "그동안 다른 것도 뭐라도 해봐").  CODEX LGB is deterministic today (no subsampling).  Fixed before running.
Candidate CB5: same physics Ridge + LGB residual (220 trees, lr .035, 12 leaves, min_child 100, lambda 15) but
  subsample .8 (freq 1) and colsample_bytree .8, seeds 11..15, predictions averaged.  BASE, PFN, blend unchanged.
k = 1, alpha .025.  Scope ALL -> primary DIAG10 AND EXT10: both BASE seeds x both PFN families better and farm x 5-day block
  bootstrap P(worse) < .025 on both; EXT12/EL1 fail iff seed-mean worse and share(better) < .025.  Guard (fixed now): on the
  fresh layouts F/G/H (stored members, CODEX refit here) the all-row seed-mean must not be worse.  Rough days excluded.
Run:  cd 집/클로드/research && PYTHONPATH="" py -3.12 -u ts3_codex_bagging_v1.py
"""
import os, sys, pathlib, importlib.util
import env  # noqa: F401
import numpy as np, pandas as pd
sys.argv = ["x"]
sys.path.insert(0, os.path.join(env.ROOT, u"집", u"코덱스", "analysis", "temp_tk_season_20261003_v1"))
import world as W  # noqa: E402
import common  # noqa: E402
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor
HERE = os.path.dirname(os.path.abspath(__file__))


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, file)); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


T12 = load("t12", "tt12_confirm_noh0_noheat_fresh_v1.py"); T13 = load("t13", "tt13_pass2_rule_confirm_v1.py"); T17 = load("t17", "tt17_p1_second_confirm_v1.py")
W.OUT = pathlib.Path(env.LOCAL) / "ts3_season"; CK = os.path.join(env.LOCAL, "ts3_ckpt"); CK1 = os.path.join(env.LOCAL, "tt1_ckpt")
RD = T13.RD; ALPHA = .025
FRESH = {"F": (T12.fresh_folds, T12.masks, os.path.join(env.LOCAL, "tt12_ckpt"), "fresh_", (202, 303), "base_%d"),
         "G": (T13.fresh_folds, T13.masks, os.path.join(env.LOCAL, "tt13_ckpt"), "G_", (404, 505), "base_R_%d"),
         "H": (T17.fresh_folds, T17.masks, os.path.join(env.LOCAL, "tt17_ckpt"), "H_", (606, 707), "base_R_%d")}
r = lambda e: float(np.sqrt(np.mean(np.square(e))))


def cb5(tr, va, w, FC, PC):
    lin = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(), Ridge(alpha=100.0))
    lin.fit(tr[PC], tr.sub_temp.values, ridge__sample_weight=w)
    res = tr.sub_temp.values - lin.predict(tr[PC]); out = []
    for s in range(11, 16):
        m = LGBMRegressor(n_estimators=220, learning_rate=0.035, num_leaves=12, max_depth=-1, min_child_samples=100, reg_lambda=15,
                          subsample=.8, subsample_freq=1, colsample_bytree=.8, verbosity=-1, n_jobs=4, random_state=s)
        m.fit(tr[FC], res, sample_weight=w); out.append(m.predict(va[FC]))
    return lin.predict(va[PC]) + np.mean(out, 0)


def run():
    os.makedirs(CK, exist_ok=True); os.makedirs(W.OUT, exist_ok=True)
    lab, pfn, ct, phc, wb, wp, wv, sets, z = W.worlds(); wp = np.asarray(wp, float)
    FC = [c if c != "day" else "season" for c in W.FEATURE_COLUMNS]; PC = list(W.TM.PHYSICS_COLUMNS)
    jobs = [(name, k, common.split_mask(lab, fd)) for name, folds in sets for k, fd in enumerate(folds)]
    jobs += [(L, k, mk(lab, fd)) for L, (ff, mk, *_) in FRESH.items() for k, fd in enumerate(ff(lab))]
    for tag, k, (tm, vm) in jobs:
        path = os.path.join(CK, "%s_%d.csv" % (tag, k))
        if os.path.exists(path) or not vm.sum():
            continue
        tr, va = W.season_fold(pfn[tm], pfn[vm], wv, "ts3_" + tag, k)
        pd.DataFrame({"validator": tag, "row_id": lab.row_id[vm].values, "codex_CB5": cb5(tr.reset_index(drop=True), va.reset_index(drop=True), wp[tm], FC, PC)}).to_csv(path, index=False)
        print("%s/%d done" % (tag, k), flush=True)


def boot(X, a, b, rng):
    cl = (X.farm + "_" + (X.day // 5).astype(str)).values
    dd = pd.Series((b - X.sub_temp.values) ** 2 - (a - X.sub_temp.values) ** 2).groupby(cl).agg(["sum", "count"])
    idx = rng.integers(0, len(dd), (20000, len(dd)))
    return float(((dd["sum"].values[idx].sum(1) / dd["count"].values[idx].sum(1)) >= 0).mean())


def summarize():
    rng = np.random.default_rng(20261024)
    C = pd.concat([pd.read_csv(os.path.join(CK, f)) for f in sorted(os.listdir(CK))])
    G = pd.concat([pd.read_csv(os.path.join(CK1, f)) for f in sorted(os.listdir(CK1))], ignore_index=True).merge(C, on=["validator", "row_id"])
    ok = True
    def ev(X, seeds, bc, rc, label, judged, primary):
        nonlocal ok
        X = X[[(f, d) not in RD for f, d in zip(X.farm, X.day)]]
        g = np.where(X.in_temp.isna(), 1, np.clip((X.in_temp - 8) / 2, 0, 1)); y = X.sub_temp.values; rel, pw = [], []
        for s in seeds:
            for f in "AB":
                bl = lambda c: (.4 * X[bc % s] + (.2 + .4 * (1 - g)) * X[c] + .4 * g * X["pfn_%s" % f]).values
                a, b = bl(rc), bl("codex_CB5"); rel.append(r(b - y) / r(a - y) - 1); pw.append(boot(X, a, b, rng) if judged else np.nan)
        m = np.mean(rel); L2 = X.day.values >= 179
        if primary:
            o = all(x < 0 for x in rel) and max(pw) < ALPHA
        elif judged:
            o = not (m > 0 and (1 - np.nanmin(pw)) < ALPHA)
        else:
            o = m <= 0
        ok &= o
        print("  %-6s mean %+.2f%% [%+.2f, %+.2f]%s -> %s" % (label, 100 * m, 100 * min(rel), 100 * max(rel), (" P max %.4f" % np.nanmax(pw)) if judged else "", "ok" if o else "FAIL"))
    for v in ("DIAG10", "EXT10", "EXT12", "EL1"):
        ev(G[G.validator == v], (7, 101), "base_REF_%d", "codex_REF", v, True, v in ("DIAG10", "EXT10"))
    for L, (_, _, ck, pre, seeds, bc) in FRESH.items():
        M = pd.concat([pd.read_csv(os.path.join(ck, f)) for f in sorted(os.listdir(ck)) if f.startswith(pre)], ignore_index=True)
        M = M.merge(C[C.validator == L].drop(columns="validator"), on="row_id")
        ev(M, seeds, bc, "codex_R", "fresh " + L, False, False)
    print("VERDICT CB5: %s" % ("PASS" if ok else "FAIL"))


if __name__ == "__main__":
    if os.environ.get("TS3_SUM") != "1":
        run()
    summarize()
