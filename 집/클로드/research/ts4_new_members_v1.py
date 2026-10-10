# -*- coding: utf-8 -*-
"""TS4: add a DIFFERENT model family as a 4th member at 10% (2026-10-10 집 클로드, user: "걔네가 생각했을 만한 거 전부 다 해봐").
Fixed before running.  Members trained on the MASK-world BASE feature set (ct + phc, same rows, same weights wb):
  ET : ExtraTreesRegressor(400 trees, min_samples_leaf 5, max_features .5, random_state 7), median-imputed
  HGB: HistGradientBoostingRegressor(max_iter 500, learning_rate .05, max_leaf_nodes 31, l2 .1, random_state 7)
Candidate blend: 0.9 * W40G-S + 0.1 * member   (k = 2, alpha .0125).
Scope ALL -> primary DIAG10 AND EXT10: both BASE seeds x both PFN families better and farm x 5-day block bootstrap
  P(worse) < .0125 on both; EXT12/EL1 fail iff seed-mean worse and share(better) < .0125.  Guard (fixed now): fresh layouts
  F/G/H (members refit here) all-row seed-mean not worse.  Rough days excluded from scoring.
Run:  cd 집/클로드/research && PYTHONPATH="" py -3.12 -u ts4_new_members_v1.py
"""
import os, sys, pathlib, importlib.util
import env  # noqa: F401
import numpy as np, pandas as pd
sys.argv = ["x"]
sys.path.insert(0, os.path.join(env.ROOT, u"집", u"코덱스", "analysis", "temp_tk_season_20261003_v1"))
import world as W  # noqa: E402
import common  # noqa: E402
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
HERE = os.path.dirname(os.path.abspath(__file__))


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, file)); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


T12 = load("t12", "tt12_confirm_noh0_noheat_fresh_v1.py"); T13 = load("t13", "tt13_pass2_rule_confirm_v1.py"); T17 = load("t17", "tt17_p1_second_confirm_v1.py")
CK = os.path.join(env.LOCAL, "ts4_ckpt"); CK1 = os.path.join(env.LOCAL, "tt1_ckpt"); RD = T13.RD; ALPHA = .0125
FRESH = {"F": (T12.fresh_folds, T12.masks, os.path.join(env.LOCAL, "tt12_ckpt"), "fresh_", (202, 303), "base_%d"),
         "G": (T13.fresh_folds, T13.masks, os.path.join(env.LOCAL, "tt13_ckpt"), "G_", (404, 505), "base_R_%d"),
         "H": (T17.fresh_folds, T17.masks, os.path.join(env.LOCAL, "tt17_ckpt"), "H_", (606, 707), "base_R_%d")}
r = lambda e: float(np.sqrt(np.mean(np.square(e))))


def run():
    os.makedirs(CK, exist_ok=True)
    lab, pfn, ct, phc, wb, wp, wv, sets, z = W.worlds(); wb = np.asarray(wb, float)
    cols = list(dict.fromkeys(ct + phc)); X = lab[cols].to_numpy(float); y = lab.sub_temp.values
    jobs = [(name, k, common.split_mask(lab, fd)) for name, folds in sets for k, fd in enumerate(folds)]
    jobs += [(L, k, mk(lab, fd)) for L, (ff, mk, *_) in FRESH.items() for k, fd in enumerate(ff(lab))]
    for tag, k, (tm, vm) in jobs:
        path = os.path.join(CK, "%s_%d.csv" % (tag, k))
        if os.path.exists(path) or not vm.sum():
            continue
        imp = SimpleImputer(strategy="median").fit(X[tm]); Xt, Xv = imp.transform(X[tm]), imp.transform(X[vm])
        et = ExtraTreesRegressor(n_estimators=400, min_samples_leaf=5, max_features=.5, n_jobs=4, random_state=7).fit(Xt, y[tm], sample_weight=wb[tm])
        hg = HistGradientBoostingRegressor(max_iter=500, learning_rate=.05, max_leaf_nodes=31, l2_regularization=.1, random_state=7).fit(X[tm], y[tm], sample_weight=wb[tm])
        pd.DataFrame({"validator": tag, "row_id": lab.row_id[vm].values, "ET": et.predict(Xv), "HGB": hg.predict(X[vm])}).to_csv(path, index=False)
        print("%s/%d done" % (tag, k), flush=True)


def boot(X, a, b, rng):
    cl = (X.farm + "_" + (X.day // 5).astype(str)).values
    dd = pd.Series((b - X.sub_temp.values) ** 2 - (a - X.sub_temp.values) ** 2).groupby(cl).agg(["sum", "count"])
    idx = rng.integers(0, len(dd), (20000, len(dd)))
    return float(((dd["sum"].values[idx].sum(1) / dd["count"].values[idx].sum(1)) >= 0).mean())


def summarize():
    rng = np.random.default_rng(20261025)
    C = pd.concat([pd.read_csv(os.path.join(CK, f)) for f in sorted(os.listdir(CK))])
    G = pd.concat([pd.read_csv(os.path.join(CK1, f)) for f in sorted(os.listdir(CK1))], ignore_index=True).merge(C, on=["validator", "row_id"])
    for mem in ("ET", "HGB"):
        ok = True; print("\n==== %s (0.9 W40G-S + 0.1 %s)" % (mem, mem))
        def ev(X, seeds, bc, rc, label, judged, primary):
            X = X[[(f, d) not in RD for f, d in zip(X.farm, X.day)]]
            g = np.where(X.in_temp.isna(), 1, np.clip((X.in_temp - 8) / 2, 0, 1)); y = X.sub_temp.values; rel, pw = [], []
            for s in seeds:
                for f in "AB":
                    a = (.4 * X[bc % s] + (.2 + .4 * (1 - g)) * X[rc] + .4 * g * X["pfn_%s" % f]).values; b = .9 * a + .1 * X[mem].values
                    rel.append(r(b - y) / r(a - y) - 1); pw.append(boot(X, a, b, rng) if judged else np.nan)
            m = np.mean(rel)
            o = (all(x < 0 for x in rel) and max(pw) < ALPHA) if primary else ((not (m > 0 and (1 - np.nanmin(pw)) < ALPHA)) if judged else m <= 0)
            print("  %-8s mean %+.2f%% [%+.2f, %+.2f]%s | member alone %.3f -> %s" % (label, 100 * m, 100 * min(rel), 100 * max(rel), (" P max %.4f" % np.nanmax(pw)) if judged else "", r(X[mem] - X.sub_temp), "ok" if o else "FAIL"))
            return o
        for v in ("DIAG10", "EXT10", "EXT12", "EL1"):
            ok &= ev(G[G.validator == v], (7, 101), "base_REF_%d", "codex_REF", v, True, v in ("DIAG10", "EXT10"))
        for L, (_, _, ck, pre, seeds, bc) in FRESH.items():
            M = pd.concat([pd.read_csv(os.path.join(ck, f)) for f in sorted(os.listdir(ck)) if f.startswith(pre)], ignore_index=True)
            M = M.merge(C[C.validator == L].drop(columns="validator"), on="row_id")
            ok &= ev(M, seeds, bc, "codex_R", "fresh " + L, False, False)
        print("VERDICT %s: %s" % (mem, "PASS" if ok else "FAIL"))


if __name__ == "__main__":
    if os.environ.get("TS4_SUM") != "1":
        run()
    summarize()
