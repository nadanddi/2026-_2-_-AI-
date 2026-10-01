# -*- coding: utf-8 -*-
"""EC source search C1: partial / transformed matches between test-day inputs
and every train record (all 51 greenhouses).  Rules fixed in
EC_정보원_입력쪽_사전고정_2026-10-02.md (commit 81ce501) before this run.
2026-10-02 집 클로드.

C1a: 6-h windows, per-channel constant offset removed, residual RMS
     in_temp <= .35 (.45 if the candidate greenhouse is integer-resolution),
     in_hum <= .8, in_co2 <= 4, all three at once.
C1b: 12-h windows, per-channel slope (0.5..2) + offset, same tolerances.
Queries: every contiguous non-trivial test window (temp range >= 1 or co2
range >= 20).  Chance baseline: the same number of F13/F47 train windows,
stratified by start hour.  Candidates: all train windows except the query's
own greenhouse within +-1 day.
GO (per variant): (i) share of query windows with >= 1 match >= 3x baseline
and Fisher one-sided p < .001, or (ii) a test stretch matched >= 12 h
continuously to one candidate with one time offset while the baseline has none.

Run:  cd 집/클로드/research && PYTHONPATH="" py -3.12 -u ec_src_C1_partial_match_v1.py
"""
import env  # noqa: F401
import os
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

CH = ["in_temp", "in_hum", "in_co2"]
TOL = np.array([0.35, 0.8, 4.0])
TOL_T_INT = 0.45
RNG = np.random.default_rng(20261002)


def load():
    tX = pd.read_csv(os.path.join(env.DATA, "train_X.csv"), usecols=["row_id"] + CH)
    sX = pd.read_csv(os.path.join(env.DATA, "test_X.csv"), usecols=["row_id"] + CH)
    tX["test"], sX["test"] = False, True
    X = pd.concat([tX, sX], ignore_index=True)
    p = X.row_id.str.split("_", expand=True)
    X["farm"], X["day"], X["hour"] = p[0], p[1].astype(int), p[2].astype(int)
    X["t"] = X.day * 24 + X.hour
    return X.sort_values(["farm", "t"]).reset_index(drop=True)


def windows(X, L):
    out = []
    for f, g in X.groupby("farm", sort=False):
        v = g[CH].to_numpy(np.float32)
        t = g.t.to_numpy()
        te = g.test.to_numpy()
        n = len(g)
        if n < L:
            continue
        idx = np.arange(n - L + 1)
        ok = (t[idx + L - 1] - t[idx]) == L - 1
        W = np.lib.stride_tricks.sliding_window_view(v, (L, 3))[:, 0]
        ok &= ~np.isnan(W).any(axis=(1, 2))
        same = np.array([te[i:i + L].all() or (~te[i:i + L]).all() for i in idx])
        ok &= same
        for i in idx[ok]:
            pass
        sel = idx[ok]
        out.append(pd.DataFrame({"farm": f, "t0": t[sel], "test": te[sel]}).assign(_w=list(W[sel])))
    D = pd.concat(out, ignore_index=True)
    A = np.stack(D.pop("_w").to_numpy()).astype(np.float32)  # (n, L, 3)
    return D, A


def main():
    X = load()
    integer_farms = {f for f, g in X[~X.test].groupby("farm")
                     if (np.abs(g.in_temp.dropna() - np.round(g.in_temp.dropna())) < 1e-9).mean() > 0.9}
    print("integer-resolution greenhouses:", len(integer_farms))
    res = {}
    for name, L, linear in (("C1a", 6, False), ("C1b", 12, True)):
        D, A = windows(X, L)
        Ac = A - A.mean(axis=1, keepdims=True)
        rng_t = A[:, :, 0].max(1) - A[:, :, 0].min(1)
        rng_c = A[:, :, 2].max(1) - A[:, :, 2].min(1)
        nontriv = (rng_t >= 1.0) | (rng_c >= 20)
        D["day0"] = D.t0 // 24
        cand = np.where(~D.test.values)[0]
        cfarm = D.farm.values[cand]
        cday = D.day0.values[cand]
        cint = np.isin(cfarm, list(integer_farms))
        tolt = np.where(cint, TOL_T_INT, TOL[0]).astype(np.float32)
        Cc = Ac[cand]
        q_test = np.where(D.test.values & nontriv)[0]
        pool = np.where(~D.test.values & D.farm.isin(["F13", "F47"]).values & nontriv)[0]
        hrs = (D.t0.values % 24)
        # stratified baseline by start hour
        want = pd.Series(hrs[q_test]).value_counts()
        q_base = np.concatenate([RNG.choice(pool[hrs[pool] == h], size=min(k, (hrs[pool] == h).sum()), replace=False)
                                 for h, k in want.items()])
        print("\n== %s (L=%d, %s) == candidates %d, test queries %d, baseline queries %d"
              % (name, L, "slope+offset" if linear else "offset", len(cand), len(q_test), len(q_base)))
        out = {}
        for tag, Q in (("test", q_test), ("base", q_base)):
            hits = {}
            for qi in Q:
                q = Ac[qi]
                excl = (cfarm == D.farm.values[qi]) & (np.abs(cday - D.day0.values[qi]) <= 1)
                ok = ~excl
                for c in (1, 2, 0):  # hum, co2, temp
                    V = Cc[ok][:, :, c] if ok.dtype == bool else None
                    idx = np.where(ok)[0]
                    V = Cc[idx, :, c]
                    if linear:
                        vv = (V * V).sum(1)
                        b = np.clip((V @ q[:, c]) / np.maximum(vv, 1e-9), 0.5, 2.0)
                        r = np.sqrt(((q[:, c] - b[:, None] * V) ** 2).mean(1))
                    else:
                        r = np.sqrt(((q[:, c] - V) ** 2).mean(1))
                    lim = tolt[idx] if c == 0 else TOL[c]
                    keep = r <= lim
                    ok = np.zeros(len(cand), bool)
                    ok[idx[keep]] = True
                    if not ok.any():
                        break
                m = np.where(ok)[0]
                if len(m):
                    hits[qi] = [(cfarm[j], int(D.t0.values[cand[j]] - D.t0.values[qi])) for j in m]
            out[tag] = hits
            print("  %s: queries with >=1 match %d / %d (%.2f%%)" % (tag, len(hits), len(Q), 100 * len(hits) / max(len(Q), 1)))
        a, b = len(out["test"]), len(out["base"])
        table = [[a, len(q_test) - a], [b, len(q_base) - b]]
        p = fisher_exact(table, alternative="greater")[1]
        ratio = (a / len(q_test)) / max(b / len(q_base), 1e-9)

        def longest(hits, Q):
            # consecutive query windows (same farm, t0 step 1) matched to the same (cand farm, offset)
            key = defaultdict(list)
            for qi, lst in hits.items():
                for cf, off in lst:
                    key[(D.farm.values[qi], cf, off)].append(D.t0.values[qi])
            best, where = 0, None
            for k, ts in key.items():
                ts = np.sort(np.unique(ts))
                run = 1
                for i in range(1, len(ts)):
                    run = run + 1 if ts[i] == ts[i - 1] + 1 else 1
                    if run > best:
                        best, where = run, (k, int(ts[i]))
                if best == 0 and len(ts):
                    best, where = 1, (k, int(ts[0]))
            return best, where
        lt, wt = longest(out["test"], q_test)
        lb, wb = longest(out["base"], q_base)
        span_t = lt + L - 1 if lt else 0
        span_b = lb + L - 1 if lb else 0
        go = (ratio >= 3 and p < 0.001) or (span_t >= 12 and span_b < 12)
        print("  ratio %.2f, Fisher p %.4g | longest continuous match test %dh %s, baseline %dh %s -> %s"
              % (ratio, p, span_t, wt, span_b, wb, "GO" if go else "STOP"))
        tf = pd.Series([cf for lst in out["test"].values() for cf, _ in lst]).value_counts().head(8)
        bf = pd.Series([cf for lst in out["base"].values() for cf, _ in lst]).value_counts().head(8)
        print("  matched candidate greenhouses (test):", tf.to_dict())
        print("  matched candidate greenhouses (base):", bf.to_dict())
        ex = sorted(out["test"].items(), key=lambda kv: -len(kv[1]))[:6]
        for qi, lst in ex:
            print("   test %s day %d h %d -> %s" % (D.farm.values[qi], D.t0.values[qi] // 24, D.t0.values[qi] % 24,
                                                   lst[:4]))
        res[name] = go
    print("\nC1 decision:", res)


if __name__ == "__main__":
    main()
