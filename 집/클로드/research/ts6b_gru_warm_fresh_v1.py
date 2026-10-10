# -*- coding: utf-8 -*-
"""TS6b: GRU member on the WARM side only, confirmed on the fresh layouts F/G/H where the GRU was never evaluated
(2026-10-10 집 클로드).  TS6 (0.1 GRU on every row): DIAG10 -0.76% (P .0295), EL1 -1.49%, but EXT10 +0.94% / EXT12 +2.81%
(the network cannot extrapolate to cold, like TabPFN).  Post-hoc variant motivated by that pattern, so judged ONLY on layouts
F/G/H (stored BASE/CODEX/PFN members; GRU refit here with the TS6 recipe).  Fixed before running.
Candidates (k = 2, alpha .0125):  blend = W40G-S + w * g * (GRU - W40G-S),  w in {0.1, 0.2},  g = clip((in_temp-8)/2, 0, 1).
PASS iff on EACH of F, G, H all 4 cells (BASE seed x PFN family) are better on all rows AND the pooled (F+G+H) farm x 5-day
  block bootstrap P(worse) < .0125.  Pass-2 rows, F13/F47 reported.  Rough days excluded from scoring.
Run:  cd 집/클로드/research && PYTHONPATH="" py -3.12 -u ts6b_gru_warm_fresh_v1.py
"""
import os, sys, importlib.util
import env  # noqa: F401
import numpy as np, pandas as pd
sys.argv = ["x"]
HERE = os.path.dirname(os.path.abspath(__file__))


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, file)); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


S6 = load("s6", "ts6_gru_member_v1.py"); T12 = load("t12", "tt12_confirm_noh0_noheat_fresh_v1.py"); T13 = load("t13", "tt13_pass2_rule_confirm_v1.py"); T17 = load("t17", "tt17_p1_second_confirm_v1.py")
W = S6.W
CK = os.path.join(env.LOCAL, "ts6b_ckpt"); RD = S6.RD; ALPHA = .0125
FRESH = {"F": (T12.fresh_folds, T12.masks, os.path.join(env.LOCAL, "tt12_ckpt"), "fresh_", (202, 303), "base_%d"),
         "G": (T13.fresh_folds, T13.masks, os.path.join(env.LOCAL, "tt13_ckpt"), "G_", (404, 505), "base_R_%d"),
         "H": (T17.fresh_folds, T17.masks, os.path.join(env.LOCAL, "tt17_ckpt"), "H_", (606, 707), "base_R_%d")}
r = lambda e: float(np.sqrt(np.mean(np.square(e))))


def run():
    os.makedirs(CK, exist_ok=True)
    import env_extra_gpu  # noqa: F401
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    lab, pfn, ct, phc, wb, wp, wv, sets, z = W.worlds(); wb = np.asarray(wb, float)
    S = S6.sequences(lab); y = lab.sub_temp.values
    for L, (ff, mk, *_) in FRESH.items():
        for k, fd in enumerate(ff(lab)):
            path = os.path.join(CK, "%s_%d.csv" % (L, k))
            if os.path.exists(path):
                continue
            tm, vm = mk(lab, fd)
            p = np.mean([S6.fit_predict(S[tm], y[tm], wb[tm], S[vm], sd, dev) for sd in (1, 2)], 0)
            pd.DataFrame({"row_id": lab.row_id[vm].values, "GRU": p}).to_csv(path, index=False)
            print("%s/%d done (GRU alone %.3f)" % (L, k, r(p - y[vm])), flush=True)


def summarize():
    rng = np.random.default_rng(20261027)
    for w in (0.1, 0.2):
        print("\n==== GRU-WARM w=%.1f" % w); ok = True; parts = []
        for L, (_, _, ck, pre, seeds, bc) in FRESH.items():
            M = pd.concat([pd.read_csv(os.path.join(ck, f)) for f in sorted(os.listdir(ck)) if f.startswith(pre)], ignore_index=True)
            M = M.merge(pd.concat([pd.read_csv(os.path.join(CK, f)) for f in sorted(os.listdir(CK)) if f.startswith(L + "_")]), on="row_id")
            M = M[[(f, d) not in RD for f, d in zip(M.farm, M.day)]]
            g = np.where(M.in_temp.isna(), 1, np.clip((M.in_temp - 8) / 2, 0, 1)); y = M.sub_temp.values; rel = []; rel2 = []
            for s in seeds:
                for f in "AB":
                    a = (.4 * M[bc % s] + (.2 + .4 * (1 - g)) * M.codex_R + .4 * g * M["pfn_%s" % f]).values
                    b = a + w * g * (M.GRU.values - a); rel.append(r(b - y) / r(a - y) - 1)
                    L2 = M.day.values >= 179; rel2.append(r((b - y)[L2]) / r((a - y)[L2]) - 1)
                    if s == seeds[0] and f == "A":
                        parts.append((M, a, b))
            ok &= all(x < 0 for x in rel)
            fr = {fm: 100 * (r((b - y)[M.farm.values == fm]) / r((a - y)[M.farm.values == fm]) - 1) for fm in ("F13", "F47")}
            print("  layout %s all rows %+.2f%% [%+.2f, %+.2f] | pass-2 %+.2f%% | F13 %+.2f%% F47 %+.2f%% | GRU alone %.3f" % (
                L, 100 * np.mean(rel), 100 * min(rel), 100 * max(rel), 100 * np.mean(rel2), fr["F13"], fr["F47"], r(M.GRU - M.sub_temp)))
        sm, n = [], []
        for X, a, b in parts:
            cl = (X.farm + "_" + (X.day // 5).astype(str)).values
            dd = pd.Series((b - X.sub_temp.values) ** 2 - (a - X.sub_temp.values) ** 2).groupby(cl).agg(["sum", "count"])
            sm.append(dd["sum"].values); n.append(dd["count"].values)
        sm, n = np.concatenate(sm), np.concatenate(n); idx = rng.integers(0, len(sm), (20000, len(sm)))
        p = float(((sm[idx].sum(1) / n[idx].sum(1)) >= 0).mean())
        print("  pooled P(worse) %.4f -> %s" % (p, "PASS" if ok and p < ALPHA else "FAIL"))


if __name__ == "__main__":
    if os.environ.get("TS6B_SUM") != "1":
        run()
    summarize()
