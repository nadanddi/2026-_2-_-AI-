# -*- coding: utf-8 -*-
"""TS2: trust TabPFN less where the members DISAGREE (2026-10-10 집 클로드, user: "그동안 다른 것도 뭐라도 해봐").
TD34: TabPFN overpredicts cold-side mornings (EXT12 F13 +0.73) while CODEX does not -> member disagreement may flag rows
where the PFN term hurts.  Uses stored predictions only (allowed: functions of inputs).  Fixed before running.
  D1: PFN weight multiplied by exp(-|PFN - CODEX| / 1.0 C); the removed weight goes to CODEX.
  D2: rows where the sd of (BASE, CODEX, PFN) > 0.5 C use the median of the three; other rows W40G-S.
k = 2, alpha .0125.  Scope WARM (acts where PFN has weight, g > 0) -> primary DIAG10: both BASE seeds x both PFN families
  better and farm x 5-day block bootstrap P(worse) < .0125; EXT10/EXT12/EL1 fail iff seed-mean worse and share(better)
  < .0125.  EXTRA guard (fixed now): on the three fresh layouts F/G/H (stored members) all-rows seed-mean must not be worse.
  Rough days excluded.  Pass-2 reported.
Run:  cd 집/클로드/research && PYTHONPATH="" py -3.12 -u ts2_member_disagreement_v1.py
"""
import env  # noqa: F401
import os
import numpy as np, pandas as pd
RD = set(map(tuple, pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "rw1_rough_days_v1.csv"))[["farm", "day"]].values))
L = lambda d, pre="": pd.concat([pd.read_csv(os.path.join(env.LOCAL, d, f)) for f in sorted(os.listdir(os.path.join(env.LOCAL, d))) if f.startswith(pre) and f.endswith(".csv")], ignore_index=True)
r = lambda e: float(np.sqrt(np.mean(np.square(e))))
ALPHA = .0125


def blends(X, b, c, p):
    g = np.where(X.in_temp.isna(), 1, np.clip((X.in_temp - 8) / 2, 0, 1))
    B, C, P = X[b].values, X[c].values, X[p].values
    ref = .4 * B + (.2 + .4 * (1 - g)) * C + .4 * g * P
    wP = .4 * g * np.exp(-np.abs(P - C) / 1.0)
    d1 = .4 * B + (1 - .4 - wP) * C + wP * P
    sd = np.std(np.c_[B, C, P], axis=1); med = np.median(np.c_[B, C, P], axis=1)
    d2 = np.where(sd > .5, med, ref)
    return ref, {"D1": d1, "D2": d2}


def boot(X, a, b, rng):
    cl = (X.farm + "_" + (X.day // 5).astype(str)).values
    dd = pd.Series((b - X.sub_temp.values) ** 2 - (a - X.sub_temp.values) ** 2).groupby(cl).agg(["sum", "count"])
    idx = rng.integers(0, len(dd), (20000, len(dd)))
    return float(((dd["sum"].values[idx].sum(1) / dd["count"].values[idx].sum(1)) >= 0).mean())


def main():
    rng = np.random.default_rng(20261023)
    G = L("tt1_ckpt"); G = G[[(f, d) not in RD for f, d in zip(G.farm, G.day)]]
    fresh = {"F": (L("tt12_ckpt", "fresh_"), (202, 303), "base_%d"), "G": (L("tt13_ckpt", "G_"), (404, 505), "base_R_%d"), "H": (L("tt17_ckpt", "H_"), (606, 707), "base_R_%d")}
    res = {}
    for v in ("DIAG10", "EXT10", "EXT12", "EL1"):
        X = G[G.validator == v]; y = X.sub_temp.values
        for s in (7, 101):
            for f in "AB":
                ref, C = blends(X, "base_REF_%d" % s, "codex_REF", "pfn_%s" % f)
                for k, b in C.items():
                    res.setdefault((k, v), []).append((r(b - y) / r(ref - y) - 1, boot(X, ref, b, rng), r((b - y)[X.day.values >= 179]) / r((ref - y)[X.day.values >= 179]) - 1))
    for Lname, (M, seeds, bc) in fresh.items():
        M = M[[(f, d) not in RD for f, d in zip(M.farm, M.day)]]; y = M.sub_temp.values
        cc = "codex_R"
        for s in seeds:
            for f in "AB":
                ref, C = blends(M, bc % s, cc, "pfn_%s" % f)
                for k, b in C.items():
                    res.setdefault((k, Lname), []).append((r(b - y) / r(ref - y) - 1, np.nan, r((b - y)[M.day.values >= 179]) / r((ref - y)[M.day.values >= 179]) - 1))
    for k in ("D1", "D2"):
        ok = True
        print("\n==== %s" % k)
        for v in ("DIAG10", "EXT10", "EXT12", "EL1", "F", "G", "H"):
            a = np.array(res[(k, v)]); m = a[:, 0].mean()
            line = "  %-6s mean %+.2f%% [%+.2f, %+.2f] 2차 %+.2f%%" % (v, 100 * m, 100 * a[:, 0].min(), 100 * a[:, 0].max(), 100 * a[:, 2].mean())
            if v == "DIAG10":
                o = (a[:, 0] < 0).all() and np.nanmax(a[:, 1]) < ALPHA; line += " P max %.4f" % np.nanmax(a[:, 1])
            elif v in ("EXT10", "EXT12", "EL1"):
                o = not (m > 0 and (1 - np.nanmin(a[:, 1])) < ALPHA); line += " P max %.4f" % np.nanmax(a[:, 1])
            else:
                o = m <= 0
            ok &= o; print(line + (" ok" if o else " FAIL"))
        print("VERDICT %s: %s" % (k, "PASS" if ok else "FAIL"))


if __name__ == "__main__":
    main()
