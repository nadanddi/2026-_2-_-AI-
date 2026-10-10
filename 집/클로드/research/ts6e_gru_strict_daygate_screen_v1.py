# -*- coding: utf-8 -*-
"""TS6e (SCREEN): stricter day gate for the GRU member (2026-10-10 집 클로드).  TS6d: GWD (gate threshold 8 C) improved
layout I -0.78% (P .032) but EXT12 still +1.07%.  Post-hoc threshold search -> SCREEN only; a pass promotes to a confirmation on
a NEW layout J.  Fixed before running.
Candidates (k = 2): blend = W40G-S + 0.1 * g * h_T * (GRU - W40G-S),  h_T = clip((m_t - T) / 2, 0, 1), T in {10, 12};
  m_t = min of in_temp over hours 0..t of the record day.
PROMOTE iff: layouts F, G, H, I each all-rows seed-mean better; pooled F+G+H+I block bootstrap P(worse) < .0125;
  EXT10 and EXT12 not significantly worse (fail iff seed-mean worse and share(better) < .0125).
Run:  cd 집/클로드/research && PYTHONPATH="" py -3.12 -u ts6e_gru_strict_daygate_screen_v1.py
"""
import env  # noqa: F401
import os
import numpy as np, pandas as pd
RD = set(map(tuple, pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "rw1_rough_days_v1.csv"))[["farm", "day"]].values))
L = lambda d, pre="": pd.concat([pd.read_csv(os.path.join(env.LOCAL, d, f)) for f in sorted(os.listdir(os.path.join(env.LOCAL, d))) if f.startswith(pre) and f.endswith(".csv")], ignore_index=True)
r = lambda e: float(np.sqrt(np.mean(np.square(e)))); ALPHA = .0125


def gates(X, T):
    g = np.where(X.in_temp.isna(), 1, np.clip((X.in_temp - 8) / 2, 0, 1))
    m = X.sort_values(["farm", "day", "hour"]).groupby(["farm", "day"]).in_temp.cummin().reindex(X.index)
    return g, np.where(m.isna(), 1, np.clip((m - T) / 2, 0, 1))


def boot_parts(parts, rng):
    sm, n = [], []
    for X, a, b in parts:
        cl = (X.farm + "_" + (X.day // 5).astype(str)).values
        dd = pd.Series((b - X.sub_temp.values) ** 2 - (a - X.sub_temp.values) ** 2).groupby(cl).agg(["sum", "count"])
        sm.append(dd["sum"].values); n.append(dd["count"].values)
    sm, n = np.concatenate(sm), np.concatenate(n); idx = rng.integers(0, len(sm), (20000, len(sm)))
    return float(((sm[idx].sum(1) / n[idx].sum(1)) >= 0).mean())


def main():
    rng = np.random.default_rng(20261029)
    LAY = {"F": (L("tt12_ckpt", "fresh_").merge(L("ts6b_ckpt", "F_"), on="row_id"), (202, 303), "base_%d"),
           "G": (L("tt13_ckpt", "G_").merge(L("ts6b_ckpt", "G_"), on="row_id"), (404, 505), "base_R_%d"),
           "H": (L("tt17_ckpt", "H_").merge(L("ts6b_ckpt", "H_"), on="row_id"), (606, 707), "base_R_%d"),
           "I": (L("tt23_ckpt", "I_").merge(L("ts6d_ckpt", "I_"), on="row_id"), (808, 909), "base_R_%d")}
    O = L("tt1_ckpt").merge(L("ts6_ckpt"), on=["validator", "row_id"])
    for T in (10, 12):
        print("\n==== day gate T=%d" % T); ok = True; parts = []
        for nm, (X, seeds, bc) in LAY.items():
            X = X[[(f, d) not in RD for f, d in zip(X.farm, X.day)]].reset_index(drop=True); g, h = gates(X, T); y = X.sub_temp.values; rel = []; r2 = []
            rc = "codex_R"
            for s in seeds:
                for f in "AB":
                    a = (.4 * X[bc % s] + (.2 + .4 * (1 - g)) * X[rc] + .4 * g * X["pfn_%s" % f]).values; b = a + .1 * g * h * (X.GRU.values - a)
                    rel.append(r(b - y) / r(a - y) - 1); L2 = X.day.values >= 179; r2.append(r((b - y)[L2]) / r((a - y)[L2]) - 1)
                    if s == seeds[0] and f == "A":
                        parts.append((X, a, b))
            ok &= np.mean(rel) < 0
            print("  layout %s all %+.2f%% [%+.2f, %+.2f] 2차 %+.2f%% | GRU 사용 비중(g*h>0) %.0f%%" % (nm, 100 * np.mean(rel), 100 * min(rel), 100 * max(rel), 100 * np.mean(r2), 100 * (g * h > 0).mean()))
        p = boot_parts(parts, rng); ok &= p < ALPHA; print("  pooled P(worse) %.4f" % p)
        for v in ("EXT10", "EXT12", "DIAG10", "EL1"):
            X = O[O.validator == v]; X = X[[(f, d) not in RD for f, d in zip(X.farm, X.day)]].reset_index(drop=True); g, h = gates(X, T); y = X.sub_temp.values; rel = []; parts2 = []
            for s in (7, 101):
                for f in "AB":
                    a = (.4 * X["base_REF_%d" % s] + (.2 + .4 * (1 - g)) * X.codex_REF + .4 * g * X["pfn_%s" % f]).values; b = a + .1 * g * h * (X.GRU.values - a)
                    rel.append(r(b - y) / r(a - y) - 1); parts2.append((X, a, b))
            pw = [boot_parts([pp], rng) for pp in parts2]
            if v in ("EXT10", "EXT12"):
                o = not (np.mean(rel) > 0 and (1 - min(pw)) < ALPHA); ok &= o
            else:
                o = True
            print("  %-6s all %+.2f%% P max %.4f %s" % (v, 100 * np.mean(rel), max(pw), ("ok" if o else "FAIL") if v in ("EXT10", "EXT12") else "(report)"))
        print("SCREEN T=%d: %s" % (T, "PROMOTE to layout-J confirmation" if ok else "stop"))


if __name__ == "__main__":
    main()
