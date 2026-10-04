# -*- coding: utf-8 -*-
"""ND0 (descriptive; 2026-10-04 집 클로드).  Where does the error on NORMAL days
(label day mean < 1) come from?  DIAG10 R3S seed-mean OOF (local/ec2_DC5_oof.csv).
1. SSE split: day-level (day mean error) vs within-day (hourly deviation from the day mean).
2. Within-day: by hour; true intraday swing vs predicted swing.
3. Day-level: bias / RMSE / SSE share by farm x pass, record role, 동, sealed fraction,
   EC level bin, month-like season block.
4. Day ranking: how concentrated is the SSE (top 10 / 20 % of days).
5. Oracles: RMSE if day-level error were 0; if within-day error were 0."""
import env  # noqa: F401
import os
import numpy as np, pandas as pd

O = pd.read_csv(os.path.join(env.LOCAL, "ec2_DC5_oof.csv")); G = O[O.validator == "DIAG10"].copy()
G["p"] = G[["r3s_7", "r3s_101", "r3s_2024"]].mean(axis=1); G["e"] = G.p - G.sub_ec
R = pd.read_csv(os.path.join(env.LOCAL, "st_dong_assign_v1.csv"))
G = G.merge(R[["farm", "day", "role", "dong"]], on=["farm", "day"], how="left")
G["ymean"] = G.groupby(["farm", "day"]).sub_ec.transform("mean")
G["pmean"] = G.groupby(["farm", "day"]).p.transform("mean")
G["eday"] = G.pmean - G.ymean; G["ein"] = G.e - G.eday
r = lambda e: float(np.sqrt(np.mean(np.square(e))))
N = G[G.ymean < 1].copy()
tot = (G.e ** 2).sum()
print("DIAG10 R3S: all RMSE %.4f | normal days %d RMSE %.4f (SSE share of all %.0f%%)" % (
    r(G.e), N.groupby(["farm", "day"]).ngroups, r(N.e), 100 * (N.e ** 2).sum() / tot))

print("\n1. normal-day SSE split")
sd, si = (N.eday ** 2).sum(), (N.ein ** 2).sum()
print("   day-level %.0f%%  within-day %.0f%%   (RMSE if day-level error 0: %.4f, if within-day 0: %.4f)" % (
    100 * sd / (sd + si), 100 * si / (sd + si), r(N.ein), r(N.eday)))
N["ydev"], N["pdev"] = N.sub_ec - N.ymean, N.p - N.pmean
print("   true within-day SD %.3f, predicted within-day SD %.3f, corr(true dev, pred dev) %.2f" % (
    N.ydev.std(), N.pdev.std(), N.ydev.corr(N.pdev)))
print("   between-day SD of true day mean %.3f, of day error %.3f" % (
    N.groupby(["farm", "day"]).ymean.first().std(), N.groupby(["farm", "day"]).eday.first().std()))

print("\n2. within-day error by hour (RMSE of within-day part, mean true deviation)")
h = N.groupby("hour").agg(rmse_in=("ein", r), ydev=("ydev", "mean"), pdev=("pdev", "mean"))
print(h.round(3).T.to_string())

D = N.groupby(["farm", "day"]).agg(y=("ymean", "first"), eday=("eday", "first"), role=("role", "first"),
                                   dong=("dong", "first"), sealed=("sealed", "mean"),
                                   sse=("e", lambda e: float((e ** 2).sum())), sse_in=("ein", lambda e: float((e ** 2).sum()))).reset_index()
D["pass"] = np.where(D.day >= 179, "p2", "p1")
D["lev"] = pd.cut(D.y, [0, .3, .5, .7, 1.0], labels=["<.3", ".3-.5", ".5-.7", ".7-1"])
D["sealg"] = pd.cut(D.sealed, [-.01, .2, .8, 1.01], labels=["open", "mixed", "sealed"])
D["blk"] = (D.day // 30) * 30
S = D.sse.sum()


def grp(col):
    g = D.groupby(col, observed=True).agg(days=("y", "size"), y=("y", "mean"), bias=("eday", "mean"),
                                          day_rmse=("eday", r), sse=("sse", "sum"), sse_in=("sse_in", "sum"))
    g["share%"] = 100 * g.sse / S; g["per_day%"] = g["share%"] / (100 * g.days / len(D))
    g["within%"] = 100 * g.sse_in / g.sse
    print("\n3. by %s" % col); print(g.drop(columns=["sse", "sse_in"]).round(3).to_string())


for c in (["farm", "pass"], "role", "dong", "sealg", "lev", "blk"):
    grp(c)

print("\n4. concentration")
Ds = D.sort_values("sse", ascending=False); cs = Ds.sse.cumsum() / S
print("   top 10 days %.0f%% of normal SSE, top 20%% of days (%d) %.0f%%" % (
    100 * cs.iloc[9], int(.2 * len(D)), 100 * cs.iloc[int(.2 * len(D)) - 1]))
print(Ds.head(15)[["farm", "day", "role", "dong", "pass", "y", "eday", "sealed", "sse"]].round(3).to_string(index=False))
D.to_csv(os.path.join(env.LOCAL, "nd0_normal_days_v1.csv"), index=False)
