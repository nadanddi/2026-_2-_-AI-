# -*- coding: utf-8 -*-
"""TM1 (re-evaluation on a TEST-MATCHED validation set, fixed before running; 2026-10-06 집 클로드).
AX1 (6.352): evaluation days are colder than labelled pass-2 days and resemble pass-1 days, so judging
only on pass-2 labelled rows may not represent the evaluation conditions.
TM set (inputs only; no labels, no leaderboard): per farm, standardize day-level input summaries
(mean in_temp, out_temp, in_hum, act_vent, act_heating, in_co2, out_rad, day-mean / night in_temp) over
the labelled DIAG10 days; for every evaluation day take its 3 nearest labelled days; TM = their union.
Re-evaluated on TM rows (DIAG10 out-of-fold, both passes; corrections applied to ALL rows here):
  (1) DP1: R3 season+DP1 (ec3_DP1_all dp_*) vs R3S (r3s_*), seeds 7/101/2024;
  (2) SG2 (anchors of hk0_rows_v1, all days) vs R3S seed mean;
  (3) HG3 and HG4 rules on top of SG2 vs SG2 (S_low as in hg3all);
  (4) TabPFN share: Codex v2 integration season_r3 vs 0.8 season_r3 + 0.2 season_pfn (shrink as stored).
Reported: TM days / rows, RMSE change per seed (where seeds exist), normal / high split, farm x 5-day
cluster bootstrap P(worse).  Descriptive re-evaluation, not a new adoption rule."""
import env  # noqa: F401
import os
import numpy as np, pandas as pd
RAW = ["out_temp", "out_rad", "in_temp", "in_hum", "in_co2", "act_vent", "act_heating"]
tr = pd.read_csv(os.path.join(env.DATA, "train_X.csv"), usecols=["row_id"] + RAW); tr["src"] = "train"
te = pd.read_csv(os.path.join(env.DATA, "test_X.csv"), usecols=["row_id"] + RAW); te["src"] = "test"
X = pd.concat([tr, te]); X = X[X.row_id.str[:3].isin(["F13", "F47"])].copy()
X["farm"], X["day"], X["hour"] = X.row_id.str[:3], X.row_id.str[4:7].astype(int), X.row_id.str[8:10].astype(int)
g = X.groupby(["farm", "day"])
D = g[RAW].mean(); D["night_t"] = X[X.hour <= 5].groupby(["farm", "day"]).in_temp.mean(); D["src"] = g.src.first(); D = D.reset_index()
O = pd.read_csv(os.path.join(env.LOCAL, "hk0_rows_v1.csv"))
lab = set(zip(O.farm, O.day))
TM = set()
for f in ("F13", "F47"):
    L = D[(D.farm == f).values & np.array([(f, d) in lab for d in D.day])]; T = D[(D.farm == f) & (D.src == "test")]
    cols = RAW + ["night_t"]; mu, sd = L[cols].mean(), L[cols].std()
    Zl, Zt = ((L[cols] - mu) / sd).values, ((T[cols] - mu) / sd).values
    for z in Zt:
        for i in np.argsort(np.nansum((Zl - z) ** 2, axis=1))[:3]:
            TM.add((f, int(L.day.values[i])))
tmd = pd.DataFrame(sorted(TM), columns=["farm", "day"])
print("TM set: %d labelled days (F13 %d, F47 %d); pass-1 share %.2f; mean day-temp TM %.1f vs all labelled %.1f" % (
    len(tmd), (tmd.farm == "F13").sum(), (tmd.farm == "F47").sum(), (tmd.day < 179).mean(),
    D.set_index(["farm", "day"]).loc[list(TM)].in_temp.mean(), D[np.array([(f, d) in lab for f, d in zip(D.farm, D.day)])].in_temp.mean()))
inTM = lambda F: np.array([(f, d) in TM for f, d in zip(F.farm, F.day)])
r = lambda e: float(np.sqrt(np.mean(np.square(e))))


def boot(F, a, b):
    F = F.copy(); F["cl"] = F.farm + "_" + (F.day // 5).astype(str)
    dd = pd.Series((b - F.sub_ec) ** 2 - (a - F.sub_ec) ** 2, index=F.index).groupby(F.cl).agg(["sum", "count"])
    sm, n = dd["sum"].values, dd["count"].values
    idx = np.random.default_rng(20261006).integers(0, len(sm), (20000, len(sm)))
    return float(((sm[idx].sum(1) / n[idx].sum(1)) >= 0).mean())


def report(name, F, a, b):
    F = F.copy(); F["dm"] = F.groupby(["farm", "day"]).sub_ec.transform("mean"); a = np.asarray(a); b = np.asarray(b)
    parts = []
    for nm, m in (("all", np.ones(len(F), bool)), ("normal", (F.dm < 1).values), ("high", (F.dm >= 1).values)):
        if m.sum():
            parts.append("%s %.4f->%.4f (%+.1f%%)" % (nm, r(a[m] - F.sub_ec.values[m]), r(b[m] - F.sub_ec.values[m]), 100 * (r(b[m] - F.sub_ec.values[m]) / r(a[m] - F.sub_ec.values[m]) - 1)))
    print("  %-34s %s  P(worse) %.4f" % (name, " | ".join(parts), boot(F, a, b)))


print("\n(1) DP1 on TM rows (per seed, then seed mean):")
P1 = pd.read_csv(os.path.join(env.LOCAL, "ec3_DP1_all.csv")); P1 = P1[(P1.validator == "DIAG10")].copy(); P1 = P1[inTM(P1)].reset_index(drop=True)
for s in (7, 101, 2024):
    report("seed %d R3S -> R3S+DP1" % s, P1, P1["r3s_%d" % s], P1["dp_%d" % s])
report("seed mean", P1, P1[["r3s_7", "r3s_101", "r3s_2024"]].mean(axis=1), P1[["dp_7", "dp_101", "dp_2024"]].mean(axis=1))
print("\n(2)(3) SG2 / HG3 / HG4 on TM rows (R3S seed mean, corrections on all rows):")
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "hg3all_all_days_evidence_v1.py"), encoding="utf-8").read()
ns = {"__file__": os.path.join(os.path.dirname(os.path.abspath(__file__)), "hg3all_all_days_evidence_v1.py")}
exec(src.split("r = lambda")[0], ns)
H = ns["O"]; gate = ns["gate"]; am = (H.a1 + H.a2) / 2
H["hg4"] = np.where(gate, H.p + .5 * (np.minimum(am, 1.8) - H.pm), H.sg)
m = inTM(H); Ht = H[m].reset_index(drop=True)
report("R3S -> R3S+SG2", Ht, Ht.p, Ht.sg)
report("SG2 -> SG2+HG3", Ht, Ht.sg, Ht.hg3)
report("SG2 -> SG2+HG4", Ht, Ht.sg, Ht.hg4)
for ps, cond in (("pass-1 TM rows", Ht.day < 179), ("pass-2 TM rows", Ht.day >= 179)):
    Hs = Ht[cond.values].reset_index(drop=True)
    if len(Hs):
        report("  %s: R3S -> SG2" % ps, Hs, Hs.p, Hs.sg)
print("\n(4) TabPFN share on TM rows (Codex v2 integration, DIAG10; seeds stacked rows):")
V = pd.read_csv(os.path.join(env.ROOT, u"집", u"코덱스", "local", "ec_dc4_integration_20261002_v1", "v2_integration_oof.csv"))
V = V[V.validator == "DIAG10"].copy(); V = V[inTM(V)].reset_index(drop=True)
for s, G in V.groupby("seed"):
    G = G.reset_index(drop=True)
    report("seed %s R3(season) only -> v2 (0.8/0.2)" % s, G, G.season_r3, G.season_v2)
tmd.to_csv(os.path.join(env.LOCAL, "tm1_set_v1.csv"), index=False)
