# -*- coding: utf-8 -*-
"""TW1: trust TabPFN more on warm rows (statistical critic's priority 1,
2026-10-02).  Evaluated on Codex phase-3 OOF (lock-excluded, 22 folds,
validators DIAG10 / A / B / EXT10 / EXT12, R3 seeds 7 / 101 / 2024) without
retraining.  2026-10-02 집 클로드 (user put Claude on EC for now).

Member reconstruction (exact while the label-range clip does not bind):
  v2_s = clip(shrink(.8 r_s + .2 bag)),  r3_s = clip(shrink(r_s)),  shrink is
  linear (0.5 p + 0.5 expanding day mean)  ->  S = (v2_s - .8 r3_s) / .2 is the
  shrunk TabPFN bag (same bag for all seeds - checked), and the raw bag
  t = shrink^{-1}(S), solved hour by hour within each (validator, fold, farm,
  day).  Rows where v2_s or r3_s sit on a clip bound keep v2_s unchanged.

Variants (k = 2, fixed before running; no thresholds tuned on EC):
  G1  TabPFN weight 0.2 + 0.2 g,  g = clip((in_temp - 8) / 2, 0, 1)  (the
      temperature G_C gate form, catalog 6.67), mixed on the shrunk members:
      (0.8 - 0.2 g) r3_s + (0.2 + 0.2 g) S
  H4  Codex H4 as registered (6.96): gate = |r3_s - t| >= 0.15 and in_temp >= 10,
      there 0.75 v2_s + 0.25 t (t raw, output level), else v2_s.
Decision: a variant PASSES only if it improves pooled RMSE for all 3 seeds x 5
validators AND every seed's DIAG10 paired bootstrap (farm x 5-day blocks,
20,000 draws) gives P(worse) < 0.025 / 2.
Limitation stated up front: one TabPFN bag (contexts 1-4) for every seed, so
"seeds" vary R3 only.  A PASS needs a second independent bag (contexts 5-8)
before any use.
Descriptive extras: the 126 DIAG10 days outside Codex's public 234-day OOF
(H4 was chosen on those 234), late (>= 179) and sealed days.

Run:  cd 집/클로드/research && PYTHONPATH="" py -3.12 -u tw1_warm_tabpfn_weight_v1.py
"""
import env  # noqa: F401
import glob
import os

import numpy as np
import pandas as pd

ROOT = env.ROOT
OOF = os.path.join(ROOT, u"집", u"코덱스", "local", "ec_restart_phase3_20261001_v1", "oof_predictions.csv")
OLD = os.path.join(ROOT, u"집", u"코덱스", "analysis", "local")
SEEDS = (7, 101, 2024)
VALS = ("DIAG10", "A", "B", "EXT10", "EXT12")
RNG = np.random.default_rng(20261002)


def inv_shrink(out):
    """out_h = .5 p_h + .5 mean(p_0..p_h)  ->  p (hour-ordered array)."""
    p = np.empty(len(out))
    s = 0.0
    for h, o in enumerate(out):
        n = h + 1
        p[h] = (o - 0.5 * s / n) / (0.5 + 0.5 / n)
        s += p[h]
    return p


def main():
    o = pd.read_csv(OOF, encoding="utf-8-sig")
    tx = pd.read_csv(os.path.join(env.DATA, "train_X.csv"), usecols=["row_id", "in_temp", "act_circfan", "act_vent"])
    o = o.merge(tx, on="row_id", how="left")
    o = o.sort_values(["validator", "validation_fold", "farm", "day", "hour"]).reset_index(drop=True)
    grp = ["validator", "validation_fold", "farm", "day"]
    lo, hi = o.sub_ec.min(), o.sub_ec.max()
    S = {s: (o["v2_%d" % s] - 0.8 * o["r3_%d" % s]) / 0.2 for s in SEEDS}
    spread = np.max(np.abs(np.c_[S[7] - S[101], S[7] - S[2024]]), axis=1)
    edge = np.zeros(len(o), bool)
    for s in SEEDS:
        for c in ("v2_%d" % s, "r3_%d" % s):
            edge |= (np.abs(o[c] - lo) < 1e-9) | (np.abs(o[c] - hi) < 1e-9)
    print("rows %d; clip-bound rows %d; bag consistency across seeds (non-clip): max |diff| %.2e"
          % (len(o), edge.sum(), spread[~edge].max()))
    o["S"] = np.mean([S[s] for s in SEEDS], axis=0)
    o["t"] = np.concatenate([inv_shrink(g.S.values) for _, g in o.groupby(grp, sort=False)])
    # sanity: shrink(t) == S
    chk = []
    for _, g in o.groupby(grp, sort=False):
        t = g.t.values
        chk.append(0.5 * t + 0.5 * np.cumsum(t) / np.arange(1, len(t) + 1))
    print("inverse-shrink check max |shrink(t) - S| %.2e" % np.max(np.abs(np.concatenate(chk) - o.S.values)))
    g_ = np.clip((o.in_temp.fillna(20) - 8) / 2, 0, 1)

    for s in SEEDS:
        r3, v2 = o["r3_%d" % s], o["v2_%d" % s]
        o["G1_%d" % s] = np.where(edge, v2, np.clip((0.8 - 0.2 * g_) * r3 + (0.2 + 0.2 * g_) * o.S, lo, hi))
        gate = ((r3 - o.t).abs() >= 0.15) & (o.in_temp >= 10) & ~edge
        o["H4_%d" % s] = np.where(gate, 0.75 * v2 + 0.25 * o.t, v2)
        o["gate_%d" % s] = gate

    r = lambda e: float(np.sqrt(np.mean(np.square(e))))
    res = {}
    print("\npooled RMSE (relative change vs v2 of the same seed)")
    for var in ("G1", "H4"):
        ok_all = True
        for v in VALS:
            line = []
            for s in SEEDS:
                g = o[o.validator == v]
                a, b = r(g["v2_%d" % s] - g.sub_ec), r(g["%s_%d" % (var, s)] - g.sub_ec)
                ok_all &= b < a
                line.append("s%d %.4f->%.4f (%+.2f%%)" % (s, a, b, 100 * (b / a - 1)))
            print("  %s %-6s %s" % (var, v, "  ".join(line)))
        res[var] = ok_all

    D = o[o.validator == "DIAG10"].copy()
    D["cl"] = D.farm + "_" + (D.day // 5).astype(str)
    print("\nDIAG10 paired bootstrap (farm x 5-day blocks, 20000), threshold %.4f" % (0.025 / 2))
    for var in ("G1", "H4"):
        ps = []
        for s in SEEDS:
            d = (D["%s_%d" % (var, s)] - D.sub_ec) ** 2 - (D["v2_%d" % s] - D.sub_ec) ** 2
            cl = d.groupby(D.cl).agg(["sum", "count"])
            sm, n = cl["sum"].values, cl["count"].values
            idx = RNG.integers(0, len(sm), (20000, len(sm)))
            bs = sm[idx].sum(1) / n[idx].sum(1)
            ps.append(float((bs >= 0).mean()))
        ok = res[var] and all(p < 0.0125 for p in ps)
        print("  %s P(worse) by seed %s  all-15-cells-better %s -> %s" % (var, [round(p, 4) for p in ps], res[var],
                                                                          "PASS" if ok else "FAIL"))

    # descriptive extras on DIAG10 (seed mean)
    old = set()
    for f in sorted(glob.glob(os.path.join(OLD, "ec_three_seed_ensemble_cv", "20260928_035914", "fold*.csv"))) + \
            sorted(glob.glob(os.path.join(OLD, "ec_locked_confirmation", "20260928_044934", "fold?.csv"))):
        z = pd.read_csv(f, usecols=["farm", "day"])
        old |= set(map(tuple, z.drop_duplicates().values))
    D["old234"] = [(f, d) in old for f, d in zip(D.farm, D.day)]
    dm = D.groupby(["farm", "day"])
    sealed = (dm.act_circfan.transform("mean") < 10) & (dm.act_vent.transform(lambda x: (x == 0).mean()) > 0.85)
    print("\nDIAG10 segments (mean of seeds): old-234 days %d" % D[D.old234][["farm", "day"]].drop_duplicates().shape[0])
    for nm, m in (("all", D.sub_ec.notna()), ("126 unused days", ~D.old234), ("234 old days", D.old234),
                  ("late>=179", D.day >= 179), ("sealed", sealed), ("in_temp<10", D.in_temp < 10)):
        g = D[m]
        base = np.mean([r(g["v2_%d" % s] - g.sub_ec) for s in SEEDS])
        print("  %-16s rows %5d  v2 %.4f  G1 %.4f  H4 %.4f" % (nm, len(g), base,
                                                                 np.mean([r(g["G1_%d" % s] - g.sub_ec) for s in SEEDS]),
                                                                 np.mean([r(g["H4_%d" % s] - g.sub_ec) for s in SEEDS])))
    print("  gate rows (H4, seed 7): %d of %d DIAG10 rows" % (D.gate_7.sum(), len(D)))


if __name__ == "__main__":
    main()
