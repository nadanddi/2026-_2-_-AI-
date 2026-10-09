# -*- coding: utf-8 -*-
"""MX2: gate improvements on the saved MX1 predictions (no refit).  (2026-10-09 집 클로드, user: "문지기 개선 실험 진행해").
Fixed before running.  MX1 lessons: N (no-high model) was not better than BASE on normal days (.109 vs .108), so the
fallback should be BASE; the hourly gate g was noisy (false spikes on normal days) and under-confident on high days
(mean g .41).
Inputs per row and seed s (mx1_ckpt): BASE_s, H_s (high-only R3), g_s (P(high day | inputs 0..h)).
Candidates (k = 3, alpha = .025 / 3 = .0083):
  C1  (1 - g) BASE + g H
  C2  (1 - gc) BASE + gc H,  gc = expanding mean of g over hours 0..h of the same day (causal day-level gate)
  C3  H if gc >= .5 else BASE   (hard gate on the day-level gate)
Folds/sets as MX1.  RULE per candidate: every seed better than BASE on DIAG10, A, B (9/9, ALL rows) AND DIAG10
seed-mean (farm, day // 5) block bootstrap share(not better) < .0083.  GUARD: EL1 or P2LOO pass-2 worse >= 2 % -> hold.
Reported: normal / high days, pass-2, days better, top-5 concentration, ORACLE (true high -> H, else BASE).
Run:  cd 집/클로드/research && PYTHONPATH="" py -3.12 -u mx2_gate_improvement_v1.py
"""
import env  # noqa: F401
import importlib.util, os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.argv = ["x"]
spec = importlib.util.spec_from_file_location("ct1", os.path.join(HERE, "ct1_thermal_schedule_features_v1.py"))
ct1 = importlib.util.module_from_spec(spec); spec.loader.exec_module(ct1)
SEEDS = (3737, 5858, 7979)
CK = os.path.join(env.LOCAL, "mx1_ckpt")
ALPHA = .025 / 3
r = lambda e: float(np.sqrt(np.mean(np.square(e))))


def main():
    G = pd.concat([pd.read_csv(os.path.join(CK, f)) for f in sorted(os.listdir(CK))], ignore_index=True)
    nf = G.groupby("validator").validation_fold.nunique().to_dict(); print("folds:", nf)
    G = G.sort_values(["validator", "validation_fold", "farm", "day", "hour"]).reset_index(drop=True)
    G["dm"] = G.groupby(["validator", "validation_fold", "farm", "day"]).sub_ec.transform("mean")
    key = [G.validator, G.validation_fold, G.farm, G.day]
    for s in SEEDS:
        g = G["g_%d" % s]; gc = g.groupby(key).transform(lambda x: x.expanding().mean())
        G["C1_%d" % s] = (1 - g) * G["BASE_%d" % s] + g * G["H_%d" % s]
        G["C2_%d" % s] = (1 - gc) * G["BASE_%d" % s] + gc * G["H_%d" % s]
        G["C3_%d" % s] = np.where(gc >= .5, G["H_%d" % s], G["BASE_%d" % s])
        G["ORA_%d" % s] = np.where(G.dm >= 1.2, G["H_%d" % s], G["BASE_%d" % s])
        G["gc_%d" % s] = gc
    mean = lambda g, c: np.mean([g["%s_%d" % (c, s)] for s in SEEDS], axis=0)
    for c in ("C1", "C2", "C3"):
        print("\n==== %s vs BASE" % c)
        cells, share, guard = [], None, []
        for v, part in (("DIAG10", "all"), ("A", "all"), ("B", "all"), ("EL1", "pass2"), ("P2LOO", "pass2"),
                        ("DIAG10", "normal"), ("DIAG10", "high"), ("DIAG10", "pass2")):
            g = G[G.validator == v]
            if part == "pass2": g = g[g.day >= 179]
            if part == "normal": g = g[g.dm < 1.2]
            if part == "high": g = g[g.dm >= 1.2]
            if g.empty:
                continue
            y = g.sub_ec.to_numpy(float)
            sr = [(r(g["BASE_%d" % s] - y), r(g["%s_%d" % (c, s)] - y)) for s in SEEDS]
            mB, mC, mO = mean(g, "BASE"), mean(g, c), mean(g, "ORA")
            d = r(mC - y) / r(mB - y) - 1
            print("  %-6s %-6s days %3d  BASE %.4f  %s %.4f (%+.1f%%) seeds %s | ORACLE %.4f | mean day-gate %.3f" % (
                v, part, g[["farm", "day"]].drop_duplicates().shape[0], r(mB - y), c, r(mC - y), 100 * d,
                "".join("+" if b < a else "-" for a, b in sr), r(mO - y), mean(g, "gc").mean()))
            if part == "all":
                cells += [b < a for a, b in sr]
            if (v, part) == ("DIAG10", "all"):
                share = ct1.boot_share(g, (mB - y) ** 2, (mC - y) ** 2)
                D = g.assign(gain=(mB - y) ** 2 - (mC - y) ** 2).groupby(["farm", "day"]).gain.sum().sort_values(ascending=False)
                print("          days better %d / %d, top-5 share %.0f%%" % ((D > 0).sum(), len(D), 100 * D.head(5).sum() / D.sum() if D.sum() > 0 else np.nan))
            if v in ("EL1", "P2LOO") and d >= .02:
                guard.append(v)
        complete = nf.get("DIAG10") == 10 and nf.get("A") == 5 and nf.get("B") == 5
        ok = complete and len(cells) == 9 and all(cells) and share < ALPHA
        print("  VERDICT %s: %d/9, DIAG10 share %.4f -> %s%s" % (c, sum(cells), share if share is not None else np.nan,
              "PASS" if ok else "FAIL", ("  GUARD HOLD %s" % guard) if guard else ("" if nf.get("P2LOO") == 46 else "  (guard sets not finished)")))


if __name__ == "__main__":
    main()
