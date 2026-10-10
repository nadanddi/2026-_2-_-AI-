# -*- coding: utf-8 -*-
"""EA3 (2026-10-12 집 클로드; researcher sweep): EC anchor with MEASURED pull strength / hourly profile, swept.  Fixed before running.
Form:  pred_h = p_h + alpha * s(h) * guard( A - cm_h ),  cm_h = causal running mean of p over 0..h (W2 / EA form),
       A = linker v7 confident 1-step previous record's 23 h EC (mirror CV, local/linker_v7/cv_all400_choice_v7m.csv).
Measured on PASS-1 rows of DIAG10 (EC OOF RV1, tag CUR, column sg, seed mean), confident p >= .8:
  alpha_m = LS slope of (y_h - p_h) on (A - cm_h) over all hours;  beta_h = the same slope per hour (s(h) = beta_h / alpha_m).
Grid: alpha in {.25, .5, .75, 1, 1.25, alpha_m}; profile in {flat, meas}; p_min in {.7, .8, .9, .95}; guard in {.3, .6, inf}
  (guard: no correction where |A - cm_h| > guard; SG2 / W2 used .3)  -> 144 settings.
Selection: PASS-1 rows of DIAG10 (seed-mean RMSE).  Judgement of the selected setting vs the EA2 rule (alpha 1, flat, p .8, no
guard) on PASS-2 rows: all 3 seeds better on DIAG10, P2LOO and EL1 AND record-level bootstrap P(worse) < .025 on DIAG10.
Main effects on pass-2 reported.
Run:  cd 집/클로드/research && PYTHONPATH="" py -3.12 -u ea3_ec_anchor_measure_sweep_v1.py
"""
import os, itertools
import env  # noqa: F401
import numpy as np, pandas as pd
LK = os.path.join(env.LOCAL, "linker_v7", "cv_all400_choice_v7m.csv")
rm = lambda e: float(np.sqrt(np.mean(np.square(e))))


def prep():
    O = pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "ec_rv1_preds_v1.csv")); O = O[O.tag == "CUR"].copy()
    O["rec"] = O.row_id.str[:7]; O = O.sort_values(["validator", "fold", "seed", "rec", "hour"]).reset_index(drop=True)
    O["cm"] = O.groupby(["validator", "fold", "seed", "rec"]).sg.transform(lambda s: s.expanding().mean())
    L = pd.read_csv(LK); L = L[L.prev_is_labelled.astype(bool) & L.anchor_ec23.notna() & (L.steps_to_anchor == 1)]
    O = O.merge(L[["rec", "anchor_ec23", "p"]], on="rec", how="left")
    O["dm"] = O.groupby(["validator", "fold", "seed", "rec"]).sub_ec.transform("mean")
    return O


def apply(O, alpha, prof, p_min, guard, beta, am):
    g = O.anchor_ec23 - O.cm
    ok = O.anchor_ec23.notna() & (O.p >= p_min) & (g.abs() <= guard)
    s = np.ones(len(O)) if prof == "flat" else beta[O.hour.values] / am
    return np.where(ok, O.sg + alpha * s * g, O.sg)


def main():
    O = prep(); D = O[O.validator == "DIAG10"]; p1 = D.day < 179
    m = p1 & D.anchor_ec23.notna() & (D.p >= .8)
    g = (D.anchor_ec23 - D.cm)[m]; r = (D.sub_ec - D.sg)[m]
    am = float((g * r).sum() / (g * g).sum())
    beta = np.array([float(((g * r)[D.hour[m] == h]).sum() / ((g * g)[D.hour[m] == h]).sum()) for h in range(24)])
    print("measured on pass-1 (confident, %d rows): alpha_m %.3f | beta_h " % (m.sum(), am) + " ".join("%d:%.2f" % (h, b) for h, b in enumerate(beta)))
    grid = list(itertools.product((.25, .5, .75, 1.0, 1.25, round(am, 3)), ("flat", "meas"), (.7, .8, .9, .95), (.3, .6, np.inf)))
    rows = []
    for gset in grid:
        pr = apply(D, *gset, beta, am)
        r1 = np.mean([rm((pr - D.sub_ec)[(D.seed == s) & p1]) for s in D.seed.unique()]); r2 = np.mean([rm((pr - D.sub_ec)[(D.seed == s) & ~p1]) for s in D.seed.unique()])
        rows.append(dict(alpha=gset[0], prof=gset[1], p_min=gset[2], guard=gset[3], pass1=r1, pass2=r2))
    T = pd.DataFrame(rows); T.to_csv(os.path.join(env.LOCAL, "linker_v7", "ea3_grid.csv"), index=False)
    sb = np.isclose(T.alpha, 1) & (T.prof == "flat") & np.isclose(T.p_min, .8) & np.isinf(T["guard"]); base = T[sb].iloc[0]
    T["d1"] = 100 * (T.pass1 / base.pass1 - 1); T["d2"] = 100 * (T.pass2 / base.pass2 - 1)
    for fac in ("alpha", "prof", "p_min", "guard"):
        print("  main effect %-6s:" % fac, " ".join("%s: p1 %+.2f%% p2 %+.2f%%" % (v, gg.d1.mean(), gg.d2.mean()) for v, gg in T.groupby(fac)))
    best = T.sort_values("pass1").iloc[0]; sel = (best["alpha"], best["prof"], best["p_min"], best["guard"])
    print("selected on pass-1:", sel, "| pass1 %+.2f%% pass2 %+.2f%% vs EA2; pass-2 rank %d of %d" % (best.d1, best.d2, int((T.pass2 < best.pass2).sum()) + 1, len(T)))
    print(T.sort_values("pass1").head(6)[["alpha", "prof", "p_min", "guard", "d1", "d2"]].round(2).to_string())
    rng = np.random.default_rng(20261203); ok = True
    for v in ("DIAG10", "P2LOO", "EL1"):
        X = O[(O.validator == v) & (O.day >= 179)].copy()
        a = apply(X, *sel, beta, am); b = apply(X, 1.0, "flat", .8, np.inf, beta, am)
        cells = [rm((a - X.sub_ec)[X.seed == s]) / rm((b - X.sub_ec)[X.seed == s]) - 1 for s in X.seed.unique()]; ok &= all(c < 0 for c in cells)
        per = pd.Series((a - X.sub_ec) ** 2 - (b - X.sub_ec) ** 2).groupby([X.seed.values, X.rec.values]).sum().groupby(level=1).mean(); per = per[per.abs() > 1e-12]
        line = "  %-6s selected vs EA2, pass-2 cells %s | changed records %d, improved %d" % (v, " ".join("%+.2f%%" % (100 * c) for c in cells), len(per), int((per < 0).sum()))
        if v == "DIAG10" and len(per):
            idx = rng.integers(0, len(per), (20000, len(per))); pw = float((per.values[idx].sum(1) >= 0).mean()); ok &= pw < .025; line += " | record P(worse) %.4f" % pw
        print(line)
    print("VERDICT EA3: %s" % ("PASS" if ok else "FAIL"))


if __name__ == "__main__":
    main()
