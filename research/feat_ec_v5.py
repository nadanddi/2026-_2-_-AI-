# -*- coding: utf-8 -*-
"""EC triage around the *better* EC reference: the 18-feature v5 recipe.

Why this file exists
--------------------
baseline.py in this environment scores, on the geometry folds,
    submitted 68-feature blend : A 0.3649 / B 0.3372
    v5 recipe, 18 features     : A 0.2898 / B 0.2615
so the submitted EC model is dominated on both fold placements (A and B are
not independent: their held-out day sets overlap, Jaccard 0.584, so B is a
robustness check on block placement, not a replication on fresh data).
Telling which of the 68 columns are useful is therefore only half the
question; the other half is which columns are worth *adding back* to the
18-column recipe.  Anything measured on the 68-column baseline is measured on
a baseline that is already 0.075 worse, so it is re-measured here.

Usage
-----
  python feat_ec_v5.py drops     leave-one-group-out inside the 18
  python feat_ec_v5.py add18     add-one-group-back onto the 18
  python feat_ec_v5.py add14     add-one-group-back onto the 14 (v5 minus the
                                 four outdoor-weather columns, once
                                 feat_confirm.py ec_out has cleared that drop)

The addable pool is every sub_ec column of features_v2 that is not already in
the recipe, grouped by physical role, PLUS the five new derived blocks built
by feat_new.py (hinge / rep24 / event / dew / duty).  Those five were only
ever measured on the 68-column baseline; the question here is whether they
still pay on the far leaner and far better baseline.
"""
import json
import sys
import time

import numpy as np

import env  # noqa: F401  MUST be first
from harness import load, score, views
import feat_lib as L
import feat_new as N

ET2 = dict(n_estimators=300, max_features=1.0, min_samples_leaf=2, n_jobs=4)
V5_MODEL = L.etf(ET2)
SEEDS = (7, 101, 2024)


def main(mode, only=None):
    panel, lab_t, lab_e, blocks, new = N.attach()
    v = views(panel)
    tgt = "sub_ec"

    base_cols = list(v["v5"])
    if mode == "add14":
        out = set(L.grouping(base_cols)["raw_out"])
        base_cols = [c for c in base_cols if c not in out]

    pool = [c for c in v["ec_full"] if c not in set(base_cols)]
    groups = L.grouping(pool)
    for b, cs in blocks.items():
        groups["NEW_" + b] = list(cs)
    if mode == "drops":
        groups = L.grouping(base_cols)
    if only:
        groups = {g: c for g, c in groups.items() if g in only}

    print("mode=%s | base %d cols: %s" % (mode, len(base_cols),
                                          ", ".join(base_cols)))
    print("candidate groups: %d" % len(groups))
    for g, cs in groups.items():
        print("   %-14s %2d  %s" % (g, len(cs), ", ".join(cs)))

    base = {}
    for kind in ("A", "B"):
        (r, sd, per), oof = score(lab_e, tgt, base_cols, V5_MODEL, kind=kind,
                                  seeds=SEEDS, return_oof=True)
        base[kind] = dict(rmse=r, std=sd, per=per, oof=oof)
        print("BASE %s %.4f (foldstd %.4f)  dayRMSE %.4f  folds %s"
              % (kind, r, sd, L.day_rmse(lab_e, tgt, oof),
                 " ".join("%.4f" % x for x in per)), flush=True)

    rows = []
    for g, cs in groups.items():
        if mode == "drops":
            cols = [c for c in base_cols if c not in set(cs)]
            label = "drop:" + g
        else:
            cols = base_cols + list(cs)
            label = "add:" + g
        rec = dict(name=label, n=len(cols), n_group=len(cs))
        t0 = time.time()
        for kind in ("A", "B"):
            (r, sd, per), oof = score(lab_e, tgt, cols, V5_MODEL, kind=kind,
                                      seeds=SEEDS, return_oof=True)
            d, lo, hi, _ = L.paired_block_boot(lab_e, tgt, base[kind]["oof"], oof)
            dd, dlo, dhi, _ = L.paired_block_boot(lab_e, tgt, base[kind]["oof"],
                                                  oof, level="day")
            pf = [p - b for p, b in zip(per, base[kind]["per"])]
            rec["rmse_" + kind], rec["std_" + kind] = r, sd
            rec["d_" + kind] = r - base[kind]["rmse"]
            rec["ci_" + kind] = [lo, hi]
            rec["dday_" + kind] = dd
            rec["ciday_" + kind] = [dlo, dhi]
            rec["perfold_" + kind] = pf
            rec["nneg_" + kind] = int(sum(1 for x in pf if x < 0))
        rec["sec"] = time.time() - t0
        rows.append(rec)
        print("%-18s n=%3d | A %.4f (%+.4f CI %+.4f..%+.4f, %d/5 better, "
              "day %+.4f) | B %.4f (%+.4f CI %+.4f..%+.4f, %d/5 better, "
              "day %+.4f)"
              % (label, rec["n"], rec["rmse_A"], rec["d_A"], rec["ci_A"][0],
                 rec["ci_A"][1], rec["nneg_A"], rec["dday_A"],
                 rec["rmse_B"], rec["d_B"], rec["ci_B"][0], rec["ci_B"][1],
                 rec["nneg_B"], rec["dday_B"]), flush=True)

    path = env.LOCAL + "/ec_v5_%s.json" % mode
    with open(path, "w") as fh:
        json.dump(dict(mode=mode, base_cols=base_cols,
                       base={k: dict(rmse=base[k]["rmse"], std=base[k]["std"],
                                     per=base[k]["per"]) for k in base},
                       trials=rows), fh, indent=1, default=float)
    print("\nwrote", path)
    print("\n--- ranked by mean(dA, dB), best first ---")
    for r in sorted(rows, key=lambda r: (r["d_A"] + r["d_B"]) / 2):
        both = "BOTH" if (r["d_A"] < 0) == (r["d_B"] < 0) else "SPLIT"
        print("  %-18s dA %+.4f  dB %+.4f  mean %+.4f  %s"
              % (r["name"], r["d_A"], r["d_B"], (r["d_A"] + r["d_B"]) / 2, both))


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0] if a else "add18",
         set(a[1].split(",")) if len(a) > 1 else None)
