# -*- coding: utf-8 -*-
"""Confirmation stage: a few candidates, measured at the real model setting,
with per-fold deltas printed so sign consistency can be judged directly.

Usage
-----
  python feat_confirm.py temp     sub_temp, submitted 1200-tree / 3-seed model:
                                  re-measures the two groups that screened as
                                  harmful on the cheap surrogate (prev_day,
                                  mem_long) individually AND together, because
                                  two redundant blocks can each look droppable
                                  while dropping both is not.
  python feat_confirm.py ec_out   sub_ec, v5 18-feature ET300/2 recipe:
                                  drops the four outdoor-weather columns.

Decision rule used throughout this study
----------------------------------------
* The fold standard deviation measures how much the five geometry folds differ
  from each other; it is an order of magnitude larger than any of these deltas
  and is NOT the uncertainty of a paired comparison.  It is printed for
  context only.
* The judgement is made on the paired greenhouse-day block bootstrap CI of the
  delta (same folds, same rows, same seeds, both models) plus the per-fold sign
  pattern, and a candidate is only promoted when A and B agree in direction.
  A and B are two placements of the same layout whose held-out day sets
  overlap (Jaccard 0.584), so agreement means robustness to placement, not
  independent replication.
"""
import json
import sys
import time

import numpy as np

import env  # noqa: F401  MUST be first
from harness import load, score, views
import feat_lib as L

ET2 = dict(n_estimators=300, max_features=1.0, min_samples_leaf=2, n_jobs=4)


def compare(lab, tgt, base_cols, cands, model, seeds, daylevel):
    base = {}
    for kind in ("A", "B"):
        (r, sd, per), oof = score(lab, tgt, base_cols, model, kind=kind,
                                  seeds=seeds, return_oof=True)
        base[kind] = dict(rmse=r, std=sd, per=per, oof=oof)
        line = "BASE %s  rmse %.4f  foldstd %.4f  folds %s" % (
            kind, r, sd, " ".join("%.4f" % x for x in per))
        if daylevel:
            line += "  dayRMSE %.4f" % L.day_rmse(lab, tgt, oof)
        print(line, flush=True)

    rows = []
    for name, cols in cands:
        rec = dict(name=name, n=len(cols))
        t0 = time.time()
        for kind in ("A", "B"):
            (r, sd, per), oof = score(lab, tgt, cols, model, kind=kind,
                                      seeds=seeds, return_oof=True)
            d, lo, hi, pw = L.paired_block_boot(lab, tgt, base[kind]["oof"], oof)
            pf = [p - b for p, b in zip(per, base[kind]["per"])]
            rec["rmse_" + kind] = r
            rec["std_" + kind] = sd
            rec["d_" + kind] = r - base[kind]["rmse"]
            rec["ci_" + kind] = [lo, hi]
            rec["perfold_" + kind] = pf
            rec["nneg_" + kind] = int(sum(1 for x in pf if x < 0))
            if daylevel:
                dd, dlo, dhi, _ = L.paired_block_boot(lab, tgt,
                                                      base[kind]["oof"], oof,
                                                      level="day")
                rec["dday_" + kind] = dd
                rec["ciday_" + kind] = [dlo, dhi]
        rec["sec"] = time.time() - t0
        rows.append(rec)
        for kind in ("A", "B"):
            msg = ("%-26s %s n=%3d | rmse %.4f  delta %+.4f  CI %+.4f..%+.4f "
                   "| per-fold %s  (%d/5 improve)"
                   % (name, kind, rec["n"], rec["rmse_" + kind],
                      rec["d_" + kind], rec["ci_" + kind][0],
                      rec["ci_" + kind][1],
                      " ".join("%+.3f" % x for x in rec["perfold_" + kind]),
                      rec["nneg_" + kind]))
            if daylevel:
                msg += "  day %+.4f" % rec["dday_" + kind]
            print(msg, flush=True)
        print("", flush=True)
    return base, rows


def main(which):
    panel, lab_t, lab_e = load()
    v = views(panel)

    if which == "temp":
        cols = v["temp"]
        g = L.grouping(cols)
        pd_, ml = set(g["prev_day"]), set(g["mem_long"])
        cands = [
            ("drop prev_day", [c for c in cols if c not in pd_]),
            ("drop mem_long", [c for c in cols if c not in ml]),
            ("drop prev_day+mem_long", [c for c in cols if c not in (pd_ | ml)]),
        ]
        print("sub_temp | submitted LGB-huber 1200 trees, seeds (7,101,2024)")
        print("prev_day(%d): %s" % (len(pd_), ", ".join(sorted(pd_))))
        print("mem_long(%d): %s" % (len(ml), ", ".join(sorted(ml))))
        base, rows = compare(lab_t, "sub_temp", cols, cands,
                             L.TEMP_MODEL, (7, 101, 2024), daylevel=False)
        tag = "confirm_temp"
    else:
        cols = v["v5"]
        out = set(L.grouping(cols)["raw_out"])
        cands = [("v5 minus raw_out (14f)", [c for c in cols if c not in out])]
        print("sub_ec | v5 recipe ET300/leaf2, seeds (7,101,2024)")
        print("raw_out(%d): %s" % (len(out), ", ".join(sorted(out))))
        base, rows = compare(lab_e, "sub_ec", cols, cands,
                             L.etf(ET2), (7, 101, 2024), daylevel=True)
        tag = "confirm_ec_out"

    path = env.LOCAL + "/%s.json" % tag
    with open(path, "w") as fh:
        json.dump(dict(base={k: dict(rmse=base[k]["rmse"], std=base[k]["std"],
                                     per=base[k]["per"]) for k in base},
                       cands=rows), fh, indent=1, default=float)
    print("wrote", path)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "temp")
