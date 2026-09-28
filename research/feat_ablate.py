# -*- coding: utf-8 -*-
"""Group-wise leave-one-out ablation on the geometry folds.

Usage
-----
  python feat_ablate.py ec                       full EC model, all groups
  python feat_ablate.py temp fast                cheap surrogate, all groups
  python feat_ablate.py temp confirm g1,g2,g3    submitted temp model, listed
                                                 groups only

For every physical group G defined in feat_lib, refit on (view - G) and compare
out-of-fold RMSE with the full view.  Both fold sets A and B are reported.

About the two fold sets: A and B are two PLACEMENTS of the same test-shaped
layout, not independent samples.  The union of days they hold out overlaps by
66 days (Jaccard 0.584; ~28% overlap fold-for-fold).  Agreement between A and
B therefore means "robust to where the blocks sit", not "replicated on fresh
data" -- a candidate that wins on A alone was fitted to the placement.

Cost control for sub_temp.  The submitted temp model is LightGBM with 1200
trees; one candidate costs ~530 s over both fold sets, so 15 group ablations
would take hours.  `fast` swaps in a 300-tree / lr 0.10 / single-seed surrogate
and re-measures ITS OWN baseline, so only deltas against that baseline are ever
quoted.  Surrogate absolute RMSE is not comparable to the submitted model's and
is never mixed into the same table.  `confirm` then re-measures the groups that
screened large at the real 1200-tree / 3-seed setting.

Sign convention: delta = RMSE(without group) - RMSE(full view).
  delta > 0  -> removing it hurts  -> the group carries information
  delta < 0  -> removing it helps  -> the group is harmful
"""
import json
import sys
import time

import numpy as np

import env  # noqa: F401  MUST be first
from harness import load, score, views
import feat_lib as L

N_BOOT = dict(fast=600, full=2000)


def run(target, mode="full", only=None):
    panel, lab_t, lab_e = load()
    v = views(panel)
    if target == "temp":
        lab, tgt, cols = lab_t, "sub_temp", v["temp"]
        if mode == "fast":
            model, seeds, tag = L.TEMP_FAST, (7,), "temp_fast"
        else:
            model, seeds, tag = L.TEMP_MODEL, (7, 101, 2024), "temp_full"
    else:
        lab, tgt, cols = lab_e, "sub_ec", v["ec"]
        model, seeds, tag = L.EC_MODEL, (7, 101, 2024), "ec"
    groups = L.grouping(cols)
    if only:
        groups = {g: c for g, c in groups.items() if g in only}

    print("target=%s  mode=%s  features=%d  groups=%d  seeds=%s"
          % (tgt, mode, len(cols), len(groups), seeds))
    for g, cs in L.grouping(cols).items():
        print("   %-11s %2d  %s" % (g, len(cs), ", ".join(cs)))

    base = {}
    for kind in ("A", "B"):
        (r, sd, per), oof = score(lab, tgt, cols, model, kind=kind,
                                  seeds=seeds, return_oof=True)
        base[kind] = dict(rmse=r, std=sd, per=per, oof=oof)
        line = "BASE[%s] %s  rmse %.4f  foldstd %.4f" % (tag, kind, r, sd)
        if target == "ec":
            line += "  dayRMSE %.4f" % L.day_rmse(lab, tgt, oof)
        print(line, flush=True)

    rows = []
    for g, cs in groups.items():
        keep = [c for c in cols if c not in set(cs)]
        rec = dict(group=g, n=len(cs), cols=cs)
        t0 = time.time()
        for kind in ("A", "B"):
            (r, sd, per), oof = score(lab, tgt, keep, model, kind=kind,
                                      seeds=seeds, return_oof=True)
            d, lo, hi, pw = L.paired_block_boot(lab, tgt, base[kind]["oof"], oof,
                                                n_boot=N_BOOT.get(mode, 2000))
            rec["rmse_" + kind] = r
            rec["std_" + kind] = sd
            rec["d_" + kind] = r - base[kind]["rmse"]
            rec["ci_" + kind] = [lo, hi]
            rec["pworse_" + kind] = pw
            if target == "ec":
                dd, dlo, dhi, dpw = L.paired_block_boot(
                    lab, tgt, base[kind]["oof"], oof, level="day")
                rec["dday_" + kind] = dd
                rec["ciday_" + kind] = [dlo, dhi]
        rec["sec"] = time.time() - t0
        rows.append(rec)
        msg = ("drop %-11s n=%2d | A %.4f (%+.4f, CI %+.4f..%+.4f, sd %.3f) "
               "| B %.4f (%+.4f, CI %+.4f..%+.4f, sd %.3f)"
               % (g, rec["n"], rec["rmse_A"], rec["d_A"], rec["ci_A"][0],
                  rec["ci_A"][1], rec["std_A"], rec["rmse_B"], rec["d_B"],
                  rec["ci_B"][0], rec["ci_B"][1], rec["std_B"]))
        if target == "ec":
            msg += " | dayA %+.4f dayB %+.4f" % (rec["dday_A"], rec["dday_B"])
        print(msg, flush=True)

    out = dict(target=tgt, mode=mode, tag=tag, n_features=len(cols),
               seeds=list(seeds),
               base={k: dict(rmse=base[k]["rmse"], std=base[k]["std"],
                             per=base[k]["per"]) for k in base},
               groups=rows)
    path = env.LOCAL + "/ablate_%s.json" % tag
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    print("\nwrote", path)

    print("\n--- ranked by mean(dA, dB), most-harmful-to-remove first ---")
    for r in sorted(rows, key=lambda r: -(r["d_A"] + r["d_B"]) / 2):
        print("  %-11s n=%2d  dA %+.4f  dB %+.4f  mean %+.4f  (foldstd ~%.3f)"
              % (r["group"], r["n"], r["d_A"], r["d_B"],
                 (r["d_A"] + r["d_B"]) / 2, (r["std_A"] + r["std_B"]) / 2))


if __name__ == "__main__":
    a = sys.argv[1:]
    t = a[0] if a else "ec"
    m = a[1] if len(a) > 1 else "full"
    o = set(a[2].split(",")) if len(a) > 2 else None
    run(t, m, o)
