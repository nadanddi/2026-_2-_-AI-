# -*- coding: utf-8 -*-
"""sub_temp: rebuild the view around the 74-column base, then add new blocks.

Where this comes from
---------------------
feat_confirm.py temp showed, at the submitted 1200-tree / 3-seed setting, that
dropping the `prev_day` (18 cols) and `mem_long` (6 cols) groups TOGETHER beats
the 98-column view: A -0.0270 (CI -0.0431..-0.0097, 5/5 folds) and B -0.0117
(CI -0.0264..+0.0033, 4/5 folds).  Separately they are worth -0.0181 / -0.0053
and -0.0068 / +0.0005, so the combination is not the sum of its parts and had
to be measured directly.

feat_new.py temp screened the five new derived blocks on the 98-column view and
every one of them pointed the same way (event A -0.0070 B -0.0196, duty A
-0.0092 B -0.0138, ALL_NEW A -0.0018 B -0.0229).

Those two results were measured on different bases, and `event` / `duty` are
actuator-history blocks while `prev_day` is largely previous-day actuator and
climate means.  Removing prev_day and then adding actuator history back can put
some of the same information back, so the combination cannot be predicted by
adding the two deltas -- it is measured here directly.

Usage
-----
  python feat_temp74.py screen            cheap surrogate, base = 74 cols,
                                          add each new block and ALL_NEW
  python feat_temp74.py confirm b1,b2     submitted model, base = the ORIGINAL
                                          98-col view, so every delta printed
                                          is against the shipped configuration
"""
import json
import sys
import time

import numpy as np

import env  # noqa: F401  MUST be first
from harness import load, score, views
import feat_lib as L
import feat_new as N

BLOCK_ORDER = ["event", "duty", "rep24", "hinge", "dew"]


def base74(cols):
    g = L.grouping(cols)
    drop = set(g["prev_day"]) | set(g["mem_long"])
    return [c for c in cols if c not in drop]


def main(mode, picks=None):
    panel, lab_t, lab_e, blocks, new = N.attach()
    v = views(panel)
    cols98 = v["temp"]
    cols74 = base74(cols98)
    tgt = "sub_temp"

    if mode == "screen":
        model, seeds, ref, tag = L.TEMP_FAST, (7,), cols74, "temp74_screen"
        cands = [("+%s" % b, cols74 + list(blocks[b])) for b in BLOCK_ORDER]
        cands.append(("+ALL_NEW", cols74 + list(new)))
        cands.append(("98f original", cols98))
    else:
        model, seeds, ref, tag = L.TEMP_MODEL, (7, 101, 2024), cols98, \
            "temp74_confirm"
        cands = [("74f base", cols74)]
        for p in (picks or []):
            # a pick may join several blocks with "+"; blocks that share
            # information (event and duty are both actuator history) must be
            # measured jointly, never inferred by adding their deltas.
            if p == "ALL_NEW":
                add = list(new)
            else:
                add = sum([list(blocks[b]) for b in p.split("+")], [])
            cands.append(("74f +%s" % p, cols74 + add))

    print("mode=%s  base=%d cols  seeds=%s" % (mode, len(ref), seeds))
    print("74f base drops prev_day+mem_long: %d -> %d cols"
          % (len(cols98), len(cols74)))

    base = {}
    for kind in ("A", "B"):
        (r, sd, per), oof = score(lab_t, tgt, ref, model, kind=kind,
                                  seeds=seeds, return_oof=True)
        base[kind] = dict(rmse=r, std=sd, per=per, oof=oof)
        print("BASE %s %.4f (foldstd %.4f)  folds %s"
              % (kind, r, sd, " ".join("%.4f" % x for x in per)), flush=True)

    rows = []
    for name, cs in cands:
        rec = dict(name=name, n=len(cs))
        t0 = time.time()
        for kind in ("A", "B"):
            (r, sd, per), oof = score(lab_t, tgt, cs, model, kind=kind,
                                      seeds=seeds, return_oof=True)
            d, lo, hi, _ = L.paired_block_boot(lab_t, tgt, base[kind]["oof"],
                                               oof,
                                               n_boot=600 if mode == "screen"
                                               else 2000)
            pf = [x - y for x, y in zip(per, base[kind]["per"])]
            rec["rmse_" + kind], rec["std_" + kind] = r, sd
            rec["d_" + kind] = r - base[kind]["rmse"]
            rec["ci_" + kind] = [lo, hi]
            rec["perfold_" + kind] = pf
            rec["nneg_" + kind] = int(sum(1 for x in pf if x < 0))
        rec["sec"] = time.time() - t0
        rows.append(rec)
        print("%-16s n=%3d | A %.4f (%+.4f CI %+.4f..%+.4f, %d/5) "
              "| B %.4f (%+.4f CI %+.4f..%+.4f, %d/5)"
              % (name, rec["n"], rec["rmse_A"], rec["d_A"], rec["ci_A"][0],
                 rec["ci_A"][1], rec["nneg_A"], rec["rmse_B"], rec["d_B"],
                 rec["ci_B"][0], rec["ci_B"][1], rec["nneg_B"]), flush=True)

    path = env.LOCAL + "/%s.json" % tag
    with open(path, "w") as fh:
        json.dump(dict(mode=mode, n_base=len(ref),
                       base={k: dict(rmse=base[k]["rmse"], std=base[k]["std"],
                                     per=base[k]["per"]) for k in base},
                       cands=rows), fh, indent=1, default=float)
    print("\nwrote", path)
    print("\n--- ranked by mean(dA, dB), best first ---")
    for r in sorted(rows, key=lambda r: (r["d_A"] + r["d_B"]) / 2):
        both = "BOTH" if (r["d_A"] < 0) == (r["d_B"] < 0) else "SPLIT"
        print("  %-16s dA %+.4f  dB %+.4f  mean %+.4f  %s"
              % (r["name"], r["d_A"], r["d_B"], (r["d_A"] + r["d_B"]) / 2, both))


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0] if a else "screen",
         a[1].split(",") if len(a) > 1 else None)
