# -*- coding: utf-8 -*-
"""TX1: temperature analogue of the EC high-day experiments.  Do days with a large UNEXPLAINED day offset (substrate
temperature level off from the model for the whole day, 6b.3 / 6b.32) hurt training?  (2026-10-09 집 클로드, user:
"진행해봐").  Fixed before running.

Model (proxy of the submitted W40G-S, WITHOUT TabPFN and without the season swap):
  BASE  = .65 (LinearRegression physics + LGB residual) + .25 ridge + .10 Nystroem   (temp_members, seeds 7 / 101)
  CODEX = Ridge physics + LGB residual (codex_fit_predict, seed 726; deterministic)
  P     = .5 BASE + .5 CODEX        MASK world, current training weights (contaminated rows and noisy days 0.2).
Step A: CUR (current weights) out-of-fold on DIAG10 -> OFFSET = days with |day-mean residual (seed mean)| >= 0.7 C
        (threshold of 6b.32, set before this run).
Step B configurations (same folds and seeds):
  CUR   current weights;   REM   OFFSET days removed from training;   W02   OFFSET days at weight min(w, 0.2)
Sets: DIAG10, EXT10, EXT12 (house temperature sets), MASK world, house split (+-1 day).
PRIMARY rows = ALL rows (lesson of the EC runs).  Reported: NORMAL rows (day not in OFFSET, contaminated rows +-3 h
removed as 4.9), OFFSET rows, pass-2 rows.
Rule (temperature rule, k = 2, alpha .0125), X in {REM, W02} vs CUR on ALL rows:
  PASS iff every BASE seed x {DIAG10, EXT10, EXT12} better (6/6) AND DIAG10 seed-mean farm x 5-day block bootstrap
  share(X not better) < .0125 (20,000 draws).
Run:  cd 집/클로드/research && PYTHONPATH="" py -3.12 -u tx1_temp_offset_days_v1.py
"""
import os, sys, json
import env  # noqa: F401
import numpy as np, pandas as pd
import common, harness, cold_v5
from common import split_mask, TARGET_FARMS
from screen_v6 import temp_members, collect
from anal_q1_errors import diag_folds
import train_flags_v6 as TF
import temp_mask_v1 as TM

SEEDS_T = (7, 101); SEED_C = 726
r = lambda e: float(np.sqrt(np.nanmean(np.square(e))))
OUT = os.path.join(env.LOCAL, "tx1")


def boot_share(lab, a, b, n=20000, seed=0):
    blk = (lab.farm.astype(str) + "_" + (lab.day // 5).astype(str)).values
    keys, inv = np.unique(blk, return_inverse=True)
    sa = np.bincount(inv, weights=a, minlength=len(keys)); sb = np.bincount(inv, weights=b, minlength=len(keys))
    rng = np.random.default_rng(seed); worse = 0
    for _ in range(n // 1000):
        ix = rng.integers(0, len(keys), size=(1000, len(keys)))
        worse += int((sb[ix].sum(1) >= sa[ix].sum(1)).sum())
    return worse / n


def main():
    os.makedirs(OUT, exist_ok=True)
    labF, ct, phc = TM.build_world()
    common.load_raw = TM.masked_loader
    try:
        labM, _, _ = TM.build_world()
    finally:
        common.load_raw = TM.ORIG; harness._CACHE.clear()
    tX, ty, sX = TM.ORIG()
    CF = TM.build_features(tX, sX).drop(columns=["farm", "day", "hour", "t"]).set_index("row_id")
    for c in TM.FEATURE_COLUMNS:
        if c not in labM.columns:
            labM[c] = CF.loc[labM.row_id, c].values
    w0 = TF.row_weights(labM, 0.2, w_noisy=0.2)
    y = labM.sub_temp.values
    dmin = labF.groupby(["farm", "day"]).ph_in_temp_3.min()
    sets = [("DIAG10", diag_folds(labM))]
    for th in (10.0, 12.0):
        cd = dmin[dmin < th]
        sets.append(("EXT%d" % th, [{f: set(int(d) for (ff, d) in cd.index if ff == f) for f in TARGET_FARMS}]))
    keys = np.array(list(zip(labM.farm.values, labM.day.values)), dtype=object)

    def run(cfg, w, drop):
        res = {}
        keep_all = np.array([(f, d) not in drop for f, d in zip(labM.farm.values, labM.day.values)])
        for s, fds in sets:
            for sd in SEEDS_T:
                cold_v5.SEED = sd
                def fn(tr, va, m):
                    k = keep_all[m]
                    return temp_members(tr[k], va, ct, phc, w[m][k])
                M = collect(labM, fds, fn)
                base = .65 * M["res"] + .25 * M["ridge"] + .10 * M["nys"]
                o = np.full(len(labM), np.nan)
                for fd in fds:
                    trm, vam = split_mask(labM, fd)
                    k = trm & keep_all
                    o[vam] = TM.codex_fit_predict(labM[k], labM[vam], w[k], SEED_C)
                res[(s, sd)] = .5 * base + .5 * o
            print("  %s %s done" % (cfg, s), flush=True)
        cold_v5.SEED = 7
        np.savez(os.path.join(OUT, "%s.npz" % cfg), row_id=labM.row_id.values, **{"%s__%d" % k: v for k, v in res.items()})
        return res

    R = {"CUR": run("CUR", w0, set())}
    # step A: OFFSET days from CUR DIAG10
    pm = np.mean([R["CUR"][("DIAG10", sd)] for sd in SEEDS_T], axis=0)
    D = pd.DataFrame({"farm": labM.farm, "day": labM.day, "e": pm - y}).groupby(["farm", "day"]).e.mean()
    offset = {(f, int(d)) for (f, d), v in D.items() if abs(v) >= .7}
    json.dump(sorted(map(list, offset)), open(os.path.join(OUT, "offset_days.json"), "w"))
    print("OFFSET days: %d of %d (pass-2 %d); F13 %d F47 %d" % (len(offset), len(D), sum(d >= 179 for _, d in offset),
          sum(f == "F13" for f, _ in offset), sum(f == "F47" for f, _ in offset)), flush=True)
    R["REM"] = run("REM", w0, offset)
    w2 = w0.copy(); w2[np.array([(f, d) in offset for f, d in zip(labM.farm, labM.day)])] = np.minimum(w2[np.array([(f, d) in offset for f, d in zip(labM.farm, labM.day)])], .2)
    R["W02"] = run("W02", w2, set())

    # summary
    ff = TF.restored_flags(); bad = ff[ff.flag]
    near = np.zeros(len(labM), bool)
    for farm, g in bad.groupby("farm"):
        m = labM.farm.values == farm
        near[m] |= (np.abs(labM.t.values[m][:, None] - g.t.values[None, :]) <= 3).any(1)
    is_off = np.array([(f, d) in offset for f, d in zip(labM.farm, labM.day)])
    views = {"ALL": np.ones(len(labM), bool), "NORMAL": ~is_off & ~near, "OFFSET": is_off, "PASS2": labM.day.values >= 179}
    verdict = {}
    for view, vm in views.items():
        print("\n== %s rows" % view)
        for s, _ in sets:
            for cfg in ("REM", "W02"):
                cells, line = [], ""
                for sd in SEEDS_T:
                    a, b = R["CUR"][(s, sd)], R[cfg][(s, sd)]
                    g = vm & ~np.isnan(a)
                    ra, rb = r(a[g] - y[g]), r(b[g] - y[g]); cells.append(rb < ra)
                    line += " seed%d %.4f->%.4f (%+.2f%%)" % (sd, ra, rb, 100 * (rb / ra - 1))
                print("  %-6s %s:%s" % (s, cfg, line))
                if view == "ALL":
                    verdict.setdefault(cfg, []).extend(cells)
    for cfg in ("REM", "W02"):
        a = np.mean([R["CUR"][("DIAG10", sd)] for sd in SEEDS_T], 0); b = np.mean([R[cfg][("DIAG10", sd)] for sd in SEEDS_T], 0)
        share = boot_share(labM, (a - y) ** 2, (b - y) ** 2)
        ok = len(verdict[cfg]) == 6 and all(verdict[cfg]) and share < .0125
        print("\nVERDICT %s: ALL seed x set better %d/6, DIAG10 share %.4f -> %s" % (cfg, sum(verdict[cfg]), share, "PASS" if ok else "FAIL"))


if __name__ == "__main__":
    main()
