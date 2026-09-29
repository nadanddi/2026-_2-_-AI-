# -*- coding: utf-8 -*-
"""Idea from a blind 'student answer': train on perturbed / partly masked
inputs so the model leans less on exact sensor values.  Motivation here: the
training days carry injected indoor-sensor noise while the test inputs are
clean (catalog), so robustness to input perturbation may transfer better.

Temperature only (MASK world, round-5 weights, members as temp_mask_v1.py).
Inside every TRAINING fold the rows are duplicated; the copy is perturbed,
original and copy keep the row's weight.  Validation rows are never touched.

Arms (fixed before running):
  N05 / N10 / N20 : copy + Gaussian noise, sd = 0.05 / 0.10 / 0.20 x the
                    feature's training-fold std, on every model feature
  M10             : copy with 10% of feature cells set to NaN (at random)
Final blend as the current candidate: 0.8*base + 0.2*Codex (Codex OOF 726
from temp_mask_v1_oof.npz); reference = the same with the un-augmented base
(recomputed here with the same code, so both sides share the pipeline).

Pre-set rule per arm: improves DIAG10, EXT10 and EXT12 for BOTH seeds 7/101,
DIAG10 CI excluding 0, and mean DIAG10 gain >= 1% (below 1% = cannot tell).
Four arms are tested, so a pass by one arm alone at the margin is reported
as such (multiple comparisons).

Run:  cd research && PYTHONPATH="" <python> -u temp_augment_v1.py
"""
import numpy as np
import pandas as pd

import env  # noqa: F401
import common
import harness
import cold_v5
from common import rmse, TARGET_FARMS
from screen_v6 import temp_members, collect, boot
from anal_q1_errors import diag_folds
import train_flags_v6 as TF
from temp_mask_v1 import masked_loader, build_world, ORIG

ARMS = {"N05": ("noise", 0.05), "N10": ("noise", 0.10), "N20": ("noise", 0.20), "M10": ("mask", 0.10)}
SEEDS = (7, 101)


def augment(tr, cols, w, kind, level, seed):
    rng = np.random.default_rng(seed)
    cp = tr.copy()
    X = cp[cols].to_numpy(dtype=float)
    if kind == "noise":
        sd = np.nanstd(X, axis=0)
        X = X + rng.normal(0.0, 1.0, X.shape) * (level * sd)[None, :]
    else:
        X[rng.random(X.shape) < level] = np.nan
    cp[cols] = X
    return pd.concat([tr, cp], ignore_index=True), np.concatenate([w, w])


def main():
    common.load_raw = masked_loader
    try:
        lab, ct, phc = build_world()
    finally:
        common.load_raw = ORIG
        harness._CACHE.clear()
    z = np.load(env.LOCAL + "/temp_mask_v1_oof.npz", allow_pickle=True)
    assert (lab.row_id.values == z["row_id"]).all()
    w = TF.row_weights(lab, 0.2, w_noisy=0.2)
    y = lab.sub_temp.values
    cols = list(dict.fromkeys(ct + phc))
    dmin = lab.groupby(["farm", "day"]).ph_in_temp_3.min()
    sets = [("DIAG10", diag_folds(lab))]
    for th in (10, 12):
        cd = dmin[dmin < float(th)]
        sets.append(("EXT%d" % th, [{f: set(int(d) for (ff, d) in cd.index if ff == f) for f in TARGET_FARMS}]))

    def blend(M):
        return 0.65 * M["res"] + 0.25 * M["ridge"] + 0.10 * M["nys"]

    res = {a: [] for a in ARMS}
    for s, fds in sets:
        cx = z["%s__CODEX__726" % s]
        for sd in SEEDS:
            cold_v5.SEED = sd
            base = blend(collect(lab, fds, lambda tr, va, m: temp_members(tr, va, ct, phc, w[m])))
            g = ~np.isnan(base)
            ref = 0.8 * base + 0.2 * cx
            for a, (kind, lv) in ARMS.items():
                def fn(tr, va, m, kind=kind, lv=lv):
                    ta, wa = augment(tr, cols, w[m], kind, lv, sd)
                    return temp_members(ta, va, ct, phc, wa)
                cand = 0.8 * blend(collect(lab, fds, fn)) + 0.2 * cx
                pr, lo, hi, pw = boot(lab[g].reset_index(drop=True), "sub_temp", ref[g], cand[g])
                d = rmse(cand[g], y[g]) / rmse(ref[g], y[g]) - 1
                res[a].append((s, sd, d, hi))
                print("%-6s seed %3d %s | ref %.5f cand %.5f (%+.2f%%) [%+.4f, %+.4f]"
                      % (s, sd, a, rmse(ref[g], y[g]), rmse(cand[g], y[g]), 100 * d, lo, hi), flush=True)
    cold_v5.SEED = 7
    print()
    for a, r in res.items():
        dg = [d for s, _, d, _ in r if s == "DIAG10"]
        ok = (all(d < 0 for _, _, d, _ in r) and all(hi < 0 for s, _, _, hi in r if s == "DIAG10")
              and -np.mean(dg) >= 0.01)
        print("%s PRE-SET RULE VERDICT: %s (DIAG10 mean %+.2f%%)" % (a, "ADOPT" if ok else "REJECT", 100 * np.mean(dg)))


if __name__ == "__main__":
    main()
