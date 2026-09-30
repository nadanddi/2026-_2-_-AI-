# -*- coding: utf-8 -*-
"""H-D: a cold-extrapolation day-bias corrector learned from early-morning inputs.
(lab Claude, 2026-09-30)

HONESTY NOTE: post-hoc hypothesis.  It comes from gap_source_v1 (catalog 6b.20),
where a small LightGBM predicted the EXT12 day residual of G_C2 from hours 0-6
inputs (R2 +0.20, both farms) but not the DIAG10 one.  EXT8 < EXT10 < EXT12 are
nested (same days), so there is no independent cold set inside the data; leakage
is blocked with nested day-chunk exclusion instead (below).  Also: the corrector
learns the bias of a model that never saw cold days, while a submitted model does
see the training cold days - over-correction is the main risk, which the DIAG10
no-harm requirement guards against.

Corrector   LGBMRegressor(n_estimators=200, lr 0.03, 7 leaves, min_child 10,
            subsample 0.8, colsample 0.7), seeds 7 / 101.
            target  = day-mean residual (pred - label) of G_C2 on EXT12 held-out days
            inputs  = gap_source_v1.early_features: hours 0-6 means of sensors and
                      actuators, in-out difference, 23h->0h jump, 06h fingerprint.
Applied     pred_new = pred - c(day)  on hours 7-23 only (inputs up to 06h are
            then all in the past); hours 0-6 unchanged.
Leakage     for a scored day d, the corrector is fitted on EXT12 days whose
            diagnostic fold (5-day chunk round robin, anal_q1_errors.diag_folds)
            differs from d's fold - never on d or its chunk neighbours.
Validators  DIAG10, EXT8* (TabPFN member := base, exact on rows <= 8 C), EXT10,
            EXT12 - G_C2 out-of-fold predictions rebuilt from saved members.

PRE-SET ADOPTION RULE (fixed before running, one candidate -> no Bonferroni):
  ADOPT-AS-CANDIDATE only if RMSE(new) < RMSE(G_C2) for BOTH corrector seeds on
  ALL FOUR validators, AND the DIAG10 paired bootstrap P(worse) < 0.025 for both
  seeds.  Otherwise REJECT.  Nothing is submitted.

Output: logs/hd_cold_corrector_v1.log
Run:  PYTHONPATH="" <python> -u hd_cold_corrector_v1.py   (from 연구실/클로드/code)
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "집", "클로드", "research"))
import env  # noqa: E402,F401
import numpy as np  # noqa: E402
from lightgbm import LGBMRegressor  # noqa: E402
from harness import load  # noqa: E402
from common import rmse  # noqa: E402
from anal_q1_errors import diag_folds  # noqa: E402
from screen_v6 import boot  # noqa: E402
from gap_source_v1 import early_features  # noqa: E402

SEEDS = (7, 101)


def gc2(split, z, z8, g):
    if split == "EXT8":
        base = np.mean([z8["EXT8__MASK__7"], z8["EXT8__MASK__101"]], axis=0)
        cx = np.mean([z8["EXT8__CODEX__726"], z8["EXT8__CODEX__727"]], axis=0)
        pfn = base
    else:
        base = np.mean([z["%s__MASK__7" % split], z["%s__MASK__101" % split]], axis=0)
        cx = np.mean([z["%s__CODEX__726" % split], z["%s__CODEX__727" % split]], axis=0)
        pfn = np.load(env.LOCAL + "/web_tabpfn_v2_temp_%s.npy" % split).mean(0)
    return (0.6 - 0.2 * (1 - g)) * base + (0.2 + 0.4 * (1 - g)) * cx + 0.2 * g * pfn


def main():
    _, lab, _ = load()
    z = np.load(env.LOCAL + "/temp_mask_v1_oof.npz", allow_pickle=True)
    z8 = np.load(env.LOCAL + "/temp_ext8_oof.npz", allow_pickle=True)
    for f in (z, z8):
        assert (f["row_id"] == lab.row_id.values).all()
    y = lab.sub_temp.values
    t = lab.in_temp.values
    g = np.where(np.isnan(t), 1.0, np.clip((t - 8.0) / 2.0, 0, 1))
    X = early_features(lab)
    cols = list(X.columns)
    folds = diag_folds(lab)
    fold_of = {(f, d): i for i, fd in enumerate(folds) for f in fd for d in fd[f]}

    # training table for the corrector: EXT12 day residuals
    p12 = gc2("EXT12", z, z8, g)
    d12 = lab.assign(e=p12 - y)[~np.isnan(p12)]
    T = X.join(d12.groupby(["farm", "day"]).e.mean().rename("bias"), how="inner")
    T_fold = np.array([fold_of[k] for k in T.index])
    print("corrector training days (EXT12): %d, day-bias mean %+.3f SD %.3f" % (len(T), T.bias.mean(), T.bias.std()))

    # one corrector per (seed, excluded fold)
    C = {}
    for sd in SEEDS:
        for k in range(len(folds)):
            m = LGBMRegressor(n_estimators=200, learning_rate=0.03, num_leaves=7, min_child_samples=10,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.7, random_state=sd,
                              verbose=-1, n_jobs=4)
            tr = T_fold != k
            m.fit(T[cols][tr], T.bias[tr])
            C[(sd, k)] = m

    keys = list(zip(lab.farm.values, lab.day.values))
    row_fold = np.array([fold_of.get(k, -1) for k in keys])
    late = lab.hour.values >= 7
    results = []
    for split in ("DIAG10", "EXT8", "EXT10", "EXT12"):
        pred = gc2(split, z, z8, g)
        ok = ~np.isnan(pred)
        for sd in SEEDS:
            corr = np.zeros(len(lab))
            for k in range(len(folds)):
                rows = ok & (row_fold == k)
                if not rows.any():
                    continue
                days = sorted(set(k2 for k2, r in zip(keys, rows) if r))
                Xd = X.reindex(days)
                cmap = dict(zip(days, C[(sd, k)].predict(Xd[cols])))
                idx = np.where(rows)[0]
                corr[idx] = [cmap[keys[i]] for i in idx]
            new = np.where(late, pred - corr, pred)
            r0, r1 = rmse(pred[ok], y[ok]), rmse(new[ok], y[ok])
            _, lo, hi, pw = boot(lab[ok].reset_index(drop=True), "sub_temp", pred[ok], new[ok])
            cold = ok & (t <= 10)
            name = split + ("*" if split == "EXT8" else "")
            print("%-7s seed %3d | G_C2 %.5f -> %.5f (%+.2f%%) CI [%+.4f, %+.4f] P(worse)=%.3f | "
                  "mean correction on hours 7-23 %+.3f | rows in_temp<=10: %.3f -> %.3f"
                  % (name, sd, r0, r1, 100 * (r1 / r0 - 1), lo, hi, pw, corr[ok & late].mean(),
                     rmse(pred[cold], y[cold]), rmse(new[cold], y[cold])), flush=True)
            results.append((split, sd, r1 < r0, pw))
    all_better = all(r[2] for r in results)
    diag_ok = all(r[3] < 0.025 for r in results if r[0] == "DIAG10")
    print("\nall 8 cells improve: %s | DIAG10 P(worse) < 0.025 both seeds: %s" % (all_better, diag_ok))
    print("H-D verdict: %s" % ("ADOPT-AS-CANDIDATE" if all_better and diag_ok else "REJECT"))


if __name__ == "__main__":
    main()
