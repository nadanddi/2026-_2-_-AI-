# -*- coding: utf-8 -*-
"""Blind analyst A's EC idea: shallow trees (num_leaves 7), no outdoor weather.
Our round-3 EC already drops outdoor weather; its LightGBM tweedie member has
31 leaves.  The label-derived "twin" feature of analyst A is NOT used (team
decision: no labels of other days as test-row features).

Variants (round-3 pipeline otherwise; ET 14f+fp .60, LGB .30, MLP .10,
causal shrink 0.5, clip):
  E1  LGB tweedie member with num_leaves 7 (min_child_samples, trees kept)
  E2  0.8 * round-3 + 0.2 * shallow LGB (L2, num_leaves 7, 14f + fp)

Pre-set rule: adopt only if the variant beats round-3 on geometry A (mean of
per-fold RMSE; folds overlap) AND DIAG10 (pooled) for BOTH seeds (7, 8), and
the DIAG10 block-bootstrap CI excludes 0.

Run:  cd research && PYTHONPATH="" <python> -u ec_shallow_v1.py
"""
import env  # noqa: F401
import numpy as np
import lightgbm as lgb

import ec_v6
from common import split_mask, rmse, USABLE, OUT_COLS
from harness import load, folds
import features_v4 as F4
from anal_q1_errors import diag_folds
from make_submission_v3 import causal_shrink
from screen_v6 import boot

SEEDS = (7, 8)


def lgb7(seed, objective):
    p = dict(ec_v6.LGBP)
    p["num_leaves"] = 7
    extra = dict(tweedie_variance_power=1.5) if objective == "tweedie" else {}
    return lgb.LGBMRegressor(random_state=seed, objective=objective, **extra, **ec_v6.DET, **p)


def members(tr, va, c_et, c_rest, seed):
    ec_v6.SEED = seed
    y = tr.sub_ec.values
    return dict(et=ec_v6.et().fit(tr[c_et], y).predict(va[c_et]),
                ltw=ec_v6.ltw().fit(tr[c_rest], y).predict(va[c_rest]),
                mlp=ec_v6.mlp().fit(tr[c_rest], y).predict(va[c_rest]),
                ltw7=lgb7(seed, "tweedie").fit(tr[c_rest], y).predict(va[c_rest]),
                l2s7=lgb7(seed, "regression").fit(tr[c_et], y).predict(va[c_et]))


def finish(p, va):
    return np.clip(causal_shrink(p, va, 0.5), 0.062, 3.46)


def main():
    _, _, lab0 = load()
    fp = F4.fp_features()
    lab = lab0.merge(fp, on="row_id", how="left").reset_index(drop=True)
    f14 = [c for c in (list(USABLE) + ["day", "hr_sin", "hr_cos", "midnight"]) if c not in OUT_COLS]
    c_et = f14 + F4.names(fp)
    y = lab.sub_ec.values
    verdict = {"E1": True, "E2": True}
    for vset, fds in (("A", folds("A")), ("DIAG10", diag_folds(lab))):
        for sd in SEEDS:
            pooled = {k: np.full(len(lab), np.nan) for k in ("base", "E1", "E2")}
            per = []
            for fd in fds:
                trm, vam = split_mask(lab, fd)
                tr, va = lab[trm], lab[vam].reset_index(drop=True)
                M = members(tr, va, c_et, f14, sd)
                base = finish(0.6 * M["et"] + 0.3 * M["ltw"] + 0.1 * M["mlp"], va)
                e1 = finish(0.6 * M["et"] + 0.3 * M["ltw7"] + 0.1 * M["mlp"], va)
                e2 = finish(0.8 * (0.6 * M["et"] + 0.3 * M["ltw"] + 0.1 * M["mlp"]) + 0.2 * M["l2s7"], va)
                idx = np.where(vam)[0]
                yy = y[idx]
                per.append((rmse(base, yy), rmse(e1, yy), rmse(e2, yy)))
                pooled["base"][idx], pooled["E1"][idx], pooled["E2"][idx] = base, e1, e2
            P = np.array(per)
            line = "%-6s seed %d | base %.4f" % (vset, sd, P[:, 0].mean() if vset == "A" else rmse(pooled["base"][~np.isnan(pooled["base"])], y[~np.isnan(pooled["base"])]))
            for j, k in ((1, "E1"), (2, "E2")):
                if vset == "A":
                    d = P[:, j].mean() / P[:, 0].mean() - 1
                    line += " | %s %.4f (%+.2f%%, better %d/%d)" % (k, P[:, j].mean(), 100 * d, int((P[:, j] < P[:, 0]).sum()), len(P))
                    ok = d < 0
                else:
                    g = ~np.isnan(pooled["base"])
                    pr, lo, hi, pw = boot(lab[g].reset_index(drop=True), "sub_ec", pooled["base"][g], pooled[k][g])
                    d = rmse(pooled[k][g], y[g]) / rmse(pooled["base"][g], y[g]) - 1
                    line += " | %s %.4f (%+.2f%%) [%+.4f, %+.4f]" % (k, rmse(pooled[k][g], y[g]), 100 * d, lo, hi)
                    ok = d < 0 and hi < 0
                verdict[k] = verdict[k] and ok
            print(line, flush=True)
    print("\nPRE-SET RULE VERDICT:", {k: ("ADOPT" if v else "REJECT") for k, v in verdict.items()})


if __name__ == "__main__":
    main()
