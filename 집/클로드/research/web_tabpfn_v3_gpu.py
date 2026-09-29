# -*- coding: utf-8 -*-
"""TabPFN step 3 (EC): how much TabPFN, and how large a context?

v2 showed the bagged TabPFN member ALONE beats the round-3 EC model by ~13%
on DIAG10 (0.181 vs 0.209), yet the pre-set 0.2 blend keeps only a small part
of that.  v2 also used 2,000 context rows out of the training fold.  With the
GPU a larger context is affordable.

Grid (fixed): context 2000 / 8000 rows  x  TabPFN weight 0.2/0.4/0.6/0.8/1.0
  cand = clip(causal_shrink((1-w)*round3 + w*bag, 0.5), 0.062, 3.46)
  bag  = mean of 4 context samples (seeds 1-4), n_estimators 4, GPU float32.

Selection is separated from confirmation (picking 1 of 10 cells on the same
data would reward luck):
  SELECT  on geometry A, round-3 seed 7            -> best cell by mean fold RMSE
  CONFIRM on geometry B and DIAG10, round-3 seed 8, bag seeds 5-8 (new samples)
Pre-set rule: the selected cell must beat (i) round-3 and (ii) the v2 cell
(ctx 2000, w 0.2) on B (fold mean) and DIAG10, DIAG10 CI vs v2 cell excluding
0, and gain vs the v2 cell >= 1% on DIAG10 (below 1% = cannot tell).

Run:  cd research && PYTHONPATH="" <python> -u web_tabpfn_v3_gpu.py
"""
import env  # noqa: F401
import env_extra_gpu  # noqa: F401
import numpy as np
import torch
from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion

import ec_v6
from common import split_mask, rmse, USABLE, OUT_COLS
from harness import load, folds
import features_v4 as F4
from anal_q1_errors import diag_folds
from make_submission_v3 import causal_shrink
from screen_v6 import boot

assert torch.cuda.is_available()
CTX = (2000, 8000)
WS = (0.2, 0.4, 0.6, 0.8, 1.0)


def fit_predict(Xtr, ytr, Xva, n_ctx, seed):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(Xtr), size=min(n_ctx, len(Xtr)), replace=False)
    m = TabPFNRegressor.create_default_for_version(ModelVersion.V2, device="cuda", n_estimators=4,
                                                   random_state=seed, ignore_pretraining_limits=True,
                                                   inference_precision=torch.float32)
    m.fit(Xtr[idx], ytr[idx])
    return m.predict(Xva)


def run(lab, fds, c_et, f14, X, y, r3_seed, bag):
    """Kept per fold: folds of geometry A/B overlap, so nothing is pooled here."""
    ec_v6.SEED = r3_seed
    out = []
    for fd in fds:
        trm, vam = split_mask(lab, fd)
        tr, va = lab[trm], lab[vam].reset_index(drop=True)
        yt = tr.sub_ec.values
        raw = (0.6 * ec_v6.et().fit(tr[c_et], yt).predict(va[c_et])
               + 0.3 * ec_v6.ltw().fit(tr[f14], yt).predict(va[f14])
               + 0.1 * ec_v6.mlp().fit(tr[f14], yt).predict(va[f14]))
        mem = {n: np.mean([fit_predict(X[trm], y[trm], X[vam], n, s) for s in bag], axis=0) for n in CTX}
        out.append(dict(idx=np.where(vam)[0], va=va, raw=raw, mem=mem))
        print("   fold done (%d rows)" % len(va), flush=True)
    return out


def fin(f, w, n=2000):
    p = f["raw"] if w == 0 else (1 - w) * f["raw"] + w * f["mem"][n]
    return np.clip(causal_shrink(p, f["va"], 0.5), 0.062, 3.46)


def fold_mean(F, y, w, n=2000):
    return float(np.mean([rmse(fin(f, w, n), y[f["idx"]]) for f in F]))


def pooled(lab, F, w, n=2000):
    o = np.full(len(lab), np.nan)
    for f in F:
        o[f["idx"]] = fin(f, w, n)
    return o


def main():
    _, _, lab0 = load()
    fp = F4.fp_features()
    lab = lab0.merge(fp, on="row_id", how="left").reset_index(drop=True)
    f14 = [c for c in (list(USABLE) + ["day", "hr_sin", "hr_cos", "midnight"]) if c not in OUT_COLS]
    c_et = f14 + F4.names(fp)
    y = lab.sub_ec.values
    X = lab[c_et].values.astype(np.float32)
    print("EC rows %d" % len(lab), flush=True)

    fA = run(lab, folds("A"), c_et, f14, X, y, 7, (1, 2, 3, 4))
    base = fold_mean(fA, y, 0)
    best, cells = None, {}
    for n in CTX:
        for w in WS:
            s = fold_mean(fA, y, w, n)
            cells[(n, w)] = s
            print("SELECT A | ctx %5d w %.1f | %.4f (round-3 %.4f, %+.2f%%)" % (n, w, s, base, 100 * (s / base - 1)), flush=True)
            if best is None or s < cells[best]:
                best = (n, w)
    print("SELECTED ctx %d w %.1f" % best, flush=True)

    ok = True
    for vset, fds in (("B", folds("B")), ("DIAG10", diag_folds(lab))):
        F = run(lab, fds, c_et, f14, X, y, 8, (5, 6, 7, 8))
        if vset == "B":
            a, b, c = fold_mean(F, y, 0), fold_mean(F, y, 0.2, 2000), fold_mean(F, y, best[1], best[0])
            print("CONFIRM B      | round-3 %.4f  v2 %.4f  selected %.4f (vs v2 %+.2f%%)" % (a, b, c, 100 * (c / b - 1)), flush=True)
            ok = ok and c < a and c < b
        else:
            p_r3, p_v2, p_sel = pooled(lab, F, 0), pooled(lab, F, 0.2, 2000), pooled(lab, F, best[1], best[0])
            g = ~np.isnan(p_r3)
            a, b, c = rmse(p_r3[g], y[g]), rmse(p_v2[g], y[g]), rmse(p_sel[g], y[g])
            pr, lo, hi, pw = boot(lab[g].reset_index(drop=True), "sub_ec", p_v2[g], p_sel[g])
            print("CONFIRM DIAG10 | round-3 %.4f  v2 %.4f  selected %.4f (vs v2 %+.2f%%) [%+.4f, %+.4f]"
                  % (a, b, c, 100 * (c / b - 1), lo, hi), flush=True)
            for n in CTX:
                for w in WS:
                    q = pooled(lab, F, w, n)
                    print("   (all cells, info) ctx %5d w %.1f | %.4f" % (n, w, rmse(q[g], y[g])), flush=True)
            ok = ok and c < a and c < b and hi < 0 and (1 - c / b) >= 0.01
    print("EC v3 PRE-SET RULE VERDICT:", "ADOPT" if ok else "REJECT", flush=True)


if __name__ == "__main__":
    main()
