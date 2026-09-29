# -*- coding: utf-8 -*-
"""EC: up-weight training rows of sealed days (catalog 6.52: sealed days carry
most of the EC error; 15% of training days).  The weight is a FIXED number,
not derived from any test statistic.  Sealed = full-day definition of 6.11,
a property of TRAINING days only (weights never touch test inputs).

Round-3 raw blend (ET 0.6 / tweedie LGB 0.3 / MLP 0.1, ExtraTrees view with
`day` as in the candidate experiments) + shrink 0.5 + clip.
Arms (fixed): SW2 = sealed training rows weight 2, SW3 = weight 3;
reference SW1 = unweighted.  Validators A, B (fold means), DIAG10 (pooled;
sealed / other days shown).  Seeds 7/8.
Rule (2026-09-27): better than SW1 in all 6 cells, DIAG10 p_worse < 0.025/2.
(If one passes, it must be re-checked together with the TabPFN member.)

Run:  cd research && PYTHONPATH="" <python> -u ec_sealedw_v1.py
"""
import env  # noqa: F401
import numpy as np

import common
import ec_v6
from common import split_mask, rmse, USABLE, OUT_COLS, TARGET_FARMS
from harness import load, folds
import features_v4 as F4
from anal_q1_errors import diag_folds
from make_submission_v3 import causal_shrink
from screen_v6 import boot
from anal_ec_regime import sealed_days

ARMS = {"SW1": 1.0, "SW2": 2.0, "SW3": 3.0}


def fit_raw(tr, va, c_et, f14, w):
    yt = tr.sub_ec.values
    e = ec_v6.et().fit(tr[c_et], yt, extratreesregressor__sample_weight=w).predict(va[c_et])
    l = ec_v6.ltw().fit(tr[f14], yt, sample_weight=w).predict(va[f14])
    try:
        m = ec_v6.mlp().fit(tr[f14], yt, mlpregressor__sample_weight=w).predict(va[f14])
    except TypeError:
        m = ec_v6.mlp().fit(tr[f14], yt).predict(va[f14])
    return 0.6 * e + 0.3 * l + 0.1 * m


def main():
    tX, _, _ = common.load_raw()
    _, _, lab0 = load()
    fp = F4.fp_features()
    lab = lab0.merge(fp, on="row_id", how="left").reset_index(drop=True)
    f14 = [c for c in (list(USABLE) + ["day", "hr_sin", "hr_cos", "midnight"]) if c not in OUT_COLS]
    c_et = f14 + F4.names(fp)
    y = lab.sub_ec.values
    s_tr = sealed_days(tX[tX.farm.isin(TARGET_FARMS)])
    is_s = np.array([bool(s_tr.get(k, False)) for k in zip(lab.farm, lab.day)])
    ok = {"SW2": True, "SW3": True}
    for vset, fds in (("A", folds("A")), ("B", folds("B")), ("DIAG10", diag_folds(lab))):
        for sd in (7, 8):
            ec_v6.SEED = sd
            per, pooled = [], {}
            for fd in fds:
                trm, vam = split_mask(lab, fd)
                tr, va = lab[trm], lab[vam].reset_index(drop=True)
                idx = np.where(vam)[0]
                row = {}
                for a, wv in ARMS.items():
                    w = np.where(is_s[trm], wv, 1.0)
                    q = np.clip(causal_shrink(fit_raw(tr, va, c_et, f14, w), va, 0.5), 0.062, 3.46)
                    pooled.setdefault(a, np.full(len(lab), np.nan))[idx] = q
                    row[a] = rmse(q, y[idx])
                per.append(row)
            if vset in ("A", "B"):
                ref = np.mean([r["SW1"] for r in per])
                txt = []
                for a in ok:
                    s = np.mean([r[a] for r in per])
                    ok[a] = ok[a] and s < ref
                    txt.append("%s %.4f (%+.2f%%)" % (a, s, 100 * (s / ref - 1)))
                print("%-6s seed %d | SW1 %.4f | %s" % (vset, sd, ref, " | ".join(txt)), flush=True)
            else:
                g = ~np.isnan(pooled["SW1"])
                ref = rmse(pooled["SW1"][g], y[g])
                txt = []
                for a in ok:
                    pw = boot(lab[g].reset_index(drop=True), "sub_ec", pooled["SW1"][g], pooled[a][g])[3]
                    s = rmse(pooled[a][g], y[g])
                    ok[a] = ok[a] and s < ref and pw < 0.0125
                    txt.append("%s %.4f (%+.2f%%, p_worse %.4f; sealed %.4f vs %.4f; other %.4f vs %.4f)"
                               % (a, s, 100 * (s / ref - 1), pw,
                                  rmse(pooled[a][g & is_s], y[g & is_s]), rmse(pooled["SW1"][g & is_s], y[g & is_s]),
                                  rmse(pooled[a][g & ~is_s], y[g & ~is_s]), rmse(pooled["SW1"][g & ~is_s], y[g & ~is_s])))
                print("DIAG10 seed %d | SW1 %.4f | %s" % (sd, ref, " | ".join(txt)), flush=True)
    for a, v in ok.items():
        print("%s PRE-SET RULE VERDICT: %s" % (a, "ADOPT" if v else "REJECT"), flush=True)


if __name__ == "__main__":
    main()
