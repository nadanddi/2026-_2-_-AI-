# -*- coding: utf-8 -*-
"""Web research 3: TabPFN v2 (pre-trained in-context tabular regressor,
Prior Labs, weights Prior-Labs/TabPFN-v2-reg, Apache-2.0 + attribution; v2 only -
later versions carry a non-commercial licence).  The context is TRAINING rows
only, so no test input enters any fit.

Context: 2,000 training-fold rows sampled with probability proportional to the
round-5 training weights (TabPFN takes no sample weights), seeds 1/2 for both
the sample and the model.  CPU, n_estimators 4.
Members use within-day features only:
  temperature : Codex resid_reset features (89)
  EC          : f14 + fingerprint (38)

Pre-set rules (same as web_gpboost_v1.py)
  temperature : 0.7*MASK base + 0.2*Codex + 0.1*TabPFN vs 0.8*MASK base + 0.2*Codex,
                DIAG10 and EXT10, base seeds 7/101 x member seeds 1/2, DIAG10 CI < 0.
  EC          : 0.8*round-3 + 0.2*TabPFN vs round-3, geometry A (mean per fold)
                and DIAG10, seeds 7/8 (round-3) paired with member seeds 1/2,
                DIAG10 CI < 0.

Run:  cd research && PYTHONPATH="" <python> -u web_tabpfn_v1.py [temp|ec|both]
"""
import os
import sys

import env  # noqa: F401
import env_extra  # noqa: F401
import numpy as np
from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion

import common
import ec_v6
from common import split_mask, rmse, USABLE, OUT_COLS, TARGET_FARMS
from harness import load, folds
import features_v4 as F4
from anal_q1_errors import diag_folds
from make_submission_v3 import causal_shrink
from screen_v6 import boot
import train_flags_v6 as TF

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "analysis", "codex_independent", "2차"))
from resid_reset_features import build_features, FEATURE_COLUMNS  # noqa: E402

N_CTX = 2000


def tabpfn_fit_predict(Xtr, ytr, wtr, Xva, seed):
    rng = np.random.default_rng(seed)
    p = wtr / wtr.sum()
    idx = rng.choice(len(Xtr), size=min(N_CTX, len(Xtr)), replace=False, p=p)
    m = TabPFNRegressor.create_default_for_version(ModelVersion.V2, device="cpu", n_estimators=4,
                                                   random_state=seed, ignore_pretraining_limits=True)
    m.fit(Xtr[idx], ytr[idx])
    return m.predict(Xva)


def temperature():
    tX, ty, sX = common.load_raw()
    _, lab0, _ = load()
    z = np.load(env.LOCAL + "/temp_mask_v1_oof.npz", allow_pickle=True)
    F = build_features(tX, sX).drop(columns=["farm", "day", "hour", "t"]).set_index("row_id")
    lab = lab0[["row_id", "farm", "day", "hour", "t", "sub_temp"]].join(F, on="row_id")
    assert (lab.row_id.values == z["row_id"]).all()
    w = TF.row_weights(lab, 0.2, w_noisy=0.2)
    y = lab.sub_temp.values
    X = lab[FEATURE_COLUMNS].values.astype(np.float32)
    ph = F4.phys_features().set_index("row_id").loc[lab.row_id]
    dmin = ph.groupby([lab.farm.values, lab.day.values]).ph_in_temp_3.min()
    cd = dmin[dmin < 10.0]
    sets = [("DIAG10", diag_folds(lab)),
            ("EXT10", [{f: set(int(d) for (ff, d) in cd.index if ff == f) for f in TARGET_FARMS}])]
    ok = True
    for s, fds in sets:
        mem = {}
        for ms in (1, 2):
            o = np.full(len(lab), np.nan)
            for fd in fds:
                trm, vam = split_mask(lab, fd)
                o[vam] = tabpfn_fit_predict(X[trm], y[trm], w[trm], X[vam], ms)
            mem[ms] = o
            print("  temp %s member %d done" % (s, ms), flush=True)
        np.save(env.LOCAL + "/web_tabpfn_temp_%s.npy" % s, np.vstack([mem[1], mem[2]]))
        cx = z["%s__CODEX__726" % s]
        for bs in (7, 101):
            base = z["%s__MASK__%d" % (s, bs)]
            g = ~np.isnan(base)
            ref = 0.8 * base + 0.2 * cx
            for ms in (1, 2):
                cand = 0.7 * base + 0.2 * cx + 0.1 * mem[ms]
                pr, lo, hi, pw = boot(lab[g].reset_index(drop=True), "sub_temp", ref[g], cand[g])
                d = rmse(cand[g], y[g]) / rmse(ref[g], y[g]) - 1
                print("TEMP %-6s base %3d member %d | TabPFN alone %.5f | ref %.5f cand %.5f (%+.2f%%) [%+.4f, %+.4f]"
                      % (s, bs, ms, rmse(mem[ms][g], y[g]), rmse(ref[g], y[g]), rmse(cand[g], y[g]), 100 * d, lo, hi),
                      flush=True)
                ok = ok and d < 0 and (s != "DIAG10" or hi < 0)
    print("TEMP PRE-SET RULE VERDICT:", "ADOPT" if ok else "REJECT", flush=True)


def ec():
    _, _, lab0 = load()
    fp = F4.fp_features()
    lab = lab0.merge(fp, on="row_id", how="left").reset_index(drop=True)
    f14 = [c for c in (list(USABLE) + ["day", "hr_sin", "hr_cos", "midnight"]) if c not in OUT_COLS]
    c_et = f14 + F4.names(fp)
    y = lab.sub_ec.values
    X = lab[c_et].values.astype(np.float32)
    ones = np.ones(len(lab))
    ok = True
    for vset, fds in (("A", folds("A")), ("DIAG10", diag_folds(lab))):
        for sd, ms in ((7, 1), (8, 2)):
            ec_v6.SEED = sd
            pooled = {k: np.full(len(lab), np.nan) for k in ("base", "cand")}
            per = []
            for fd in fds:
                trm, vam = split_mask(lab, fd)
                tr, va = lab[trm], lab[vam].reset_index(drop=True)
                yt = tr.sub_ec.values
                raw = (0.6 * ec_v6.et().fit(tr[c_et], yt).predict(va[c_et])
                       + 0.3 * ec_v6.ltw().fit(tr[f14], yt).predict(va[f14])
                       + 0.1 * ec_v6.mlp().fit(tr[f14], yt).predict(va[f14]))
                tp = tabpfn_fit_predict(X[trm], y[trm], ones[trm], X[vam], ms)
                pb = np.clip(causal_shrink(raw, va, 0.5), 0.062, 3.46)
                pc = np.clip(causal_shrink(0.8 * raw + 0.2 * tp, va, 0.5), 0.062, 3.46)
                idx = np.where(vam)[0]
                pooled["base"][idx], pooled["cand"][idx] = pb, pc
                per.append((rmse(pb, y[idx]), rmse(pc, y[idx]), rmse(tp, y[idx])))
            P = np.array(per)
            if vset == "A":
                d = P[:, 1].mean() / P[:, 0].mean() - 1
                print("EC   A      seed %d/%d | base %.4f cand %.4f (%+.2f%%, better %d/5) | TabPFN alone %.4f"
                      % (sd, ms, P[:, 0].mean(), P[:, 1].mean(), 100 * d, int((P[:, 1] < P[:, 0]).sum()), P[:, 2].mean()), flush=True)
                ok = ok and d < 0
            else:
                g = ~np.isnan(pooled["base"])
                pr, lo, hi, pw = boot(lab[g].reset_index(drop=True), "sub_ec", pooled["base"][g], pooled["cand"][g])
                d = rmse(pooled["cand"][g], y[g]) / rmse(pooled["base"][g], y[g]) - 1
                print("EC   DIAG10 seed %d/%d | base %.4f cand %.4f (%+.2f%%) [%+.4f, %+.4f] | TabPFN alone %.4f"
                      % (sd, ms, rmse(pooled["base"][g], y[g]), rmse(pooled["cand"][g], y[g]), 100 * d, lo, hi,
                         P[:, 2].mean()), flush=True)
                ok = ok and d < 0 and hi < 0
    print("EC PRE-SET RULE VERDICT:", "ADOPT" if ok else "REJECT", flush=True)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    if which in ("temp", "both"):
        temperature()
    if which in ("ec", "both"):
        ec()
