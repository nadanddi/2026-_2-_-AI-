# -*- coding: utf-8 -*-
"""Web research 2 & 3: GPBoost - tree boosting + a random effect per record
day (farm, day).  The day-level offset (56-58% of the temperature error, 83%
of the EC error) is absorbed by the random effect during training, so the
trees learn input-response relations less polluted by source offsets.  Test
days are new groups -> predicted with the fixed (tree) part only.

Members use within-day features only (no cross-day path):
  temperature : Codex resid_reset features (89) , round-5 weights
  EC          : f14 + fingerprint features (the round-3 ExtraTrees view)
GPBoost settings fixed in advance: 300 rounds, lr 0.05, num_leaves 15,
min_data_in_leaf 100, feature_fraction 0.8 (seeded; GPBoost does not allow bagging).

Pre-set rules
  temperature : 0.7*MASK base + 0.2*Codex + 0.1*GPB  vs  0.8*MASK base + 0.2*Codex
                must win DIAG10 and EXT10 for base seeds 7/101 x member seeds 1/2,
                DIAG10 CI excluding 0 in every combination.
  EC          : 0.8*round-3 + 0.2*GPB  vs  round-3, geometry A (mean per fold)
                and DIAG10 for seeds 7/8 (round-3 seed = GPB seed), DIAG10 CI
                excluding 0.

Run:  cd research && PYTHONPATH="" <python> -u web_gpboost_v1.py
"""
import os
import sys

import env  # noqa: F401
import env_extra  # noqa: F401
import numpy as np
import gpboost as gpb

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


def gpb_fit_predict(Xtr, ytr, gtr, Xva, w, seed):
    kw = {} if w is None else {"weights": np.asarray(w, dtype=float)}
    gm = gpb.GPModel(group_data=gtr, likelihood="gaussian", **kw)
    ds = gpb.Dataset(Xtr, ytr)
    params = dict(learning_rate=0.05, num_leaves=15, min_data_in_leaf=100,
                  feature_fraction=0.8, feature_fraction_seed=seed, seed=seed, verbose=-1, num_threads=4)
    bst = gpb.train(params=params, train_set=ds, gp_model=gm, num_boost_round=300)
    pr = bst.predict(data=Xva, group_data_pred=np.full(len(Xva), -1), predict_var=False, pred_latent=True)
    return np.asarray(pr["fixed_effect"])


def day_group(df):
    return (df.farm.map({"F13": 0, "F47": 1}).astype(int) * 1000 + df.day.astype(int)).to_numpy(dtype=np.int64)


def temperature():
    tX, ty, sX = common.load_raw()
    _, lab0, _ = load()
    z = np.load(env.LOCAL + "/temp_mask_v1_oof.npz", allow_pickle=True)
    F = build_features(tX, sX).drop(columns=["farm", "day", "hour", "t"]).set_index("row_id")
    lab = lab0[["row_id", "farm", "day", "hour", "t", "sub_temp"]].join(F, on="row_id")
    assert (lab.row_id.values == z["row_id"]).all()
    w = TF.row_weights(lab, 0.2, w_noisy=0.2)
    y = lab.sub_temp.values
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
                tr, va = lab[trm], lab[vam]
                o[vam] = gpb_fit_predict(tr[FEATURE_COLUMNS].values.astype(float), tr.sub_temp.values, day_group(tr),
                                         va[FEATURE_COLUMNS].values.astype(float), w[trm], ms)
            mem[ms] = o
        cx = z["%s__CODEX__726" % s]
        for bs in (7, 101):
            base = z["%s__MASK__%d" % (s, bs)]
            g = ~np.isnan(base)
            ref = 0.8 * base + 0.2 * cx
            for ms in (1, 2):
                cand = 0.7 * base + 0.2 * cx + 0.1 * mem[ms]
                pr, lo, hi, pw = boot(lab[g].reset_index(drop=True), "sub_temp", ref[g], cand[g])
                d = rmse(cand[g], y[g]) / rmse(ref[g], y[g]) - 1
                print("TEMP %-6s base %3d member %d | GPB alone %.5f | ref %.5f cand %.5f (%+.2f%%) [%+.4f, %+.4f]"
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
    ok = True
    for vset, fds in (("A", folds("A")), ("DIAG10", diag_folds(lab))):
        for sd in (7, 8):
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
                gp = gpb_fit_predict(tr[c_et].values.astype(float), yt, day_group(tr),
                                     va[c_et].values.astype(float), None, sd)
                pb = np.clip(causal_shrink(raw, va, 0.5), 0.062, 3.46)
                pc = np.clip(causal_shrink(0.8 * raw + 0.2 * gp, va, 0.5), 0.062, 3.46)
                idx = np.where(vam)[0]
                pooled["base"][idx], pooled["cand"][idx] = pb, pc
                per.append((rmse(pb, y[idx]), rmse(pc, y[idx]), rmse(gp, y[idx])))
            P = np.array(per)
            if vset == "A":
                d = P[:, 1].mean() / P[:, 0].mean() - 1
                print("EC   A      seed %d | base %.4f cand %.4f (%+.2f%%, better %d/5) | GPB alone %.4f"
                      % (sd, P[:, 0].mean(), P[:, 1].mean(), 100 * d, int((P[:, 1] < P[:, 0]).sum()), P[:, 2].mean()), flush=True)
                ok = ok and d < 0
            else:
                g = ~np.isnan(pooled["base"])
                pr, lo, hi, pw = boot(lab[g].reset_index(drop=True), "sub_ec", pooled["base"][g], pooled["cand"][g])
                d = rmse(pooled["cand"][g], y[g]) / rmse(pooled["base"][g], y[g]) - 1
                print("EC   DIAG10 seed %d | base %.4f cand %.4f (%+.2f%%) [%+.4f, %+.4f]"
                      % (sd, rmse(pooled["base"][g], y[g]), rmse(pooled["cand"][g], y[g]), 100 * d, lo, hi), flush=True)
                ok = ok and d < 0 and hi < 0
    print("EC PRE-SET RULE VERDICT:", "ADOPT" if ok else "REJECT", flush=True)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    if which in ("temp", "both"):
        temperature()
    if which in ("ec", "both"):
        ec()
