# -*- coding: utf-8 -*-
"""TabPFN, step 2: context bagging + validators not used in v1.

v1 (web_tabpfn_v1.py) passed its pre-set rules, but a single 2,000-row context
sample made the member seed-sensitive.  v2 averages K=4 context samples
(each its own sample seed and model seed).  Blend weights are NOT re-tuned:
temperature 0.1, EC 0.2, exactly as v1.

Two disjoint bags: seeds {1,2,3,4} and {5,6,7,8}.

Pre-set rules (fixed before running)
  temperature : 0.7*MASK base + 0.2*Codex + 0.1*bag  vs  0.8*MASK base + 0.2*Codex
                validators DIAG10, EXT10, EXT12 (EXT12 unused in v1),
                base seeds 7/101 x both bags: must improve in ALL 12 cases,
                DIAG10 CI excluding 0 in all 4 DIAG10 cases.
  EC          : 0.8*round-3 + 0.2*bag  vs  round-3 (causal_shrink 0.5, clip)
                geometry A, geometry B (unused in v1), DIAG10;
                round-3 seeds 7/8 paired with bags 1/2: must improve in all
                6 cases, DIAG10 CI excluding 0.
  Also reported (not part of the rule): bag vs single-sample member RMSE.

Run:  cd research && PYTHONPATH="" <python> -u web_tabpfn_v2.py [temp|ec|both]
"""
import os
import sys

import env  # noqa: F401
import env_extra  # noqa: F401
import numpy as np

import common
import ec_v6
from common import split_mask, rmse, USABLE, OUT_COLS, TARGET_FARMS
from harness import load, folds
import features_v4 as F4
from anal_q1_errors import diag_folds
from make_submission_v3 import causal_shrink
from screen_v6 import boot
import train_flags_v6 as TF
from web_tabpfn_v1 import tabpfn_fit_predict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "analysis", "codex_independent", "2차"))
from resid_reset_features import build_features, FEATURE_COLUMNS  # noqa: E402

BAGS = {1: (1, 2, 3, 4), 2: (5, 6, 7, 8)}


def bag_predict(Xtr, ytr, wtr, Xva, seeds):
    P = np.vstack([tabpfn_fit_predict(Xtr, ytr, wtr, Xva, s) for s in seeds])
    return P.mean(0), P[0]


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
    sets = [("DIAG10", diag_folds(lab))]
    for th in (10, 12):
        cd = dmin[dmin < float(th)]
        sets.append(("EXT%d" % th, [{f: set(int(d) for (ff, d) in cd.index if ff == f) for f in TARGET_FARMS}]))
    ok = True
    for s, fds in sets:
        mem, one = {}, {}
        for b, seeds in BAGS.items():
            o, o1 = np.full(len(lab), np.nan), np.full(len(lab), np.nan)
            for fd in fds:
                trm, vam = split_mask(lab, fd)
                o[vam], o1[vam] = bag_predict(X[trm], y[trm], w[trm], X[vam], seeds)
            mem[b], one[b] = o, o1
            g = ~np.isnan(o)
            print("  temp %s bag %d: bag %.5f vs single %.5f" % (s, b, rmse(o[g], y[g]), rmse(o1[g], y[g])), flush=True)
        np.save(env.LOCAL + "/web_tabpfn_v2_temp_%s.npy" % s, np.vstack([mem[1], mem[2]]))
        cx = z["%s__CODEX__726" % s]
        for bs in (7, 101):
            base = z["%s__MASK__%d" % (s, bs)]
            g = ~np.isnan(base)
            ref = 0.8 * base + 0.2 * cx
            for b in BAGS:
                cand = 0.7 * base + 0.2 * cx + 0.1 * mem[b]
                pr, lo, hi, pw = boot(lab[g].reset_index(drop=True), "sub_temp", ref[g], cand[g])
                d = rmse(cand[g], y[g]) / rmse(ref[g], y[g]) - 1
                print("TEMP %-6s base %3d bag %d | ref %.5f cand %.5f (%+.2f%%) [%+.4f, %+.4f]"
                      % (s, bs, b, rmse(ref[g], y[g]), rmse(cand[g], y[g]), 100 * d, lo, hi), flush=True)
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
    for vset, fds in (("A", folds("A")), ("B", folds("B")), ("DIAG10", diag_folds(lab))):
        for sd, b in ((7, 1), (8, 2)):
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
                tp, t1 = bag_predict(X[trm], y[trm], ones[trm], X[vam], BAGS[b])
                pb = np.clip(causal_shrink(raw, va, 0.5), 0.062, 3.46)
                pc = np.clip(causal_shrink(0.8 * raw + 0.2 * tp, va, 0.5), 0.062, 3.46)
                idx = np.where(vam)[0]
                pooled["base"][idx], pooled["cand"][idx] = pb, pc
                per.append((rmse(pb, y[idx]), rmse(pc, y[idx]), rmse(tp, y[idx]), rmse(t1, y[idx])))
            P = np.array(per)
            if vset in ("A", "B"):
                d = P[:, 1].mean() / P[:, 0].mean() - 1
                print("EC   %-6s seed %d/bag %d | base %.4f cand %.4f (%+.2f%%, better %d/%d) | bag %.4f single %.4f"
                      % (vset, sd, b, P[:, 0].mean(), P[:, 1].mean(), 100 * d, int((P[:, 1] < P[:, 0]).sum()), len(P),
                         P[:, 2].mean(), P[:, 3].mean()), flush=True)
                ok = ok and d < 0
            else:
                g = ~np.isnan(pooled["base"])
                pr, lo, hi, pw = boot(lab[g].reset_index(drop=True), "sub_ec", pooled["base"][g], pooled["cand"][g])
                d = rmse(pooled["cand"][g], y[g]) / rmse(pooled["base"][g], y[g]) - 1
                print("EC   DIAG10 seed %d/bag %d | base %.4f cand %.4f (%+.2f%%) [%+.4f, %+.4f] | bag %.4f single %.4f"
                      % (sd, b, rmse(pooled["base"][g], y[g]), rmse(pooled["cand"][g], y[g]), 100 * d, lo, hi,
                         P[:, 2].mean(), P[:, 3].mean()), flush=True)
                ok = ok and d < 0 and hi < 0
    print("EC PRE-SET RULE VERDICT:", "ADOPT" if ok else "REJECT", flush=True)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    if which in ("temp", "both"):
        temperature()
    if which in ("ec", "both"):
        ec()
