# -*- coding: utf-8 -*-
"""Temperature member LPFN: linear physics base + TabPFN on its RESIDUAL.
Motivation: the plain TabPFN member cannot extrapolate into unseen cold
(catalog 6.67), while the "linear + tree residual" structure extrapolates
best on the other greenhouses (6.75).  LPFN = the Codex member's ridge
physics base (PHYSICS_COLUMNS, alpha 100, round-5 weights) + TabPFN fitted on
the residual with the Codex-view features; 8 samples (401..408 / 411..418,
never used), weighted context sampling, GPU float32.

Arms (fixed), reference = candidate G_C2 (6.70) with its own TabPFN member
(samples 1-8 / 17-24 saved OOFs) and the same base/Codex seed pairs:
  LP_FLAT : 0.6*base + 0.2*Codex + 0.2*LPFN            (no gate)
  LP_GATE : G_C2 weights with LPFN in the TabPFN slot
Cells DIAG10/EXT10/EXT12 x seed pairs 7/726, 101/727 x 2 series.
Rule: better than G_C2 in all 12 cells, DIAG10 p_worse < 0.025/2.

Run:  cd research && PYTHONPATH="" <python> -u web_tabpfn_linres_gpu.py
"""
import os
import sys

import env  # noqa: F401
import env_extra_gpu  # noqa: F401
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import common
from common import split_mask, rmse, TARGET_FARMS
from harness import load
import features_v4 as F4
from anal_q1_errors import diag_folds
from screen_v6 import boot
import train_flags_v6 as TF
import web_tabpfn_v2_gpu  # noqa: F401  (CUDA float32 TabPFN call)
import web_tabpfn_v2 as V2

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "analysis", "codex_independent", "2차"))
from resid_reset_features import build_features, FEATURE_COLUMNS, PHYSICS_COLUMNS  # noqa: E402

SERIES = {"A": (tuple(range(401, 409)), "v2"), "B": (tuple(range(411, 419)), "v6")}


def main():
    tX, ty, sX = common.load_raw()
    _, lab0, _ = load()
    z = np.load(env.LOCAL + "/temp_mask_v1_oof.npz", allow_pickle=True)
    F = build_features(tX, sX).drop(columns=["farm", "day", "hour", "t"]).set_index("row_id")
    lab = lab0[["row_id", "farm", "day", "hour", "t", "sub_temp"]].join(F, on="row_id")
    t = lab0.in_temp.values
    assert (lab.row_id.values == z["row_id"]).all()
    w = TF.row_weights(lab, 0.2, w_noisy=0.2)
    y = lab.sub_temp.values
    X = lab[FEATURE_COLUMNS].values.astype(np.float32)
    g8 = np.where(np.isnan(t), 1.0, np.clip((t - 8.0) / 2.0, 0, 1))
    ph = F4.phys_features().set_index("row_id").loc[lab.row_id]
    dmin = ph.groupby([lab.farm.values, lab.day.values]).ph_in_temp_3.min()
    sets = [("DIAG10", diag_folds(lab))]
    for th in (10, 12):
        cd = dmin[dmin < float(th)]
        sets.append(("EXT%d" % th, [{f: set(int(d) for (ff, d) in cd.index if ff == f) for f in TARGET_FARMS}]))
    ok = {"LP_FLAT": True, "LP_GATE": True}
    for s, fds in sets:
        lp = {}
        for k, (seeds, _) in SERIES.items():
            o = np.full(len(lab), np.nan)
            for fd in fds:
                trm, vam = split_mask(lab, fd)
                lin = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(),
                                    Ridge(alpha=100.0))
                lin.fit(lab.loc[trm, PHYSICS_COLUMNS], y[trm], ridge__sample_weight=w[trm])
                r = y[trm] - lin.predict(lab.loc[trm, PHYSICS_COLUMNS])
                res = np.mean([V2.tabpfn_fit_predict(X[trm], r, w[trm], X[vam], sd) for sd in seeds], axis=0)
                o[vam] = lin.predict(lab.loc[vam, PHYSICS_COLUMNS]) + res
            lp[k] = o
            print("  %s series %s done" % (s, k), flush=True)
        np.save(env.LOCAL + "/web_tabpfn_linres_temp_%s.npy" % s, np.vstack([lp["A"], lp["B"]]))
        cxv = {"v2": np.load(env.LOCAL + "/web_tabpfn_v2_temp_%s.npy" % s).mean(0),
               "v6": np.load(env.LOCAL + "/web_tabpfn_v6_temp_%s.npy" % s)[:8].mean(0)}
        for bs, cs in ((7, 726), (101, 727)):
            base, cx = z["%s__MASK__%d" % (s, bs)], z["%s__CODEX__%d" % (s, cs)]
            for k, (_, src) in SERIES.items():
                m, L = cxv[src], lp[k]
                g = ~np.isnan(base) & ~np.isnan(m) & ~np.isnan(L)
                wb, wc = 0.6 - 0.2 * (1 - g8), 0.2 + 0.4 * (1 - g8)
                ref = wb * base + wc * cx + 0.2 * g8 * m
                arms = {"LP_FLAT": 0.6 * base + 0.2 * cx + 0.2 * L, "LP_GATE": wb * base + wc * cx + 0.2 * g8 * L}
                txt = []
                for a, c in arms.items():
                    d = rmse(c[g], y[g]) / rmse(ref[g], y[g]) - 1
                    pw = boot(lab[g].reset_index(drop=True), "sub_temp", ref[g], c[g])[3] if s == "DIAG10" else np.nan
                    ok[a] = ok[a] and d < 0 and (s != "DIAG10" or pw < 0.0125)
                    txt.append("%s %+.2f%% (p_worse %.4f)" % (a, 100 * d, pw))
                cold = g & (t <= 8)
                print("%-6s seeds %3d/%d series %s | LPFN alone %.4f (<=8C %.3f) vs PFN %.4f (<=8C %.3f) | G_C2 %.5f | %s"
                      % (s, bs, cs, k, rmse(L[g], y[g]), rmse(L[cold], y[cold]) if cold.any() else np.nan,
                         rmse(m[g], y[g]), rmse(m[cold], y[cold]) if cold.any() else np.nan, rmse(ref[g], y[g]),
                         " | ".join(txt)), flush=True)
    for a, v in ok.items():
        print("%s PRE-SET RULE VERDICT: %s" % (a, "ADOPT" if v else "REJECT"), flush=True)


if __name__ == "__main__":
    main()
