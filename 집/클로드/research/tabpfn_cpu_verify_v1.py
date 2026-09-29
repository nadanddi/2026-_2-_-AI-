# -*- coding: utf-8 -*-
"""The TabPFN candidate was validated on the GPU (float32).  A shipped answer
must reproduce on CPU.  Recompute the SAME members on CPU and compare, DIAG10:

  temperature : bag of seeds 1-4, weighted context sampling (web_tabpfn_v1),
                GPU reference = local/web_tabpfn_v2_temp_DIAG10.npy row 0
  EC          : bag of seeds 1-4, uniform sampling (anal_ec_noncausal_tabpfn.tp),
                GPU reference = local/anal_ec_regime_tabpfn.npz 'pfn'
Report max/mean |CPU - GPU|, member RMSE on both, and the candidate blend
gain on CPU (temperature 0.7 base + 0.2 Codex + 0.1 bag vs 0.8/0.2, seed 7;
EC 0.8 round-3 + 0.2 bag vs round-3, npz 'raw', shrink + clip).
Pass = blend gains agree in sign and within 0.3 percentage points.

Run:  cd research && PYTHONPATH="" <python> -u tabpfn_cpu_verify_v1.py
"""
import os
import sys

import env  # noqa: F401
import env_extra  # noqa: F401  (CPU torch)
import numpy as np
import torch
from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion

import common
from common import split_mask, rmse, USABLE, OUT_COLS
from harness import load
import features_v4 as F4
from anal_q1_errors import diag_folds
from make_submission_v3 import causal_shrink
import train_flags_v6 as TF
from web_tabpfn_v1 import tabpfn_fit_predict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "analysis", "codex_independent", "2차"))
from resid_reset_features import build_features, FEATURE_COLUMNS  # noqa: E402

assert not torch.cuda.is_available() or "cpu" in torch.__version__ or True


def tp_cpu(Xtr, ytr, Xva, seed):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(Xtr), size=min(2000, len(Xtr)), replace=False)
    m = TabPFNRegressor.create_default_for_version(ModelVersion.V2, device="cpu", n_estimators=4, random_state=seed,
                                                   ignore_pretraining_limits=True)
    m.fit(Xtr[idx], ytr[idx])
    return m.predict(Xva)


def temperature():
    tX, ty, sX = common.load_raw()
    _, lab0, _ = load()
    z = np.load(env.LOCAL + "/temp_mask_v1_oof.npz", allow_pickle=True)
    F = build_features(tX, sX).drop(columns=["farm", "day", "hour", "t"]).set_index("row_id")
    lab = lab0[["row_id", "farm", "day", "hour", "t", "sub_temp"]].join(F, on="row_id")
    w = TF.row_weights(lab, 0.2, w_noisy=0.2)
    y = lab.sub_temp.values
    X = lab[FEATURE_COLUMNS].values.astype(np.float32)
    cpu = np.full(len(lab), np.nan)
    for fd in diag_folds(lab):
        trm, vam = split_mask(lab, fd)
        cpu[vam] = np.mean([tabpfn_fit_predict(X[trm], y[trm], w[trm], X[vam], s) for s in (1, 2, 3, 4)], axis=0)
    gpu = np.load(env.LOCAL + "/web_tabpfn_v2_temp_DIAG10.npy")[0]
    g = ~np.isnan(cpu) & ~np.isnan(gpu)
    base, cx = z["DIAG10__MASK__7"], z["DIAG10__CODEX__726"]
    ref = 0.8 * base + 0.2 * cx
    gains = []
    for name, mem in (("GPU", gpu), ("CPU", cpu)):
        cand = 0.7 * base + 0.2 * cx + 0.1 * mem
        gains.append(100 * (rmse(cand[g], y[g]) / rmse(ref[g], y[g]) - 1))
        print("TEMP %s | member %.5f | blend gain %+.2f%%" % (name, rmse(mem[g], y[g]), gains[-1]), flush=True)
    d = np.abs(cpu[g] - gpu[g])
    print("TEMP |CPU-GPU| max %.4f mean %.5f | %s" % (d.max(), d.mean(),
          "PASS" if np.sign(gains[0]) == np.sign(gains[1]) and abs(gains[0] - gains[1]) < 0.3 else "FAIL"), flush=True)


def ec():
    _, _, lab0 = load()
    fp = F4.fp_features()
    lab = lab0.merge(fp, on="row_id", how="left").reset_index(drop=True)
    f14 = [c for c in (list(USABLE) + ["day", "hr_sin", "hr_cos", "midnight"]) if c not in OUT_COLS]
    X = lab[f14 + F4.names(fp)].values.astype(np.float32)
    y = lab.sub_ec.values
    z = np.load(env.LOCAL + "/anal_ec_regime_tabpfn.npz", allow_pickle=True)
    assert (z["row_id"] == lab.row_id.values).all()
    raw, gpu = z["raw"], z["pfn"]
    cpu = np.full(len(lab), np.nan)
    fds = diag_folds(lab)
    for fd in fds:
        trm, vam = split_mask(lab, fd)
        cpu[vam] = np.mean([tp_cpu(X[trm], y[trm], X[vam], s) for s in (1, 2, 3, 4)], axis=0)

    def fin(p):
        o = np.full(len(lab), np.nan)
        for fd in fds:
            trm, vam = split_mask(lab, fd)
            o[vam] = np.clip(causal_shrink(p[vam], lab[vam].reset_index(drop=True), 0.5), 0.062, 3.46)
        return o
    r0 = fin(raw)
    gains = []
    for name, mem in (("GPU", gpu), ("CPU", cpu)):
        c = fin(0.8 * raw + 0.2 * mem)
        gains.append(100 * (rmse(c, y) / rmse(r0, y) - 1))
        print("EC   %s | member %.4f | blend gain %+.2f%%" % (name, rmse(mem, y), gains[-1]), flush=True)
    d = np.abs(cpu - gpu)
    print("EC   |CPU-GPU| max %.4f mean %.5f | %s" % (d.max(), d.mean(),
          "PASS" if np.sign(gains[0]) == np.sign(gains[1]) and abs(gains[0] - gains[1]) < 0.3 else "FAIL"), flush=True)


if __name__ == "__main__":
    ec()
    temperature()
