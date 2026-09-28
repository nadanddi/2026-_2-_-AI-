# -*- coding: utf-8 -*-
"""Why did tabpfn_checks_v1 [O] fail (reversed test order moved predictions
by up to 0.003)?  Numerical (float32 batch-shape effects) or a real path
between test rows?

One fitted TabPFN (temperature, the checks_v1 base pipeline).  For the first
40 test rows compare predictions made:
  a) in the full 1440-row batch        b) in the reversed batch
  c) each row alone                    d) with 1400 random-noise partner rows
under float32 (default) and float64 (inference_precision=torch.float64).
Information path  -> (d) differs from (c) by much more than (a)/(b)/(c) differ,
                     and the gap survives float64.
Numerical         -> all gaps shrink to ~1e-10 in float64.

Run:  cd research && PYTHONPATH="" <python> -u tabpfn_rowdiag_v1.py
"""
import env  # noqa: F401
import env_extra  # noqa: F401
import numpy as np
import torch
from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion

import harness
import train_flags_v6 as TF
from tabpfn_checks_v1 import temp_frames, ORIG, N_CTX

K = 40


def main():
    lab_t = harness.load()[1]
    w = TF.row_weights(lab_t, 0.2, w_noisy=0.2)
    harness._CACHE.clear()
    Xtr, ytr, Xte = temp_frames(ORIG)
    rng = np.random.default_rng(1)
    idx = rng.choice(len(Xtr), size=N_CTX, replace=False, p=w / w.sum())
    noise = (Xte[K:] * np.random.default_rng(5).normal(3.0, 2.0, Xte[K:].shape)).astype(np.float32)
    for prec in ("auto", torch.float64):
        m = TabPFNRegressor.create_default_for_version(ModelVersion.V2, device="cpu", n_estimators=4, random_state=1,
                                                       ignore_pretraining_limits=True, inference_precision=prec)
        m.fit(Xtr[idx], ytr[idx])
        a = m.predict(Xte)[:K]
        b = m.predict(Xte[::-1])[::-1][:K]
        c = np.array([m.predict(Xte[i:i + 1])[0] for i in range(K)])
        d = m.predict(np.vstack([Xte[:K], noise]))[:K]
        name = "float32" if prec == "auto" else "float64"
        print("%s | max|a-b| %.3g  max|a-c| %.3g  max|c-d| %.3g  max|a-d| %.3g"
              % (name, np.abs(a - b).max(), np.abs(a - c).max(), np.abs(c - d).max(), np.abs(a - d).max()), flush=True)


if __name__ == "__main__":
    main()
