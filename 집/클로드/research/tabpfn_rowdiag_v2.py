# -*- coding: utf-8 -*-
"""Faster version of tabpfn_rowdiag_v1.py (v1 ran >4 h on CPU without output:
40 single-row predictions, each a full forward pass, twice).

Same question: is the temperature [O] failure (reversed test order moved
predictions by up to 0.003) numerical or an information path between test
rows?  One fitted TabPFN (CPU, as the check), K = 5 test rows:
  a) full 1440-row batch        b) reversed batch
  c) each row alone             d) the 5 rows + 1435 noise partner rows
float32 first (printed immediately), then float64.
Information path -> |c-d| large and persists in float64.
Numerical        -> gaps shrink by orders of magnitude in float64.

Run:  cd research && PYTHONPATH="" <python> -u tabpfn_rowdiag_v2.py
"""
import time

import env  # noqa: F401
import env_extra  # noqa: F401
import numpy as np
import torch
from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion

import harness
import train_flags_v6 as TF
from tabpfn_checks_v1 import temp_frames, ORIG, N_CTX

K = 5


def main():
    t0 = time.time()
    lab_t = harness.load()[1]
    w = TF.row_weights(lab_t, 0.2, w_noisy=0.2)
    harness._CACHE.clear()
    Xtr, ytr, Xte = temp_frames(ORIG)
    rng = np.random.default_rng(1)
    idx = rng.choice(len(Xtr), size=N_CTX, replace=False, p=w / w.sum())
    noise = (Xte[K:] * np.random.default_rng(5).normal(3.0, 2.0, Xte[K:].shape)).astype(np.float32)
    print("frames ready %.0fs" % (time.time() - t0), flush=True)
    for prec in ("auto", torch.float64):
        name = "float32" if prec == "auto" else "float64"
        m = TabPFNRegressor.create_default_for_version(ModelVersion.V2, device="cpu", n_estimators=4, random_state=1,
                                                       ignore_pretraining_limits=True, inference_precision=prec)
        m.fit(Xtr[idx], ytr[idx])
        a = m.predict(Xte)[:K]
        b = m.predict(Xte[::-1])[::-1][:K]
        print("%s a,b done %.0fs | max|a-b| %.3g" % (name, time.time() - t0, np.abs(a - b).max()), flush=True)
        c = np.array([m.predict(Xte[i:i + 1])[0] for i in range(K)])
        d = m.predict(np.vstack([Xte[:K], noise]))[:K]
        print("%s | max|a-b| %.3g  max|a-c| %.3g  max|c-d| %.3g  max|a-d| %.3g  (%.0fs)"
              % (name, np.abs(a - b).max(), np.abs(a - c).max(), np.abs(c - d).max(), np.abs(a - d).max(),
                 time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
