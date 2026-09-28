# -*- coding: utf-8 -*-
"""web_tabpfn_v2.py on the laptop GPU (RTX 4050), for speed only.
Same bags, same blend weights, same pre-set rules; only the TabPFN call runs
on CUDA with inference_precision fixed to float32 (no fp16 autocast), so the
member is numerically close to, but not bit-identical with, the CPU member.
Anything shipped must be regenerated and re-checked on CPU.

Run:  cd research && PYTHONPATH="" <python> -u web_tabpfn_v2_gpu.py [temp|ec|both]
"""
import sys

import env  # noqa: F401
import env_extra_gpu  # noqa: F401  (CUDA torch first)
import numpy as np
import torch
from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion

import web_tabpfn_v1 as V1
import web_tabpfn_v2 as V2

assert torch.cuda.is_available(), "CUDA torch not loaded"


def tabpfn_fit_predict_gpu(Xtr, ytr, wtr, Xva, seed):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(Xtr), size=min(V1.N_CTX, len(Xtr)), replace=False, p=wtr / wtr.sum())
    m = TabPFNRegressor.create_default_for_version(ModelVersion.V2, device="cuda", n_estimators=4,
                                                   random_state=seed, ignore_pretraining_limits=True,
                                                   inference_precision=torch.float32)
    m.fit(Xtr[idx], ytr[idx])
    return m.predict(Xva)


V2.tabpfn_fit_predict = tabpfn_fit_predict_gpu

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    if which in ("temp", "both"):
        V2.temperature()
    if which in ("ec", "both"):
        V2.ec()
