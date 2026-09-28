# -*- coding: utf-8 -*-
"""Rule and reproducibility checks for a TabPFN member, END TO END (features
-> context sample -> TabPFN fit -> test predictions), before any use.

Pipeline under test (the way a submission would use it):
  training rows : features built in the MASK world (every test_X input NaN)
  test rows     : features built from the full history (train + test inputs)
  context       : 2,000 training rows sampled with the round-5 weights
                  (temperature) or uniformly (EC), seed 1
  temperature   : Codex resid_reset features (89)
  EC            : f14 + fingerprint (38), as web_tabpfn_v1.py

Checks (all must be exact, bit for bit):
  [D] determinism        two independent runs -> identical predictions
  [O] row independence   test rows predicted in reversed order -> identical
  [F] future invariance  per greenhouse, 4 cuts: test_X inputs after the cut
                         x10 + noise + 33% blanking -> every test prediction
                         at or before the cut unchanged
  [X] greenhouse isol.   other greenhouse's test_X x3 + noise -> this
                         greenhouse's test predictions unchanged
  [M] MASK               training-row features identical in every run above
                         (the context never sees test inputs)

Run:  cd research && PYTHONPATH="" <python> -u tabpfn_checks_v1.py
"""
import os
import sys

import env  # noqa: F401
import env_extra  # noqa: F401
import numpy as np
from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion

import common
import harness
from common import USABLE, OUT_COLS, TARGET_FARMS
import features_v4 as F4
import train_flags_v6 as TF

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "analysis", "codex_independent", "2차"))
from resid_reset_features import build_features, FEATURE_COLUMNS  # noqa: E402

ORIG = common.load_raw
N_CTX = 2000
SEED = 1


def masked(loader):
    def f():
        tX, ty, sX = loader()
        sX = sX.copy()
        sX[[c for c in sX.columns if c not in ("row_id", "farm", "day", "hour", "t")]] = np.nan
        return tX, ty, sX
    return f


def run_with(loader, fn):
    common.load_raw = loader
    harness._CACHE.clear()
    try:
        return fn()
    finally:
        common.load_raw = ORIG
        harness._CACHE.clear()


def temp_frames(loader):
    """(training X, y, test X) for temperature."""
    tX, ty, sX = ORIG()
    lab = run_with(ORIG, lambda: harness.load()[1])[["row_id", "farm", "day", "sub_temp"]]
    a, _, c = masked(loader)()
    FM = build_features(a, c).set_index("row_id")
    a, _, c = loader()
    FF = build_features(a, c).set_index("row_id")
    Xtr = FM.loc[lab.row_id, FEATURE_COLUMNS].values.astype(np.float32)
    Xte = FF.loc[sX.row_id, FEATURE_COLUMNS].values.astype(np.float32)
    return Xtr, lab.sub_temp.values, Xte


def ec_frames(loader):
    f14 = [c for c in (list(USABLE) + ["day", "hr_sin", "hr_cos", "midnight"]) if c not in OUT_COLS]

    def build():
        panel = harness.load()[0]
        fp = F4.fp_features()
        cols = f14 + F4.names(fp)
        return panel.merge(fp, on="row_id", how="left").set_index("row_id"), cols

    tX, ty, sX = ORIG()
    lab = run_with(ORIG, lambda: harness.load()[2])
    PM, cols = run_with(masked(loader), build)
    PF, _ = run_with(loader, build)
    Xtr = PM.loc[lab.row_id, cols].values.astype(np.float32)
    Xte = PF.loc[sX.row_id, cols].values.astype(np.float32)
    return Xtr, lab.sub_ec.values, Xte


def fit_predict(Xtr, ytr, wtr, Xte, seed=SEED):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(Xtr), size=min(N_CTX, len(Xtr)), replace=False, p=wtr / wtr.sum())
    m = TabPFNRegressor.create_default_for_version(ModelVersion.V2, device="cpu", n_estimators=4,
                                                   random_state=seed, ignore_pretraining_limits=True)
    m.fit(Xtr[idx], ytr[idx])
    return m.predict(Xte), m


def same(a, b):
    return bool(((a == b) | (np.isnan(a) & np.isnan(b))).all())


def perturb_future(farm, T):
    def ld():
        a, b, c = ORIG()
        c = c.copy()
        rng = np.random.RandomState(T)
        m = ((c.farm == farm) & (c.t > T)).values
        for col in USABLE:
            c.loc[m, col] = c.loc[m, col] * 10.0 + rng.normal(0, 5.0, int(m.sum()))
        c.loc[m & (rng.rand(len(c)) < 0.33), USABLE] = np.nan
        return a, b, c
    return ld


def perturb_other(farm):
    def ld():
        a, b, c = ORIG()
        c = c.copy()
        rng = np.random.RandomState(7)
        m = (c.farm != farm).values
        for col in USABLE:
            c.loc[m, col] = c.loc[m, col] * 3.0 + rng.normal(0, 5.0, int(m.sum()))
        return a, b, c
    return ld


def check(name, frames, weights):
    _, _, sX = ORIG()
    fails = []
    Xtr, ytr, Xte = frames(ORIG)
    w = weights(len(Xtr))
    p0, m0 = fit_predict(Xtr, ytr, w, Xte)
    print("[%s] base: %d context rows, %d test rows, pred mean %.4f" % (name, N_CTX, len(p0), p0.mean()), flush=True)

    Xtr2, ytr2, Xte2 = frames(ORIG)
    p1, _ = fit_predict(Xtr2, ytr2, w, Xte2)
    ok = same(Xtr, Xtr2) and same(Xte, Xte2) and same(p0, p1)
    print("[%s][D] determinism: %s (max |diff| %.3g)" % (name, "PASS" if ok else "FAIL", np.abs(p0 - p1).max()), flush=True)
    fails += [] if ok else [("D",)]

    pr = m0.predict(Xte[::-1])[::-1]
    ok = same(p0, pr)
    print("[%s][O] reversed order: %s (max |diff| %.3g)" % (name, "PASS" if ok else "FAIL", np.abs(p0 - pr).max()), flush=True)
    fails += [] if ok else [("O",)]

    for farm in TARGET_FARMS:
        ts = np.sort(sX[sX.farm == farm].t.unique())
        for frac, hh in ((0.1, 5), (0.4, 13), (0.7, 0), (0.9, 23)):
            T = int(ts[int(len(ts) * frac)] // 24 * 24 + hh)
            a, b, c = frames(perturb_future(farm, T))
            p, _ = fit_predict(a, b, w, c)
            past = ((sX.farm == farm) & (sX.t <= T)).values
            fut = ((sX.farm == farm) & (sX.t > T)).values
            okm = same(Xtr, a)
            ok = okm and same(p0[past], p[past])
            print("[%s][F] %s cut day %d h%02d: %s | past rows %d, MASK context same %s, future rows moved %.3f"
                  % (name, farm, T // 24, T % 24, "PASS" if ok else "FAIL", int(past.sum()), okm,
                     float(np.abs(p0[fut] - p[fut]).mean()) if fut.any() else 0.0), flush=True)
            fails += [] if ok else [("F", farm, T)]

        a, b, c = frames(perturb_other(farm))
        p, _ = fit_predict(a, b, w, c)
        own = (sX.farm == farm).values
        okm = same(Xtr, a)
        ok = okm and same(p0[own], p[own])
        print("[%s][X] %s isolation: %s | MASK context same %s, other farm moved %.3f"
              % (name, farm, "PASS" if ok else "FAIL", okm, float(np.abs(p0[~own] - p[~own]).mean())), flush=True)
        fails += [] if ok else [("X", farm)]
    return fails


def main():
    lab_t = harness.load()[1]
    wt = TF.row_weights(lab_t, 0.2, w_noisy=0.2)
    harness._CACHE.clear()
    fails = check("TEMP", temp_frames, lambda n: wt)
    fails += check("EC", ec_frames, lambda n: np.ones(n))
    print("\n%s" % ("ALL PASS" if not fails else "FAIL %s" % fails))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
