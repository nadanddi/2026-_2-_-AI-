# -*- coding: utf-8 -*-
"""End-to-end rule checks for make_submission_v17_temp.py (W40G-S + day-gated GRU member; whole pipeline:
features -> weights -> BASE + CODEX + TABPFN -> gate -> GRU sequences (MASK world for training) -> day gate).

TabPFN uses one sample here (sample 1) to keep the run short; the property
checked (which inputs a prediction can depend on) does not depend on how many
samples are averaged.

  [M] MASK      training-row features are identical in every run (the fitted
                models never see test inputs, so they cannot change)
  [F] future    per greenhouse, 2 cuts: test_X inputs after the cut x10 + noise
                + 33% blanking -> every test prediction at or before the cut
                is unchanged (bit for bit)
  [X] isolation other greenhouse's test_X x3 + noise -> this greenhouse's test
                predictions unchanged
  [D] determinism  two clean runs identical

Run:  cd code && python temp_v17_checks.py
"""
import sys

import env  # noqa: F401
import env_extra  # noqa: F401
import numpy as np

from common import USABLE, TARGET_FARMS
import train_flags_v6 as TF
import make_submission_v13_temp as V8
import make_submission_v17_temp as V17

ORIG = V8.ORIG


def run(loader=None):
    R = V17.final_prediction(loader, pfn_samples=(1,), log=lambda s: None)
    lab = R["lab"]
    seq = V17.gru_sequences(loader, lab.row_id.values, mask_test=True)
    feats = np.concatenate([lab[R["ct"] + R["phc"] + [c for c in lab.columns if c.startswith("cx__")]].to_numpy(float).ravel(), seq.ravel()])
    return R["pred"], feats, R["sX"]


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


def main():
    fails = []
    p0, f0, sX = run()
    p1, f1, _ = run()
    ok = same(p0, p1) and same(f0, f1)
    print("[D] determinism: %s (max |diff| %.3g)" % ("PASS" if ok else "FAIL", np.abs(p0 - p1).max()), flush=True)
    fails += [] if ok else ["D"]

    for farm in TARGET_FARMS:
        ts = np.sort(sX[sX.farm == farm].t.unique())
        for frac, hh in ((0.3, 7), (0.8, 18)):
            T = int(ts[int(len(ts) * frac)] // 24 * 24 + hh)
            p, f, _ = run(perturb_future(farm, T))
            past = ((sX.farm == farm) & (sX.t <= T)).values
            fut = ((sX.farm == farm) & (sX.t > T)).values
            okm, okp = same(f0, f), same(p0[past], p[past])
            print("[F] %s cut day %d h%02d: %s | past rows %d unchanged %s, MASK features same %s, future rows moved %.3f"
                  % (farm, T // 24, T % 24, "PASS" if okm and okp else "FAIL", int(past.sum()), okp, okm,
                     float(np.abs(p0[fut] - p[fut]).mean())), flush=True)
            fails += [] if okm and okp else ["F %s %d" % (farm, T)]
        p, f, _ = run(perturb_other(farm))
        own = (sX.farm == farm).values
        okm, okp = same(f0, f), same(p0[own], p[own])
        print("[X] %s isolation: %s | own rows unchanged %s, MASK features same %s, other farm moved %.3f"
              % (farm, "PASS" if okm and okp else "FAIL", okp, okm, float(np.abs(p0[~own] - p[~own]).mean())), flush=True)
        fails += [] if okm and okp else ["X %s" % farm]

    print("\n%s" % ("ALL PASS" if not fails else "FAIL %s" % fails))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
