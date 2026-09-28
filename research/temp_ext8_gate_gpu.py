# -*- coding: utf-8 -*-
"""Out-of-sample check of the temperature gate G_C (catalog 6.67) on EXT8, a
validator never used for any temperature decision (days with min 3-h indoor
temperature < 8 C held out at once; base/Codex OOFs from temp_ext8_base.py).

TabPFN members on GPU for the EXT8 split: samples 1-8 and 17-24 (8-sample
means; W10 uses the first 4 of each).
Pre-set rule: G_C must beat BOTH the pre-TabPFN reference (0.8 base + 0.2
Codex) and W10 (0.7 / 0.2 / 0.1 four-sample) in all 4 cells (seed pairs
7/726, 101/727 x 2 members).  Also printed: W20_8 and the <= 8 C band.

Run:  cd research && PYTHONPATH="" <python> -u temp_ext8_gate_gpu.py
"""
import os
import sys

import env  # noqa: F401
import env_extra_gpu  # noqa: F401
import numpy as np

import common
from common import split_mask, rmse, TARGET_FARMS
from harness import load
import features_v4 as F4
import train_flags_v6 as TF
import web_tabpfn_v2_gpu  # noqa: F401  (CUDA float32 TabPFN call)
import web_tabpfn_v2 as V2

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "analysis", "codex_independent", "2차"))
from resid_reset_features import build_features, FEATURE_COLUMNS  # noqa: E402


def main():
    tX, ty, sX = common.load_raw()
    _, lab0, _ = load()
    z = np.load(env.LOCAL + "/temp_ext8_oof.npz", allow_pickle=True)
    F = build_features(tX, sX).drop(columns=["farm", "day", "hour", "t"]).set_index("row_id")
    lab = lab0[["row_id", "farm", "day", "hour", "t", "sub_temp"]].join(F, on="row_id")
    t_raw = lab0.in_temp.values
    assert (lab.row_id.values == z["row_id"]).all()
    w = TF.row_weights(lab, 0.2, w_noisy=0.2)
    y = lab.sub_temp.values
    X = lab[FEATURE_COLUMNS].values.astype(np.float32)
    ph = F4.phys_features().set_index("row_id").loc[lab.row_id]
    dmin = ph.groupby([lab.farm.values, lab.day.values]).ph_in_temp_3.min()
    cd = dmin[dmin < 8.0]
    fd = {f: set(int(d) for (ff, d) in cd.index if ff == f) for f in TARGET_FARMS}
    trm, vam = split_mask(lab, fd)
    per = {}
    for sd in list(range(1, 9)) + list(range(17, 25)):
        o = np.full(len(lab), np.nan)
        o[vam] = V2.tabpfn_fit_predict(X[trm], y[trm], w[trm], X[vam], sd)
        per[sd] = o
    t = t_raw
    gate = np.where(np.isnan(t), 1.0, np.clip((t - 8.0) / 2.0, 0, 1))
    ok = True
    for name, seeds in (("1-8", range(1, 9)), ("17-24", range(17, 25))):
        s = list(seeds)
        m8, m4 = np.mean([per[k] for k in s], axis=0), np.mean([per[k] for k in s[:4]], axis=0)
        for bs, cs in ((7, 726), (101, 727)):
            base, cx = z["EXT8__MASK__%d" % bs], z["EXT8__CODEX__%d" % cs]
            g = ~np.isnan(base) & ~np.isnan(m8)
            P = {"noPFN": 0.8 * base + 0.2 * cx, "W10": 0.7 * base + 0.2 * cx + 0.1 * m4,
                 "W20_8": 0.6 * base + 0.2 * cx + 0.2 * m8,
                 "G_C": 0.6 * base + (0.4 - 0.2 * gate) * cx + 0.2 * gate * m8}
            r = {k: rmse(p[g], y[g]) for k, p in P.items()}
            ok = ok and r["G_C"] < r["noPFN"] and r["G_C"] < r["W10"]
            cold = g & (t <= 8)
            print("EXT8 seeds %3d/%d member %-5s (%d rows) | %s | <=8C (%d rows): %s"
                  % (bs, cs, name, g.sum(), " | ".join("%s %.4f" % kv for kv in r.items()), cold.sum(),
                     " | ".join("%s %.3f" % (k, rmse(p[cold], y[cold])) for k, p in P.items())), flush=True)
    print("G_C EXT8 VERDICT:", "ADOPT" if ok else "REJECT", flush=True)


if __name__ == "__main__":
    main()
