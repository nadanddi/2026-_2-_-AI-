# -*- coding: utf-8 -*-
"""EXT8 validator for temperature (never used for any temperature decision):
all days whose minimum 3-h smoothed indoor temperature is below 8 C held out
at once.  Builds the MASK base (seeds 7/101) and Codex (726/727) OOFs exactly
as temp_mask_v1.py does, saved to local/temp_ext8_oof.npz.  The TabPFN part
and the gate test follow in temp_ext8_gate_gpu.py.

Run:  cd research && PYTHONPATH="" <python> -u temp_ext8_base.py
"""
import env  # noqa: F401
import numpy as np

import common
import harness
import cold_v5
from common import split_mask, TARGET_FARMS
from screen_v6 import temp_members, collect
import train_flags_v6 as TF
from temp_mask_v1 import masked_loader, build_world, ORIG, codex_fit_predict

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "analysis", "codex_independent", "2차"))
from resid_reset_features import build_features, FEATURE_COLUMNS  # noqa: E402


def main():
    common.load_raw = masked_loader
    try:
        lab, ct, phc = build_world()
    finally:
        common.load_raw = ORIG
        harness._CACHE.clear()
    tX, ty, sX = ORIG()
    CF = build_features(tX, sX).drop(columns=["farm", "day", "hour", "t"]).set_index("row_id")
    for c in FEATURE_COLUMNS:
        if c not in lab.columns:
            lab[c] = CF.loc[lab.row_id, c].values
    w = TF.row_weights(lab, 0.2, w_noisy=0.2)
    dmin = lab.groupby(["farm", "day"]).ph_in_temp_3.min()
    cd = dmin[dmin < 8.0]
    fds = [{f: set(int(d) for (ff, d) in cd.index if ff == f) for f in TARGET_FARMS}]
    print("EXT8 held-out days: %d" % len(cd), flush=True)
    out = {}
    for sd in (7, 101):
        cold_v5.SEED = sd
        M = collect(lab, fds, lambda tr, va, m: temp_members(tr, va, ct, phc, w[m]))
        out["EXT8__MASK__%d" % sd] = 0.65 * M["res"] + 0.25 * M["ridge"] + 0.10 * M["nys"]
    cold_v5.SEED = 7
    for sc in (726, 727):
        o = np.full(len(lab), np.nan)
        for fd in fds:
            trm, vam = split_mask(lab, fd)
            o[vam] = codex_fit_predict(lab[trm], lab[vam], w[trm], sc)
        out["EXT8__CODEX__%d" % sc] = o
    np.savez(env.LOCAL + "/temp_ext8_oof.npz", row_id=lab.row_id.values, **out)
    print("saved", flush=True)


if __name__ == "__main__":
    main()
