# -*- coding: utf-8 -*-
"""submission_26 temperature, stage 0 (2026-10-15 집 클로드, '5-point / field-information' build; catalog G5 / A3):
W40G-S 'NX' = the 9th-round W40G-S (make_submission_v13_temp.py, same members, weights, seeds, season tables) with the two
competition-file markers removed:
  - BASE members: the raw record-day index `day` is dropped from the feature list (ct).
  - Codex and TabPFN members: `second` (= day >= 179, 'pass-2 / evaluation-period' flag) is dropped from FEATURE_COLUMNS.
  (The date information stays as the season index of the Codex/TabPFN members = calendar date in the field.)
Local evidence (G5, same validators as TT1): W40G-S NX vs W40G-S: DIAG10 +0.4%, EL1 -0.5%, EXT10 -0.4%, EXT12 -0.6%
(A3, through chain + anchors: DIAG10 +0.8%, EL1 -0.6%, EXT10 -0.6%, EXT12 -1.0%).
Run:  cd code && python make_submission_v26_temp.py <out.csv>     (data from env.DATA / AGRI_DATA; CPU)
"""
import os
import sys
import time

import env  # noqa: F401  MUST be the first project import
import env_extra  # noqa: F401
import numpy as np
import pandas as pd

import make_submission_v13_temp as V13
import train_flags_v6 as TF
import resid_reset_features as RR


def main():
    out = sys.argv[1]
    assert not os.path.exists(out), "refusing to overwrite %s" % out
    if "second" in RR.FEATURE_COLUMNS: RR.FEATURE_COLUMNS.remove("second")     # same list object used by V13 and temp_mask_v1
    assert "second" not in V13.FEATURE_COLUMNS
    lab, test, ct, phc, sX = V13.build_frames()
    ct = [c for c in ct if c != "day"]
    w = TF.row_weights(lab, V13.W_FLAG, w_noisy=V13.W_NOISY)
    t0 = time.time()
    base, codex, pfn_all = V13.predict_members(lab, test, ct, phc, w, log=lambda s: print(s, flush=True))
    g = V13.gate(test.in_temp.values); pt = V13.combine(base, codex, pfn_all.mean(0), g)
    S = pd.DataFrame({"row_id": sX.row_id.values, "sub_temp": pt})
    assert len(S) == 1440 and S.row_id.is_unique and np.isfinite(S.sub_temp).all()
    S.to_csv(out, index=False, encoding="utf-8", float_format="%.6f")
    print("wrote %s | NX: base cols %d (no `day`), codex/tabpfn cols %d (no `second`) | %.0fs" % (out, len(ct), len(V13.FEATURE_COLUMNS), time.time() - t0))


if __name__ == "__main__":
    main()
