# -*- coding: utf-8 -*-
"""Feature blocks for submission 5 (platform round 4): features_v4 + cold hinges.

Round 3 confirmed that the test's cold nights dominate the temperature error
(the round-2 -> round-3 change moved 81 rows below 6 C by -1.26 C on average
and the real score improved 15.6%).  The extrapolation validator
(cold_v5.py), which reproduces the leaderboard ratio (0.858 vs 0.844 real),
still shows the round-3 model over-predicting cold rows (+0.06 / +0.54 /
+0.66 at hold-out thresholds 7 / 8 / 10 C), while training labels say the slab
runs further below the air the colder it gets.

Hinge terms let the linear physics baseline bend at the cold end:
    hg_ewm3_k = max(0, k - ewm(in_temp, hl=3 h)),  k = 8, 10, 12
    hg_raw_k  = max(0, k - in_temp),               k = 8, 10, 12
They are pointwise functions of phys_features columns, which are causal
(features_v4 / causality_v4.py), so they are causal too.
"""
import numpy as np
import pandas as pd

from features_v4 import seg_features, phys_features, fp_features, names  # noqa: F401

HINGE_K = (8, 10, 12)


def hinge_features(ph):
    """ph: output of phys_features()."""
    h = pd.DataFrame({"row_id": ph.row_id.values})
    for k in HINGE_K:
        h["hg_ewm3_%d" % k] = np.maximum(0.0, k - ph["ph_in_temp_3"].values)
        h["hg_raw_%d" % k] = np.maximum(0.0, k - ph["ph_in_temp_None"].values)
    return h
