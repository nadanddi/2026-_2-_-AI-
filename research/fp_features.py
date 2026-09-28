# -*- coding: utf-8 -*-
"""Causal "source fingerprint" features for F13/F47.

Verified structure: F13 and F47 are NOT one continuous greenhouse each.  The
same 24-h outdoor-weather vector (= the same calendar date) appears twice
inside F13 in 90 of 159 weather groups, usually on ADJACENT day indices
(median gap 1), with different indoor temperature (median 0.96 C) and EC
(median 0.154).  Each record is a day-level interleave of several source
greenhouses; the label jumps at midnight when the source changes.

So a row should know WHICH source its day came from.  The source shows in the
actuator operating pattern (heating set point, curtain schedule, fan use).
Everything here is computed within the row's own greenhouse-day, from hour 0
up to the row's own hour, so it uses only current and earlier inputs.

  <v>_h0      value of v at hour 00 of the same day
  <v>_tdm     expanding mean of v over hours 0..h of the same day
  <v>_tdz     expanding share of hours 0..h where v == 0
"""
import numpy as np
import pandas as pd

import common
from common import TARGET_FARMS

ACTS = ["act_vent", "act_shade", "act_thermal", "act_heating",
        "act_circfan", "act_co2", "act_fog"]
INDOOR = ["in_temp", "in_hum", "in_co2"]


def build():
    tX, ty, sX = common.load_raw()
    cols = ["row_id", "farm", "day", "hour", "t"] + ACTS + INDOOR
    a = pd.concat([tX[cols], sX[cols]], ignore_index=True)
    a = a[a.farm.isin(TARGET_FARMS)].sort_values(["farm", "t"]).reset_index(drop=True)
    g = a.groupby(["farm", "day"], sort=False)
    out = pd.DataFrame({"row_id": a.row_id.values})
    # value AT hour 00 only.  groupby.first() would skip a missing hour-0
    # value and pull a LATER hour of the day into earlier rows (a leak).
    h0 = a[a.hour == 0].set_index(["farm", "day"])
    key = pd.MultiIndex.from_arrays([a.farm.values, a.day.values])
    for v in ACTS + INDOOR:
        out[v + "_h0"] = h0[v].reindex(key).values
    for v in ACTS:
        out[v + "_tdm"] = g[v].transform(lambda s: s.expanding().mean()).values
        out[v + "_tdz"] = g[v].transform(
            lambda s: (s == 0).astype(float).where(s.notna()).expanding().mean()).values
    return out


def names(df):
    return [c for c in df.columns if c != "row_id"]
