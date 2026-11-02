# -*- coding: utf-8 -*-
"""Feature blocks added for submission 3, in one self-contained module.

Background (research/데이터_단서_카탈로그.md): F13 and F47 are day-level
splices of several source greenhouses.  The source changes at midnight, the
slab follows the air with a ~3 h lag, and the test period is colder than the
labelled data.  Three blocks answer that:

  seg   day-restarted exponential filters of in_temp, effective radiation,
        heating and outside temperature (hl 1/2/4/8 h), plus the hour-00
        in_temp and the deviation from it.  Memory that does not leak across
        a source switch.
  phys  continuous first-order filters feeding a LINEAR baseline for the
        residual-target temperature model.  Linear, so it extrapolates into
        the colder test period where trees floor out.
  fp    source fingerprint: hour-00 actuator/indoor values and same-day
        expanding means / zero shares of the actuators (fp_features.py).

Every column uses only inputs of the same greenhouse at the row's hour or
earlier: day-restarted filters restart at 00 of the row's own day, continuous
filters are causal EWMs, and hour-00 values come from hour 00 only.
"""
import numpy as np
import pandas as pd

import common
from common import TARGET_FARMS
import fp_features as FP

PHYS_SPEC = [("in_temp", None), ("in_temp", 1), ("in_temp", 3), ("in_temp", 8),
             ("in_temp", 24), ("in_temp", 72), ("rad", None), ("rad", 2), ("rad", 6),
             ("heat", 1), ("heat", 4), ("out_temp", 6), ("out_temp", 48)]


def _series():
    tX, ty, sX = common.load_raw()
    cols = ["row_id", "farm", "t", "in_temp", "out_temp", "out_rad",
            "act_shade", "act_thermal", "act_heating"]
    a = pd.concat([tX[cols], sX[cols]], ignore_index=True)
    a = a[a.farm.isin(TARGET_FARMS)]
    out = {}
    for f, g in a.groupby("farm"):
        g = g.sort_values("t").set_index("t")
        g = g.reindex(np.arange(g.index.min(), g.index.max() + 1))
        g["farm"] = f
        out[f] = g
    return out


def _sources(g):
    rad = g.out_rad * (g.act_shade.fillna(100) / 100) * (g.act_thermal.fillna(100) / 100)
    return {"in_temp": g.in_temp, "rad": rad, "heat": g.act_heating, "out_temp": g.out_temp}


def seg_features():
    parts = []
    for f, g in _series().items():
        day = g.index // 24
        src = _sources(g)
        d = pd.DataFrame({"row_id": g.row_id.values}, index=g.index)
        for c, s in src.items():
            for hl in (1, 2, 4, 8):
                d["seg_%s_%d" % (c, hl)] = s.groupby(day).transform(
                    lambda x, hl=hl: x.ewm(halflife=hl, ignore_na=True).mean()).values
        first = g.in_temp.where(g.index % 24 == 0).groupby(day).transform("max")
        d["seg_in_temp_h0"] = first.values
        d["seg_in_temp_dev_h0"] = g.in_temp.values - first.values
        parts.append(d.dropna(subset=["row_id"]))
    return pd.concat(parts, ignore_index=True)


def phys_features():
    parts = []
    for f, g in _series().items():
        src = _sources(g)
        d = pd.DataFrame({"row_id": g.row_id.values})
        for c, hl in PHYS_SPEC:
            s = src[c]
            d["ph_%s_%s" % (c, hl)] = (s if hl is None else s.ewm(halflife=hl, ignore_na=True).mean()).values
        d["ph_farm_id"] = float(f == "F47")
        parts.append(d.dropna(subset=["row_id"]))
    return pd.concat(parts, ignore_index=True)


def fp_features():
    return FP.build()


def names(df):
    return [c for c in df.columns if c != "row_id"]
