# -*- coding: utf-8 -*-
"""Forensic 27: daily EC values in train days bracketing each test block (late period)."""
import env
import numpy as np, pandas as pd, common
tX, ty, sX = common.load_raw()
for f in ["F13","F47"]:
    y=ty[ty.farm==f].groupby("day").sub_ec.agg(["mean","max"]).round(2)
    print(f, y[y.index>=175]["mean"].to_dict())
