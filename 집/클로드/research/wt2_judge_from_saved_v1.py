# -*- coding: utf-8 -*-
"""Run the pre-registered WT2 judgement (ec3_WT2_pfn_share_with_sg2_v1.py, commit 03d7961) on its saved output
local/ec3_WT2_all.csv (the fold loop had completed; the run stopped before printing).  Judgement code copied verbatim."""
import env  # noqa: F401
import os
import numpy as np, pandas as pd
S = (47, 1414, 6464); W = (.4, .5, .6, .8, 1.0)
O = pd.read_csv(os.path.join(env.LOCAL, "ec3_WT2_all.csv"))
src = open("ec3_WT2_pfn_share_with_sg2_v1.py", encoding="utf-8").read()
exec(src[src.index("r = lambda e:"):])
