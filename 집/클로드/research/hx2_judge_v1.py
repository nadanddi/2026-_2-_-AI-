# -*- coding: utf-8 -*-
"""HX2 judge: re-runs hx2_unexplained_highec_confirm_v1.summarize() on the saved checkpoints with ONE fix - the
per-day grouping in the sensitivity step crashed (pandas read a list of (farm, day) tuples as group keys).  The rule,
sets, thresholds and data are unchanged (imported from the committed script).  2026-10-09 집 클로드.
Run:  cd 집/클로드/research && PYTHONPATH="" py -3.12 -u hx2_judge_v1.py
"""
import env  # noqa: F401
import importlib.util, os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.argv = ["x", "sum"]
spec = importlib.util.spec_from_file_location("hx2", os.path.join(HERE, "hx2_unexplained_highec_confirm_v1.py"))
hx2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(hx2)

_orig_groupby = pd.Series.groupby


def _groupby(self, by=None, *a, **k):
    if isinstance(by, list) and by and isinstance(by[0], tuple):
        by = pd.MultiIndex.from_tuples(by)            # the only change: tuples -> MultiIndex keys
    return _orig_groupby(self, by, *a, **k)


pd.Series.groupby = _groupby
hx2.summarize()
