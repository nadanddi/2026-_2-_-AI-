# -*- coding: utf-8 -*-
"""P1b (POST-HOC, descriptive; not a candidate): smoothing strength a = 0 and 0.125
after P1 showed a monotone 'less smoothing is better' trend.  Same code path as P1.
Run:  cd 집/클로드/research && PYTHONPATH="" py -3.12 -u ec2_P1b_a0_posthoc.py"""
import env  # noqa
import numpy as np
from ec2_common import GRP, SEEDS, judge, load
from ec2_P1_shrink_strength_v1 import inv_shrink
o = load(); lo, hi = o.sub_ec.min(), o.sub_ec.max()
for s in SEEDS:
    c = "v2_%d" % s
    raw = np.concatenate([inv_shrink(g.values) for _, g in o.groupby(GRP, sort=False)[c]])
    o["raw_%d" % s] = raw
    cm = o.groupby(GRP, sort=False)["raw_%d" % s].transform(lambda x: x.expanding().mean())
    for a in (0.0, 0.125):
        o["a%.3f_%d" % (a, s)] = np.clip((1 - a) * raw + a * cm, lo, hi)
for a in (0.0, 0.125):
    judge(o, lambda s, a=a: "a%.3f_%d" % (a, s), 1, "P1b posthoc a=%.3f" % a)
