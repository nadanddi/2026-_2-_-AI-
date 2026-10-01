# -*- coding: utf-8 -*-
"""EC stage-2 P1: within-day smoothing strength.  out = (1-a) p + a m, a in
{0.25, 0.75, 1.0} vs current a = 0.5 (k = 3).  Rules fixed in
ec2_P1_ST1_사전고정_2026-10-02.md before running.  2026-10-02 집 클로드.
Run:  cd 집/클로드/research && PYTHONPATH="" py -3.12 -u ec2_P1_shrink_strength_v1.py"""
import env  # noqa: F401
import numpy as np

from ec2_common import GRP, SEEDS, judge, load


def inv_shrink(out):
    p = np.empty(len(out)); s = 0.0
    for h, o in enumerate(out):
        n = h + 1
        p[h] = (o - 0.5 * s / n) / (0.5 + 0.5 / n)
        s += p[h]
    return p


def main():
    o = load()
    lo, hi = o.sub_ec.min(), o.sub_ec.max()
    for s in SEEDS:
        c = "v2_%d" % s
        edge = (np.abs(o[c] - lo) < 1e-9) | (np.abs(o[c] - hi) < 1e-9)
        raw = np.concatenate([inv_shrink(g.values) for _, g in o.groupby(GRP, sort=False)[c]])
        o["raw_%d" % s] = raw
        cm = o.groupby(GRP, sort=False)["raw_%d" % s].transform(lambda x: x.expanding().mean())
        chk = np.max(np.abs((0.5 * o["raw_%d" % s] + 0.5 * cm) - o[c]))
        print("seed %d: clip-bound rows %d, reconstruction max err %.2e" % (s, edge.sum(), chk))
        for a in (0.25, 0.75, 1.0):
            o["a%.2f_%d" % (a, s)] = np.where(edge, o[c], np.clip((1 - a) * o["raw_%d" % s] + a * cm, lo, hi))
    res = {a: judge(o, lambda s, a=a: "a%.2f_%d" % (a, s), 3, "P1 a=%.2f" % a) for a in (0.25, 0.75, 1.0)}
    print("\nP1 decision:", res)


if __name__ == "__main__":
    main()
