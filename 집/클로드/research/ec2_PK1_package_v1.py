# -*- coding: utf-8 -*-
"""EC stage-2 PK1: one pre-registered package of two small, consistent gains
(k = 1, fixed before running; 2026-10-02 집 클로드).
  * TabPFN weight 0.2 -> 0.4 (E40, catalog 6.64/6.71: 12/12 A-B-DIAG10 cells
    improved earlier, failed on one seed's p .028 and on EXT12)
  * within-day smoothing a 0.5 -> 0.25 (P1, catalog 6.155: DIAG10/A/B/EXT10
    improved, p .027-.037)
Candidate (per R3 seed s), built on Codex phase-3 OOF without retraining:
  raw_r = shrink^-1(r3_s), raw_t = shrink^-1(S)  (S = (v2_s - .8 r3_s)/.2)
  p = 0.6 raw_r + 0.4 raw_t ; out = 0.75 p + 0.25 * (day running mean of p)
  clipped to the label range.
Judged by the user's rule (all 3 seeds x 5 validators improve + DIAG10
P(worse) < 0.025); the 3-validator (DIAG10/A/B) view is reported as
description only, pending the user's decision in ec2_V1_*.md.
Known limitation: the two parts were each seen before (post-hoc package);
P1 used this same OOF -> a pass here still needs confirmation on Codex's new
TabPFN bag (request K3).
Run:  cd 집/클로드/research && PYTHONPATH="" py -3.12 -u ec2_PK1_package_v1.py"""
import env  # noqa: F401
import numpy as np

from ec2_common import GRP, SEEDS, judge, load
from ec2_P1_shrink_strength_v1 import inv_shrink


def main():
    o = load()
    lo, hi = o.sub_ec.min(), o.sub_ec.max()
    for s in SEEDS:
        o["S_%d" % s] = (o["v2_%d" % s] - 0.8 * o["r3_%d" % s]) / 0.2
        rr = np.concatenate([inv_shrink(g.values) for _, g in o.groupby(GRP, sort=False)["r3_%d" % s]])
        rt = np.concatenate([inv_shrink(g.values) for _, g in o.groupby(GRP, sort=False)["S_%d" % s]])
        o["p_%d" % s] = 0.6 * rr + 0.4 * rt
        cm = o.groupby(GRP, sort=False)["p_%d" % s].transform(lambda x: x.expanding().mean())
        o["pk_%d" % s] = np.clip(0.75 * o["p_%d" % s] + 0.25 * cm, lo, hi)
        # sanity: rebuilding v2 from the same raw members reproduces it
        p2 = 0.8 * rr + 0.2 * rt
        cm2 = o.assign(_p=p2).groupby(GRP, sort=False)["_p"].transform(lambda x: x.expanding().mean())
        print("seed %d: v2 rebuild max err %.2e" % (s, np.max(np.abs(np.clip(0.5 * p2 + 0.5 * cm2, lo, hi) - o["v2_%d" % s]))))
    ok = judge(o, lambda s: "pk_%d" % s, 1, "PK1 TabPFN 0.4 + smoothing 0.25")
    print("\nPK1 decision (user rule):", ok)


if __name__ == "__main__":
    main()
