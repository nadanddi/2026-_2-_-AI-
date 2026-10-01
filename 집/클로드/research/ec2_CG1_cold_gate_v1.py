# -*- coding: utf-8 -*-
"""EC stage-2 CG1: remove TabPFN on cold rows with the TEMPERATURE G_C gate form
(catalog 6.67, not tuned on EC), k = 1, fixed before running (2026-10-02 집 클로드).
Motivation (descriptive, ec2_late_member_check_posthoc.log): on DIAG10 late days
(>= 179, the test period) TabPFN alone .449 vs R3 .375 and v2 .383 > R3; in
temperature TabPFN also failed out of range in the cold and the gate fixed it.
Candidate per seed: g = clip((in_temp - 8)/2, 0, 1) (missing in_temp -> g = 1),
out = (1 - 0.2 g) r3_s + 0.2 g S_s on the shrunk members (S_s = TabPFN part).
Rule: user rule (3 seeds x 5 validators improve + DIAG10 P(worse) < .025).
Also reported: late / early / cold-row RMSE.
Run:  cd 집/클로드/research && PYTHONPATH="" py -3.12 -u ec2_CG1_cold_gate_v1.py"""
import env  # noqa: F401
import os
import numpy as np
import pandas as pd
from ec2_common import SEEDS, judge, load, rmse

def main():
    o = load()
    t = pd.read_csv(os.path.join(env.DATA, "train_X.csv"), usecols=["row_id", "in_temp"])
    o = o.merge(t, on="row_id", how="left")
    g = np.clip((o.in_temp.fillna(99) - 8) / 2, 0, 1)
    for s in SEEDS:
        S = (o["v2_%d" % s] - 0.8 * o["r3_%d" % s]) / 0.2
        o["cg_%d" % s] = (1 - 0.2 * g) * o["r3_%d" % s] + 0.2 * g * S
    print("rows with g<1: %.1f%% (DIAG10 late %.1f%%)" % (100 * (g < 1).mean(), 100 * (g[(o.validator == "DIAG10") & (o.day >= 179)] < 1).mean()))
    ok = judge(o, lambda s: "cg_%d" % s, 1, "CG1 EC TabPFN cold gate (temperature form)")
    D = o[o.validator == "DIAG10"]
    c = D.in_temp < 10
    print("  DIAG10 in_temp<10 rows %d: v2 %.4f -> %.4f" % (c.sum(), rmse(D.v2_7[c] - D.sub_ec[c]), rmse(D.cg_7[c] - D.sub_ec[c])))
    print("\nCG1 decision:", ok)

if __name__ == "__main__":
    main()
