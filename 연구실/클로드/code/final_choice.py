# -*- coding: utf-8 -*-
"""Last decision: keep or drop the literature-derived feature block.

The ablation showed it contributes nothing measurable (late delta +0.0005 for
sub_temp, +0.0047 for sub_ec, both inside the noise band).  When two options
score the same, the smaller feature set is the right default under the
overfitting pressure this problem has, so this re-runs the ACTUAL final
ensembles -- not the single-model probes the ablation used -- on both views.
"""
import numpy as np

from common import TARGET_FARMS
from model_v2 import get_panel
from model_v4 import T_HUB, E_HUB, MODEL_SEEDS, blend_multi
from ablation import is_domain
import features_v2 as F2

FOLD_SEEDS = [0, 1, 2, 3, 4, 5, 6]


def main():
    panel, tX, ty, sX = get_panel()
    days = {f: sorted(panel[(panel.farm == f) & (~panel.is_test)].day.unique())
            for f in TARGET_FARMS}

    vt = F2.view(panel, "sub_temp")
    ve = F2.view(panel, "sub_ec")
    vt_s = [c for c in vt if not is_domain(c)]
    ve_s = [c for c in ve if not is_domain(c)]

    print("=== sub_temp final ensemble: huber x3 seeds (%d partitions) ==="
          % len(FOLD_SEEDS))
    for nm, fc in [("full view  (%d)" % len(vt), vt),
                   ("slim view  (%d)" % len(vt_s), vt_s)]:
        r = blend_multi(panel, days, "sub_temp",
                        [dict(fcols=fc, params=T_HUB, seeds=MODEL_SEEDS)],
                        fold_seeds=FOLD_SEEDS)
        print("  %-20s all %.4f | late %.4f +-%.3f" % ((nm,) + r))

    print("\n=== sub_ec final ensemble: blend(huber, huber+recency120) ===")
    for nm, fc in [("full view  (%d)" % len(ve), ve),
                   ("slim view  (%d)" % len(ve_s), ve_s)]:
        r = blend_multi(panel, days, "sub_ec",
                        [dict(fcols=fc, params=E_HUB, seeds=MODEL_SEEDS),
                         dict(fcols=fc, params=E_HUB, seeds=MODEL_SEEDS,
                              recency_tau=120)],
                        fold_seeds=FOLD_SEEDS)
        print("  %-20s all %.4f | late %.4f +-%.3f" % ((nm,) + r))


if __name__ == "__main__":
    main()
