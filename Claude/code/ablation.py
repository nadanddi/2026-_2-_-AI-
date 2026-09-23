# -*- coding: utf-8 -*-
"""Did the literature-derived features actually earn their place?

The domain block went in together with the feature-set rewrite, so its effect
was never measured on its own.  This removes it and re-scores, so the claim
"the research helped" is checked instead of asserted.

Domain block = radiation reaching the slab (rad_eff, reconstructing the absent
in_rad), Baille simplified Penman-Monteith transpiration (transp_pm), the
season-long salt-accumulation integrals, the root-zone degree-hour and heating
terms, and the RTR steering ratio.
"""
import numpy as np

from common import TARGET_FARMS
from model_v2 import get_panel
from model_v3 import multi
from model_v4 import T_HUB, E_HUB, MODEL_SEEDS, FOLD_SEEDS, blend_multi
import features_v2 as F2

DOMAIN_MARKS = ("rad_eff", "transp_pm", "root_dh", "heat_input", "rtr_",
                "transp_per_rad", "_cum", "screen_ins", "heat_screen")


def is_domain(c):
    return any(m in c for m in DOMAIN_MARKS)


def main():
    panel, tX, ty, sX = get_panel()
    days = {f: sorted(panel[(panel.farm == f) & (~panel.is_test)].day.unique())
            for f in TARGET_FARMS}

    for target, params in [("sub_temp", T_HUB), ("sub_ec", E_HUB)]:
        full = F2.view(panel, target)
        plain = [c for c in full if not is_domain(c)]
        dom = [c for c in full if is_domain(c)]
        print("\n===== %s : %d features, %d of them domain-derived ====="
              % (target, len(full), len(dom)))
        r_full = blend_multi(panel, days, target,
                             [dict(fcols=full, params=params,
                                   seeds=MODEL_SEEDS)])
        r_plain = blend_multi(panel, days, target,
                              [dict(fcols=plain, params=params,
                                    seeds=MODEL_SEEDS)])
        print("  with domain block     all %.4f | late %.4f +-%.3f" % r_full)
        print("  without domain block  all %.4f | late %.4f +-%.3f" % r_plain)
        print("  -> late delta %+.4f (negative = domain features help)"
              % (r_full[1] - r_plain[1]))
        print("  domain columns: %s" % ", ".join(sorted(dom)[:8]) + " ...")


if __name__ == "__main__":
    main()
