# -*- coding: utf-8 -*-
"""v5-recipe EC variants on the test-geometry CV (tree count / leaf size / weight)."""
import numpy as np
from common import USABLE, TARGET_FARMS
from model_v2 import get_panel
from make_submission import DOMAIN_MARKS, ET as CUR_ET, E_HUB
from geometry_cv import geometry_folds, score, et_fp, lgb_fp, blend
import features_v2 as F2

panel, tX, ty, sX = get_panel()
panel["midnight"] = (panel.hour == 0).astype(float)
v5 = USABLE + ["day", "hr_sin", "hr_cos", "midnight"]
v_ec = [c for c in F2.view(panel, "sub_ec") if not any(m in c for m in DOMAIN_MARKS)]
lab = panel[(~panel.is_test) & panel.sub_ec.notna()].reset_index(drop=True)
folds = geometry_folds()

print("===== v5 recipe variants (ref: ET300/leaf2 = 0.2898) =====")
for n, leaf in [(300, 2), (100, 2), (100, 4), (100, 8), (100, 1)]:
    r = score(lab, folds, "sub_ec",
              et_fp(v5, dict(n_estimators=n, max_features=1.0, min_samples_leaf=leaf, n_jobs=4)))
    print("  ET %3d trees, leaf %d           %.4f  (std %.3f)" % (n, leaf, *r))
for nm, cols in [("v5 minus midnight", [c for c in v5 if c != "midnight"]),
                 ("v5 minus day", [c for c in v5 if c != "day"]),
                 ("v5 plus day_par", v5 + ["day_par"])]:
    r = score(lab, folds, "sub_ec", et_fp(cols, dict(n_estimators=100, max_features=1.0,
                                                     min_samples_leaf=2, n_jobs=4)))
    print("  %-30s %.4f  (std %.3f)" % (nm, *r))

print("\n===== weighted blends v5 : current =====")
v5f = et_fp(v5, dict(n_estimators=100, max_features=1.0, min_samples_leaf=2, n_jobs=4))
cur = blend(et_fp(v_ec, CUR_ET), lgb_fp(v_ec, E_HUB))
for w in (1.0, 0.8, 0.65):
    def fp(tr, va, t, w=w):
        return w * v5f(tr, va, t) + (1 - w) * cur(tr, va, t)
    r = score(lab, folds, "sub_ec", fp)
    print("  v5 weight %.2f                  %.4f  (std %.3f)" % (w, *r))
