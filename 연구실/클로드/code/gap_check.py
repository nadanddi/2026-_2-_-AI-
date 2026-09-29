# -*- coding: utf-8 -*-
"""Why was the late-period CV pessimistic?  Compare label-gap geometry."""
import numpy as np
from common import load_raw, make_folds, split_mask, TARGET_FARMS

tX, ty, sX = load_raw()
lab = ty[ty.row_id.str[:3].isin(TARGET_FARMS)].copy()


def gap_stats(val_days_by_farm, train_days_by_farm):
    g = []
    for f, vd in val_days_by_farm.items():
        tr = np.array(sorted(train_days_by_farm[f]))
        for d in vd:
            g.append(np.abs(tr - d).min())
    g = np.array(g)
    return g.mean(), np.median(g), g.max()


tr_days = {f: set(lab[lab.farm == f].day) for f in TARGET_FARMS}
te_days = {f: sorted(sX[sX.farm == f].day.unique()) for f in TARGET_FARMS}
print("distance (days) from a validation day to the nearest LABELLED training day")
print("  real test blocks               mean %.1f  median %.0f  max %d" % gap_stats(te_days, tr_days))

days = {f: sorted(tr_days[f]) for f in TARGET_FARMS}
late_g, early_g = [], []
for fs in range(5):
    for fd in make_folds(days, n_folds=6, seed=fs):
        trm, vam = split_mask(lab, fd)
        trd = {f: set(lab[trm & (lab.farm == f).values].day) for f in TARGET_FARMS}
        for f, vd in fd.items():
            tr = np.array(sorted(trd[f]))
            for d in vd:
                (late_g if d >= 183 else early_g).append(np.abs(tr - d).min())
late_g, early_g = np.array(late_g), np.array(early_g)
print("  my CV, late (>=183) held-out   mean %.1f  median %.0f  max %d"
      % (late_g.mean(), np.median(late_g), late_g.max()))
print("  my CV, early (<183) held-out   mean %.1f  median %.0f  max %d"
      % (early_g.mean(), np.median(early_g), early_g.max()))
print()
allT, allE = tr_days["F13"], set(te_days["F13"])
print("F13 layout, day 183-245  (T=train  E=test  .=absent):")
print("  " + "".join("T" if d in allT else ("E" if d in allE else ".") for d in range(183, 246)))
print("  when a late T-run (e.g. 191-198) is held out, its neighbours are E blocks with")
print("  NO labels, so the nearest label is 8-12 days away; a real E block has labels 2 days away.")
