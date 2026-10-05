# -*- coding: utf-8 -*-
"""NB2 (validator design from STRUCTURE only, 2026-10-05 집 클로드).  Layout B for EC = folds made of
pass-2 labelled days only (pass 1 never validated), consecutive blocks of L records per farm.
Choose L so that the validation days' neighbour availability (+-3 dates, same farm, labelled
reference outside the fold; NB1 count) matches the 60 real evaluation days (mean 11.67, share with
>= 2 EC>=1 neighbours .48).  No model, no prediction, no score is looked at.
Rule (fixed): the smallest L in (1, 2, 3, 5) whose mean count is within 1.0 of 11.67."""
import env  # noqa: F401
import importlib.util, os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.argv = ["x"]
spec = importlib.util.spec_from_file_location("nb1", os.path.join(HERE, "nb1_anchor_availability_test_vs_layouts_v1.py"))
src = open(os.path.join(HERE, "nb1_anchor_availability_test_vs_layouts_v1.py"), encoding="utf-8").read()
src = src[:src.index("rows = []")]
exec(compile(src, "nb1", "exec"))
res = []
for L in (1, 2, 3, 5):
    sets = []
    for f in ("F13", "F47"):
        ds = sorted(lab[(lab.farm == f) & (lab.day >= 179)].day.unique())
        for k in range(0, len(ds), L):
            sets.append({(f, int(d)) for d in ds[k:k + L]})
    C = []
    for vd in sets:
        C += counts(vd, labset - set(vd))
    C = pd.DataFrame(C)
    res.append((L, len(sets), C.n.mean(), (C.n10 >= 2).mean()))
    print("L=%d folds %d: neighbours mean %.2f, share >=2 high neighbours %.2f" % res[-1])
print("target (real evaluation days): 11.67, .48")
ok = [r for r in res if abs(r[2] - 11.67) <= 1.0]
print("chosen L:", ok[0][0] if ok else "none within 1.0 -> take closest: %d" % min(res, key=lambda r: abs(r[2] - 11.67))[0])
