# -*- coding: utf-8 -*-
"""DC6-check (diagnostic; 2026-10-03 집 클로드).  SE0 side finding: DC4 anchors a
pass-2 training day to the mean record day of exact-twin pass-1 days of BOTH farms,
but F13/F47 record days of one date differ by -8..+8 (C6.205).  How much would the
season index change with SAME-FARM anchors?  Full training set (all labelled
non-lock days), training and test days."""
import env  # noqa: F401
import importlib.util
import os
import sys
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

HERE = os.path.dirname(os.path.abspath(__file__))
sys.argv = ["x"]
spec = importlib.util.spec_from_file_location("dc5", os.path.join(HERE, "ec2_DC5_r3_season_v1.py"))
dc5 = importlib.util.module_from_spec(spec); spec.loader.exec_module(dc5)
dc4, p3 = dc5.dc4, dc5.p3
spec0 = importlib.util.spec_from_file_location("se0", os.path.join(HERE, "se0_season_position_error_v1.py"))
se0 = importlib.util.module_from_spec(spec0); spec0.loader.exec_module(se0)
raw, full, lab, lock, signatures, fds = p3.prepare()
wv = dc4.weather_vectors(full)
lk = {(f, d + j) for f, d in lock for j in (-1, 0, 1)}
tdays = lab[[(f, d) not in lk for f, d in zip(lab.farm, lab.day)]][["farm", "day"]].drop_duplicates()
te = pd.read_csv(os.path.join(env.DATA, "test_X.csv"), usecols=["row_id"]); te = te[te.row_id.str[:3].isin(["F13", "F47"])]
q = pd.DataFrame({"farm": te.row_id.str[:3], "day": te.row_id.str[4:7].astype(int)}).drop_duplicates().reset_index(drop=True)
s_old, q_old = dc4.season_index(tdays, q, wv)
p1, p2 = tdays[tdays.day < 179], tdays[tdays.day >= 179]
ZZ = se0.zscaled(wv, p1)
new_q = np.full(len(q), np.nan); s_new = {}
for f in ("F13", "F47"):
    P1 = p1[p1.farm == f]
    A = ZZ.reindex(list(zip(P1.farm, P1.day))).values
    ax, ay = [], []
    for d in sorted(p2[p2.farm == f].day):
        dist = np.sqrt(np.nanmean((A - ZZ.reindex([(f, d)]).values) ** 2, axis=1))
        if np.nanmin(dist) <= .05:
            ax.append(d); ay.append(float(P1.day.values[dist <= .05].mean()))
    sm = IsotonicRegression(increasing=True).fit(ax, ay).predict(ax)
    for d in p2[p2.farm == f].day:
        s_new[(f, d)] = float(np.interp(d, ax, sm))
    m = (q.farm == f) & (q.day >= 179)
    new_q[m.values] = np.interp(q.day[m], ax, sm)
    print("%s same-farm anchors %d" % (f, len(ax)))
dt = pd.Series({k: s_new[k] - s_old[k] for k in s_new})
dq = pd.Series(new_q - q_old)[q.day.values >= 179]
print("pass-2 training days: |new-old| median %.1f, max %.1f, n>5: %d of %d" % (dt.abs().median(), dt.abs().max(), (dt.abs() > 5).sum(), len(dt)))
print("test days: |new-old| median %.1f, max %.1f, n>5: %d of %d" % (dq.abs().median(), dq.abs().max(), (dq.abs() > 5).sum(), len(dq)))
print(pd.DataFrame({"farm": q.farm, "day": q.day, "old": q_old, "new": new_q}).round(1).to_string(index=False))
