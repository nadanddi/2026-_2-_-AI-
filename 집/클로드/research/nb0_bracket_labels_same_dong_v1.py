# -*- coding: utf-8 -*-
"""NB0 (diagnostic, fixed before running; 2026-10-03 집 클로드).
Rules (문제설명서 5): the time restriction applies to INPUTS; public train_y "may be
used for model training" without a before/after restriction (only hidden evaluation
labels are banned).  Evaluation blocks have labelled records on BOTH sides.
Earlier interpolation (6.142) grouped sources by parity, which is wrong (C6.205).
Question: can the day level of a held-out pass-2 day be recovered from the nearest
labelled records of the SAME 동 before and after (gap >= g records from the day,
emulating evaluation blocks), better than R3S?
동: pair role / classifier (local/st_dong_assign_v1.csv).  Labels: train_y day means,
lock days excluded.  For each labelled pass-2 day d (lock excluded) and each g in
(2, 4, 6): previous same-동 labelled record q1 <= d-g, next q2 >= d+g (same pass);
interp = linear in date index; also 'any-동' version.  Compare day-level RMSE vs
R3S DIAG10 OOF day mean, and 0.5/0.5 blend.  Test geometry: for each evaluation day,
date distance to the nearest labelled same-동 record before / after.
Reading: clue if same-동 interpolation or blend beats R3S day level by >= 10% for g=4."""
import env  # noqa: F401
import importlib.util
import json
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.argv = ["x"]
spec = importlib.util.spec_from_file_location("dc5", os.path.join(HERE, "ec2_DC5_r3_season_v1.py"))
dc5 = importlib.util.module_from_spec(spec); spec.loader.exec_module(dc5)
lock = {(z["farm"], int(z["day"])) for z in json.loads(open(dc5.p3.LOCK, encoding="utf-8").read())["selected"]}
D = pd.read_csv(os.path.join(env.LOCAL, "st_dong_assign_v1.csv")).sort_values(["farm", "day"])
D["date"] = D.groupby("farm").role.transform(lambda s: np.cumsum(s.values != "second") - 1)
Y = pd.read_csv(os.path.join(env.DATA, "train_y.csv")); Y = Y[Y.row_id.str[:3].isin(["F13", "F47"])].copy()
Y["farm"], Y["day"] = Y.row_id.str[:3], Y.row_id.str[4:7].astype(int)
EC = Y.groupby(["farm", "day"]).sub_ec.mean()
EC = EC[[k not in lock for k in EC.index]]
O = pd.read_csv(os.path.join(env.LOCAL, "ec2_DC5_oof.csv")); O = O[O.validator == "DIAG10"]
P = O.assign(p=O[["r3s_7", "r3s_101", "r3s_2024"]].mean(axis=1)).groupby(["farm", "day"]).p.mean()
r = lambda e: float(np.sqrt(np.mean(np.square(e))))
for g in (2, 4, 6):
    rows = []
    for f in ("F13", "F47"):
        G = D[D.farm == f].set_index("day")
        lab = [d for d in G.index if (f, d) in EC.index]
        for d in lab:
            if d < 179 or (f, d) not in P.index:
                continue
            for mode in ("same", "any"):
                cand = [q for q in lab if q >= 179 and (mode == "any" or G.loc[q, "dong"] == G.loc[d, "dong"])]
                b = [q for q in cand if q <= d - g]; a = [q for q in cand if q >= d + g]
                if not b or not a:
                    continue
                q1, q2 = max(b), min(a)
                t1, t2, t = G.loc[q1, "date"], G.loc[q2, "date"], G.loc[d, "date"]
                w = 0.5 if t2 == t1 else (t - t1) / (t2 - t1)
                rows.append(dict(farm=f, day=d, mode=mode, y=EC[(f, d)], r3s=P[(f, d)], interp=(1 - w) * EC[(f, q1)] + w * EC[(f, q2)],
                                 gap=(t - t1) + (t2 - t)))
    R = pd.DataFrame(rows)
    for mode, H in R.groupby("mode"):
        print("g=%d %-4s n %2d | day RMSE R3S %.3f | interp %.3f | blend .5 %.3f | mean date span %.1f" % (
            g, mode, len(H), r(H.r3s - H.y), r(H.interp - H.y), r(0.5 * H.r3s + 0.5 * H.interp - H.y), H.gap.mean()))
# test geometry
te = pd.read_csv(os.path.join(env.DATA, "test_X.csv"), usecols=["row_id"]); te = te[te.row_id.str[:3].isin(["F13", "F47"])]
td = pd.DataFrame({"farm": te.row_id.str[:3], "day": te.row_id.str[4:7].astype(int)}).drop_duplicates()
geo = []
for f, d in zip(td.farm, td.day):
    G = D[D.farm == f].set_index("day"); lab = [q for q in G.index if (f, q) in EC.index and q >= 179]
    same = [q for q in lab if G.loc[q, "dong"] == G.loc[d, "dong"]]
    b = [q for q in same if q < d]; a = [q for q in same if q > d]
    geo.append(dict(farm=f, day=d, dong=G.loc[d, "dong"], before=(G.loc[d, "date"] - G.loc[max(b), "date"]) if b else np.nan,
                    after=(G.loc[min(a), "date"] - G.loc[d, "date"]) if a else np.nan))
GE = pd.DataFrame(geo)
print("\nTEST days: date distance to nearest labelled same-dong record (median / max): before %.0f / %.0f, after %.0f / %.0f; missing before %d after %d" % (
    GE.before.median(), GE.before.max(), GE.after.median(), GE.after.max(), GE.before.isna().sum(), GE.after.isna().sum()))
print(GE.assign(span=GE.before + GE.after).span.describe().round(1).to_dict())
