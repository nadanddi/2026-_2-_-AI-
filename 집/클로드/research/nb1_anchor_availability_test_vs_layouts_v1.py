# -*- coding: utf-8 -*-
"""NB1 (structure check, 2026-10-05 집 클로드).  Does a validation layout give pass-2 days the
same NEIGHBOUR availability as the real evaluation days?  For each pass-2 query day (validation
days of DIAG10 / DIAG10q / EL1 layouts, and the 60 real test days), with the reference = labelled
training records outside the query set (test: all 400 labelled days), count reference labelled
records of the same farm within +-3 calendar dates (SG2 calendar, full-day query date, same-date
records excluded) - all, and those with label >= 1.0 / >= 1.2.  Inputs only for the test days
(no labels exist); structural description, nothing is fitted on test_X."""
import env  # noqa: F401
import importlib.util, os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.argv = ["x"]
spec = importlib.util.spec_from_file_location("sg2", os.path.join(HERE, "ec3_SG2_reference_knn_level_v1.py"))
sg2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(sg2)
raw, full, lab, lock, signatures, fds = sg2.p3.prepare()
R, WV, hrs, SIG = sg2.prepare_structure()
ec = lab.groupby(["farm", "day"]).sub_ec.mean(); labset = set(ec.index)
roles = R.set_index(["farm", "day"]).role
days = lab[["farm", "day"]].drop_duplicates()
lay = {"DIAG10": [x[2] for x in fds if x[0] == "DIAG10"],
       "DIAG10q": [{(f, int(d)) for f, d in zip(days.farm, days.day) if (d // 9) % 10 == k and (f, int(d)) not in lock} for k in range(10)],
       "EL1": []}
for f in ("F13", "F47"):
    ds = sorted(lab[(lab.farm == f) & (lab.day >= 179)].day.unique())
    for k in range(0, len(ds), 5):
        lay["EL1"].append({(f, int(d)) for d in ds[k:k + 5]})
te = pd.read_csv(os.path.join(env.DATA, "test_X.csv"), usecols=["row_id"])
te = te[te.row_id.str[:3].isin(["F13", "F47"])]
tdays = sorted({(r[:3], int(r[4:7])) for r in te.row_id})
lay["TEST"] = [set(tdays)]


def counts(vd, ref):
    cal = sg2.ref_calendar(R, WV, ref); cache = {}
    alld = {f: sorted(R[R.farm == f].day) for f in ("F13", "F47")}
    def full_date(f, d):
        if (f, d) in cal: return cal[(f, d)]
        if (f, d) in cache: return cache[(f, d)]
        E = [e for e in alld[f] if (f, e) in ref]
        dist = np.sqrt(np.nanmean((WV.loc[[(f, e) for e in E]].values - WV.loc[(f, d)].values) ** 2, axis=1)); ok = dist <= .05
        if ok.any(): t = float(np.mean([cal[(f, e)] for e, o in zip(E, ok) if o]))
        else:
            i = alld[f].index(d); t = (full_date(f, alld[f][i - 1]) + (0.0 if roles.get((f, d)) == "second" else 0.1)) if i else 0.0
        cache[(f, d)] = t; return t
    out = []
    for f, d in sorted(vd):
        if d < 179: continue
        cq = full_date(f, d)
        E = [e for e in alld[f] if (f, e) in ref and (f, e) in ec.index and (f, e) not in lock]
        c = np.array([cal[(f, e)] for e in E]); v = np.array([ec[(f, e)] for e in E])
        m = (np.abs(c - cq) <= 3) & (c != cq)
        out.append(dict(farm=f, day=d, n=int(m.sum()), n10=int((m & (v >= 1)).sum()), n12=int((m & (v >= 1.2)).sum()),
                        y=ec.get((f, d), np.nan)))
    return out


rows = []
for name, sets in lay.items():
    for vd in sets:
        ref = labset - set(vd) if name != "TEST" else labset
        for r in counts(vd, ref):
            r["layout"] = name; rows.append(r)
D = pd.DataFrame(rows)
D.to_csv(os.path.join(env.LOCAL, "nb1_anchor_counts_v1.csv"), index=False)
print("pass-2 query days: neighbours within +-3 dates (same farm, labelled, not same date)")
print(D.groupby("layout").agg(days=("day", "size"), n_mean=("n", "mean"), n_zero=("n", lambda x: (x == 0).mean()),
                              ge2_high10=("n10", lambda x: (x >= 2).mean()), n10_mean=("n10", "mean")).round(2).to_string())
print("\nhigh days only (label >= 1):")
H = D[D.y >= 1]
print(H.groupby("layout").agg(days=("day", "size"), n_mean=("n", "mean"), ge2_high10=("n10", lambda x: (x >= 2).mean())).round(2).to_string())
