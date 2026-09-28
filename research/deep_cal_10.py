# -*- coding: utf-8 -*-
"""Q2: link days of consecutive calendar dates within a record into source chains (analysis)."""
import env  # noqa
import numpy as np, pandas as pd, itertools
import common
from scipy.optimize import linear_sum_assignment
tX, ty, sX = common.load_raw()
k = pd.read_csv(env.LOCAL + "/deep_cal_9_days.csv")
a = pd.concat([tX, sX], ignore_index=True)
a = a[a.farm.isin(["F13", "F47"])].merge(ty[["row_id", "sub_temp", "sub_ec"]], on="row_id", how="left")
a = a.sort_values(["farm", "day", "hour"])
IN = ["in_temp", "in_hum", "in_co2"]
ACT = ["act_vent", "act_shade", "act_thermal", "act_heating", "act_circfan", "act_co2", "act_fog"]
rec = {}
for (f, d), g in a.groupby(["farm", "day"]):
    g = g.set_index("hour")
    r = dict(farm=f, day=d)
    for v in IN + ["sub_temp", "sub_ec"]:
        for h in (0, 1, 22, 23):
            r["%s_%d" % (v, h)] = g[v].get(h, np.nan)
    for v in ACT:
        r[v + "_h0"] = g[v].get(0, np.nan); r[v + "_h23"] = g[v].get(23, np.nan); r[v + "_m"] = g[v].mean()
    r["ec_m"] = g.sub_ec.mean(); r["st_m"] = g.sub_temp.mean()
    rec[(f, d)] = r
S = pd.DataFrame(rec.values()).merge(k[["farm", "day", "cal", "is_test"]], on=["farm", "day"])
S.to_csv(env.LOCAL + "/deep_cal_10_daysum.csv", index=False)
sd = {"in_temp": 1.0, "in_hum": 5.0, "in_co2": 40.0}
def cont(x, y, use=IN):
    c = 0
    for v in use:
        g = y["%s_0" % v] - x["%s_23" % v]
        p = 0.5 * ((x["%s_23" % v] - x["%s_22" % v]) + (y["%s_1" % v] - y["%s_0" % v]))
        e = ((g - p) / sd[v]) ** 2
        c += 0 if np.isnan(e) else e
    return c
def act_cost(x, y):
    # heating/curtain states rarely change at midnight within a source
    return sum(abs(y[v + "_h0"] - x[v + "_h23"]) / 50 for v in ["act_heating", "act_thermal", "act_circfan", "act_vent"] if not np.isnan(y[v + "_h0"] - x[v + "_h23"]))
links = []
for f in ["F13", "F47"]:
    Sf = S[S.farm == f]
    byc = {c: g.to_dict("records") for c, g in Sf.groupby("cal")}
    for c in sorted(byc):
        if c + 1 not in byc:
            continue
        A, B = byc[c], byc[c + 1]
        M = np.array([[cont(x, y) + act_cost(x, y) for y in B] for x in A])
        r_, c_ = linear_sum_assignment(M)
        for i, j in zip(r_, c_):
            alt = np.delete(M[i], j).min() if M.shape[1] > 1 else np.nan
            x, y = A[i], B[j]
            links.append(dict(farm=f, cal=c, d_from=x["day"], d_to=y["day"], cost=M[i, j], alt=alt,
                              n_from=len(A), n_to=len(B),
                              ec_jump=y["sub_ec_0"] - x["sub_ec_23"], st_jump=y["sub_temp_0"] - x["sub_temp_23"],
                              ec_jump_alt=(np.nan if M.shape[1] < 2 else B[1 - j]["sub_ec_0"] - x["sub_ec_23"]) if M.shape[1] == 2 else np.nan,
                              st_jump_alt=(B[1 - j]["sub_temp_0"] - x["sub_temp_23"]) if M.shape[1] == 2 else np.nan))
L = pd.DataFrame(links)
L["gap"] = L.d_to - L.d_from
L.to_csv(env.LOCAL + "/deep_cal_10_links.csv", index=False)
two = L[(L.n_from == 2) & (L.n_to == 2)]
print("links total", len(L), " 2x2 transitions links", len(two))
print("decisive share (alt cost > 2x chosen): %.2f" % (two.alt > 2 * two.cost + 0.5).mean())
ok = two.dropna(subset=["ec_jump", "ec_jump_alt"])
print("label check on 2x2: |EC jump| chosen median %.3f vs alt %.3f; chosen smaller in %.2f"
      % (ok.ec_jump.abs().median(), ok.ec_jump_alt.abs().median(), (ok.ec_jump.abs() < ok.ec_jump_alt.abs()).mean()))
print("                 |subT jump| chosen median %.3f vs alt %.3f; chosen smaller in %.2f"
      % (ok.st_jump.abs().median(), ok.st_jump_alt.abs().median(), (ok.st_jump.abs() < ok.st_jump_alt.abs()).mean()))
print("all links |EC jump| median %.3f  (reference: same-day hour-to-hour |dEC| ~ see below)" % L.ec_jump.abs().median())
print("gap (day index of same-source next date) distribution:")
print(L.gap.value_counts().sort_index().to_string())
