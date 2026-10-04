# -*- coding: utf-8 -*-
"""LB5 (diagnostic, fixed before running; 2026-10-04 집 클로드).  LB4 (6.303 below): the
signature kNN works with the deep_cal calendar (pass-2 rows .268 -> .152, P .0000) but
not with LB2's causal calendar (.218, P .12).  The causal calendar failed on pass-2
records WITHOUT an exact outdoor twin (dates absent from pass 1, e.g. deep_cal's
pass-2-only block), which LB2 carried as previous record + 1.  LB5 changes only that:
no twin -> predecessor = the earlier record (pass 1 or earlier pass 2, same farm) whose
hour-23 outdoor weather best continues into this record's hour 0 (deep_cal_8 cost:
trend-corrected jump of out_temp / out_hum / out_wspd(.3) / out_rad scaled by the
within-day hourly-change SD of this farm), cal = cal(pred) + 0.1.  Uses this record's
inputs and earlier records only (same farm).  Everything else as LB4; clue on cc.
--- LB4 docstring ---  Follows LB1-LB3 (6.301,
6.302): day-to-day causal links reach only ~70 % agreement with CH1 chains and errors
compound along the walk.  LB4 drops the chain walk: each source (동) is run with its own
control settings, so a record's CONTROL SIGNATURE should resemble earlier records of the
same source on nearby calendar dates.
Signature (per record, inputs only, z per farm): night (0-5 h) mean in_temp / in_hum /
in_co2, day (10-15 h) mean in_temp / in_co2, daily max in_temp, hours with thermal
curtain > 0, heating > 0, co2 > 0, shade > 0, vent > 0, first vent-open hour,
night mean circfan.
Query r: candidates = labelled records with day < r.day (earlier in record order),
calendar |cal - cal(r)| <= 3, cal != cal(r) for pass-2 (same date = other source), not
in r's DIAG10 fold, not locked.  Score = signature Euclidean distance / sqrt(dim) +
0.15 x |cal diff|.  Anchor K1 = label of the best, K2 = mean of the best two.
Calendars: CR = deep_cal global calendar (uses future inputs; reference), CC = LB2's
causal calendar (exact outdoor twin among earlier records, else carried +1).
Accuracy reported: best candidate in the same CH1 chain / same 동 (st_dong_assign).
Blend 0.5 R3S + 0.5 anchor; row level keeps R3S shape.
Clue (fixed): a CC variant (K1 or K2) beats R3S on pass-2 rows with cluster bootstrap
P(worse) < .0125 (k = 2) AND pass-2 normal days better.  CR variants are reference."""
import env  # noqa: F401
import json, os
import numpy as np, pandas as pd

LOCK = os.path.join(env.ROOT, u"집", u"코덱스", "analysis", "codex_independent", "ec_final_lock", "locked_days.json")
lock = {(s["farm"], int(s["day"])) for s in json.load(open(LOCK, encoding="utf-8"))["selected"]}
R = pd.read_csv(os.path.join(env.LOCAL, "st_dong_assign_v1.csv")).sort_values(["farm", "day"]).reset_index(drop=True)
R["date"] = R.groupby("farm").role.transform(lambda s: np.cumsum(s.values != "second") - 1)
X = pd.concat([pd.read_csv(os.path.join(env.DATA, f)) for f in ("train_X.csv", "test_X.csv")])
X = X[X.row_id.str[:3].isin(["F13", "F47"])].copy()
X["farm"], X["day"], X["hour"] = X.row_id.str[:3], X.row_id.str[4:7].astype(int), X.row_id.str[8:10].astype(int)

# --- causal calendar (LB2 rule) ---
W = ["out_temp", "out_hum", "out_rad", "out_wspd"]
WV = X.pivot_table(index=["farm", "day"], columns="hour", values=W)
for f in ("F13", "F47"):
    m = WV.index.get_level_values(0) == f; p1 = m & (WV.index.get_level_values(1) < 179)
    for v in W:
        mu, sd = np.nanmean(WV.loc[p1, v].values), np.nanstd(WV.loc[p1, v].values)
        WV.loc[m, v] = (WV.loc[m, v].values - mu) / sd
XO = X.pivot_table(index=["farm", "day"], columns="hour", values=["out_temp", "out_hum", "out_wspd", "out_rad"])
def cont_cost(f, a, b):
    c = 0.0
    for v, w in (("out_temp", 1), ("out_hum", 1), ("out_wspd", .3), ("out_rad", 1)):
        A, B = XO.loc[(f, a), v], XO.loc[(f, b), v]
        pg = .5 * ((A[23] - A[22]) + (B[1] - B[0]))
        e = (((B[0] - A[23]) - pg) / SCV[(f, v)]) ** 2
        c += w * (50 if np.isnan(e) else e)
    return c
SCV = {}
for f in ("F13", "F47"):
    for v in ("out_temp", "out_hum", "out_wspd", "out_rad"):
        SCV[(f, v)] = np.nanstd(np.diff(XO.loc[f][v].values, axis=1))
CAL = {}; nk = {"twin": 0, "pred": 0}
for f in ("F13", "F47"):
    G = R[R.farm == f].sort_values("day")
    for r in G.itertuples():
        if r.day < 179:
            CAL[(f, r.day)] = float(r.date); continue
        E = [d for d in G.day if d < r.day and (f, d) in WV.index]
        dist = np.sqrt(np.nanmean((WV.loc[[(f, d) for d in E]].values - WV.loc[(f, r.day)].values) ** 2, axis=1))
        if (dist <= .05).any():
            CAL[(f, r.day)] = float(np.mean([CAL[(f, d)] for d, o in zip(E, dist <= .05) if o])); nk["twin"] += 1
        else:
            cs = [cont_cost(f, d, r.day) for d in E]
            CAL[(f, r.day)] = CAL[(f, E[int(np.argmin(cs))])] + 0.1; nk["pred"] += 1
print("pass-2 calendar:", nk)
R["cal_c"] = [CAL[(f, d)] for f, d in zip(R.farm, R.day)]
CH = pd.read_csv(os.path.join(env.LOCAL, "deep_cal_11_days.csv"))[["farm", "day", "cal", "chain"]]
R = R.merge(CH.rename(columns={"cal": "cal_r"}), on=["farm", "day"], how="left")

# --- signatures ---
def sig(g):
    g = g.sort_values("hour"); n = g[g.hour <= 5]; d = g[(g.hour >= 10) & (g.hour <= 15)]
    v = g.act_vent.fillna(0).values
    return pd.Series(dict(
        n_t=n.in_temp.mean(), n_h=n.in_hum.mean(), n_c=n.in_co2.mean(), d_t=d.in_temp.mean(), d_c=d.in_co2.mean(),
        mx_t=g.in_temp.max(), th=(g.act_thermal > 0).sum(), he=(g.act_heating > 0).sum(), co=(g.act_co2 > 0).sum(),
        sh=(g.act_shade > 0).sum(), ve=(v > 0).sum(), fo=(g.hour[v > 0].min() if (v > 0).any() else 24), cf=n.act_circfan.mean()))
SG = X.groupby(["farm", "day"]).apply(sig)
for f in ("F13", "F47"):
    m = SG.index.get_level_values(0) == f
    SG.loc[m] = ((SG.loc[m] - SG.loc[m].mean()) / SG.loc[m].std()).values
SG = SG.fillna(0)

Y = pd.read_csv(os.path.join(env.DATA, "train_y.csv")); Y = Y[Y.row_id.str[:3].isin(["F13", "F47"])].copy()
Y["farm"], Y["day"] = Y.row_id.str[:3], Y.row_id.str[4:7].astype(int)
ec = Y.groupby(["farm", "day"]).sub_ec.mean(); ec = ec[[k not in lock for k in ec.index]]
O = pd.read_csv(os.path.join(env.LOCAL, "ec2_DC5_oof.csv")); O = O[O.validator == "DIAG10"].copy()
O["p"] = O[["r3s_7", "r3s_101", "r3s_2024"]].mean(axis=1)
D = O.groupby(["farm", "day"]).agg(fold=("validation_fold", "first"), pm=("p", "mean"), y=("sub_ec", "mean")).reset_index()
fold_of = dict(zip(zip(D.farm, D.day), D.fold))
RI = R.set_index(["farm", "day"])
rows = []
for r in D.itertuples(index=False):
    k = (r.farm, r.day); out = {}
    for cc, calcol in (("cc", "cal_c"), ("cr", "cal_r")):
        c0 = RI.loc[k, calcol]
        G = R[(R.farm == r.farm) & (R.day < r.day)].copy()
        G = G[(G[calcol] - c0).abs() <= 3]
        if r.day >= 179:
            G = G[G[calcol] != c0]
        G = G[[(f, d) in ec.index and fold_of.get((f, d), -1) != r.fold for f, d in zip(G.farm, G.day)]]
        if not len(G):
            out.update({cc + "_k1": np.nan, cc + "_k2": np.nan, cc + "_chain": np.nan, cc + "_dong": np.nan}); continue
        dist = np.sqrt(((SG.loc[list(zip(G.farm, G.day))].values - SG.loc[k].values) ** 2).mean(axis=1)) + 0.15 * (G[calcol] - c0).abs().values
        o = np.argsort(dist); b = G.iloc[o[0]]
        out[cc + "_k1"] = ec[(b.farm, b.day)]
        out[cc + "_k2"] = np.mean([ec[(G.iloc[i].farm, G.iloc[i].day)] for i in o[:2]])
        out[cc + "_chain"] = float(b.chain == RI.loc[k, "chain"]); out[cc + "_dong"] = float(b.dong == RI.loc[k, "dong"])
    rows.append(out)
D = pd.concat([D, pd.DataFrame(rows)], axis=1); D["p2"] = D.day >= 179
P = ["cc_k1", "cc_k2", "cr_k1", "cr_k2"]
for c in P:
    D["b_" + c] = 0.5 * D.pm + 0.5 * D[c].fillna(D.pm)
r_ = lambda e: float(np.sqrt(np.mean(np.square(e))))
for cc in ("cc", "cr"):
    print("%s: anchor %d/%d (pass-2 %d/46); best candidate same CH1 chain %.2f (pass-2 %.2f), same 동 %.2f (pass-2 %.2f)" % (
        cc, D[cc + "_k1"].notna().sum(), len(D), D[D.p2][cc + "_k1"].notna().sum(),
        D[cc + "_chain"].mean(), D[D.p2][cc + "_chain"].mean(), D[cc + "_dong"].mean(), D[D.p2][cc + "_dong"].mean()))
cols = ["pm"] + P + ["b_" + c for c in P]
print("\nday-level RMSE")
print("%-14s %4s " % ("segment", "n") + " ".join("%8s" % c for c in cols))
for nm, m in (("all", D.y.notna()), ("pass-2", D.p2), ("pass-2 normal", D.p2 & (D.y < 1)), ("pass-2 high", D.p2 & (D.y >= 1)), ("pass-1 normal", ~D.p2 & (D.y < 1))):
    g = D[m]
    print("%-14s %4d " % (nm, len(g)) + " ".join("%8.4f" % r_(g[c].fillna(g.pm) - g.y) for c in cols))
O2 = O.merge(D[["farm", "day", "pm", "p2"] + ["b_" + c for c in P]], on=["farm", "day"])
O2["cl"] = O2.farm + "_" + (O2.day // 5).astype(str)
O2["nrm"] = O2.groupby(["farm", "day"]).sub_ec.transform("mean") < 1
rng = np.random.default_rng(20261004); res = {}
print("\nrow-level RMSE (R3S shape + blended level)")
for nm, m in (("all", O2.sub_ec.notna()), ("pass-2", O2.p2), ("pass-2 normal", O2.p2 & O2.nrm)):
    g = O2[m]; e0 = g.p - g.sub_ec; line = "  %-14s R3S %.4f" % (nm, r_(e0))
    for c in P:
        e1 = g.p - g.pm + g["b_" + c] - g.sub_ec; line += "  %s %.4f" % (c, r_(e1))
        if nm == "pass-2":
            dd = (e1 ** 2 - e0 ** 2).groupby(g.cl).agg(["sum", "count"]); s_, n_ = dd["sum"].values, dd["count"].values
            idx = rng.integers(0, len(s_), (20000, len(s_)))
            res[c] = (float(((s_[idx].sum(1) / n_[idx].sum(1)) >= 0).mean()), r_(e1) < r_(e0))
        if nm == "pass-2 normal":
            res[c] = res[c] + (r_(e1) < r_(e0),)
    print(line)
print("\npass-2 P(worse): " + "  ".join("%s %.4f" % (c, v[0]) for c, v in res.items()))
clue = any(res[c][0] < .0125 and res[c][1] and res[c][2] for c in ("cc_k1", "cc_k2"))
D.to_csv(os.path.join(env.LOCAL, "lb5_daytable_v1.csv"), index=False)
print("\nLB5 clue (causal calendar, signature kNN, previous labels):", clue)
