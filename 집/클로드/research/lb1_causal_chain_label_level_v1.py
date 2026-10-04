# -*- coding: utf-8 -*-
"""LB1 (diagnostic, fixed before running; 2026-10-04 집 클로드).
Hypothesis A (top teams): the daily EC level of an evaluation day is recovered from
public train_y labels of the SAME SOURCE (동) on nearby dates.  CH1 (6.147) showed
0.5*model + 0.5*chain-label lowered late days .375 -> .223, but (i) the baseline was
v2 with the `day` bias (since fixed by DC4 season), (ii) its chains used future inputs.
Here: baseline = R3S (DIAG10 OOF, seed mean, local/ec2_DC5_oof.csv), and a CAUSAL
backward chain built from inputs only:
  link(r) = the previous record (calendar date t-1..t-3, same farm) whose hour-23
  indoor state (in_temp, in_hum, in_co2, z-scored per farm) is closest to r's hour-0
  state; distance d_link reported.  Uses r's hour 0 + earlier records' inputs only.
Anchor (causal): walk back along links (max 6 steps) to the first labelled record not
in the query's DIAG10 fold and not locked -> CC_prev; CC_prev2 = mean of first two.
References (NOT legal as is, upper-bound style): CH1 chains (future inputs) with
prev / lin anchors (deep_cal_11_days.csv chain, same rules as CH1).
Predictor of the day mean: 0.5*R3S + 0.5*anchor (fixed, no tuning); no anchor -> R3S.
Row level: R3S hourly shape + new level.
Segments: pass-2 days (>= 179; all eval days are pass 2), pass-2 normal (label < 1),
pass-2 high, all days.
Clue (fixed): CC_prev or CC_prev2 blend beats R3S on pass-2 days (row RMSE) with farm x
5-day cluster bootstrap P(worse) < .0125 (k = 2), AND pass-2 normal days also better.
Label direction: previous labels only for CC (the strict reading); lin is reference."""
import env  # noqa: F401
import json, os
import numpy as np, pandas as pd

LOCK = os.path.join(env.ROOT, u"집", u"코덱스", "analysis", "codex_independent", "ec_final_lock", "locked_days.json")
lock = {(s["farm"], int(s["day"])) for s in json.load(open(LOCK, encoding="utf-8"))["selected"]}
R = pd.read_csv(os.path.join(env.LOCAL, "st_dong_assign_v1.csv")).sort_values(["farm", "day"]).reset_index(drop=True)
R["date"] = R.groupby("farm").role.transform(lambda s: np.cumsum(s.values != "second") - 1)
X = pd.concat([pd.read_csv(os.path.join(env.DATA, f), usecols=["row_id", "in_temp", "in_hum", "in_co2"]) for f in ("train_X.csv", "test_X.csv")])
X = X[X.row_id.str[:3].isin(["F13", "F47"])].copy()
X["farm"], X["day"], X["hour"] = X.row_id.str[:3], X.row_id.str[4:7].astype(int), X.row_id.str[8:10].astype(int)
S = ["in_temp", "in_hum", "in_co2"]
for f in ("F13", "F47"):
    m = X.farm == f
    X.loc[m, S] = (X.loc[m, S] - X.loc[m, S].mean()) / X.loc[m, S].std()
h0 = X[X.hour == 0].set_index(["farm", "day"])[S]; h23 = X[X.hour == 23].set_index(["farm", "day"])[S]
Y = pd.read_csv(os.path.join(env.DATA, "train_y.csv")); Y = Y[Y.row_id.str[:3].isin(["F13", "F47"])].copy()
Y["farm"], Y["day"] = Y.row_id.str[:3], Y.row_id.str[4:7].astype(int)
ec = Y.groupby(["farm", "day"]).sub_ec.mean()
ec = ec[[k not in lock for k in ec.index]]

# causal backward links
link, dl = {}, {}
for f in ("F13", "F47"):
    G = R[R.farm == f]
    for r in G.itertuples():
        C = G[(G.date >= r.date - 3) & (G.date < r.date)]
        if (f, r.day) not in h0.index or not len(C):
            continue
        a = h0.loc[(f, r.day)].values
        best, bd = None, np.inf
        for c in C.itertuples():
            if (f, c.day) not in h23.index:
                continue
            b = h23.loc[(f, c.day)].values
            dd = np.sqrt(np.nanmean((a - b) ** 2)) + 0.25 * (r.date - c.date - 1)
            if dd < bd:
                best, bd = c.day, dd
        if best is not None:
            link[(f, r.day)], dl[(f, r.day)] = best, bd

O = pd.read_csv(os.path.join(env.LOCAL, "ec2_DC5_oof.csv")); O = O[O.validator == "DIAG10"].copy()
O["p"] = O[["r3s_7", "r3s_101", "r3s_2024"]].mean(axis=1)
D = O.groupby(["farm", "day"]).agg(fold=("validation_fold", "first"), pm=("p", "mean"), y=("sub_ec", "mean")).reset_index()
fold_of = dict(zip(zip(D.farm, D.day), D.fold))
CH = pd.read_csv(os.path.join(env.LOCAL, "deep_cal_11_days.csv"))[["farm", "day", "cal", "chain"]]
D = D.merge(CH, on=["farm", "day"], how="left")
CHL = CH.merge(ec.rename("ec").reset_index(), on=["farm", "day"])


def usable(k, fold):
    return k in ec.index and fold_of.get(k, -1) != fold


rows = []
for r in D.itertuples(index=False):
    k = (r.farm, r.day); anc = []; steps = []
    cur = k
    for s in range(6):
        if cur not in link:
            break
        cur = (r.farm, link[cur])
        if usable(cur, r.fold):
            anc.append(ec[cur]); steps.append(s + 1)
            if len(anc) == 2:
                break
    A = CHL[(CHL.farm == r.farm) & (CHL.chain == r.chain) & (CHL.day != r.day)]
    A = A[[fold_of.get((f, d), -1) != r.fold for f, d in zip(A.farm, A.day)]]
    chp = chl = np.nan
    if len(A):
        P = A[A.day < r.day]
        if len(P):
            dp = (P.cal - r.cal).abs(); chp = P.ec[dp == dp.min()].mean()
        lo, hi = A[A.cal <= r.cal], A[A.cal >= r.cal]
        if len(lo) and len(hi):
            a, b = lo.loc[lo.cal.idxmax()], hi.loc[hi.cal.idxmin()]
            chl = a.ec if b.cal == a.cal else a.ec + (b.ec - a.ec) * (r.cal - a.cal) / (b.cal - a.cal)
        else:
            dd = (A.cal - r.cal).abs(); chl = A.ec[dd == dd.min()].mean()
    rows.append(dict(cc_prev=anc[0] if anc else np.nan, cc_prev2=np.mean(anc) if anc else np.nan,
                     cc_steps=steps[0] if steps else np.nan, ch_prev=chp, ch_lin=chl, dlink=dl.get(k, np.nan)))
D = pd.concat([D, pd.DataFrame(rows)], axis=1)
P = ["cc_prev", "cc_prev2", "ch_prev", "ch_lin"]
for c in P:
    D["has_" + c] = D[c].notna()
    D["b_" + c] = 0.5 * D.pm + 0.5 * D[c].fillna(D.pm)
D["p2"] = D.day >= 179
r_ = lambda e: float(np.sqrt(np.mean(np.square(e))))

# link sanity: does the causal link connect records whose labels continue?
L = pd.DataFrame([(f, d, p) for (f, d), p in link.items()], columns=["farm", "day", "pday"])
L["y"] = [ec.get((f, d), np.nan) for f, d in zip(L.farm, L.day)]; L["yp"] = [ec.get((f, d), np.nan) for f, d in zip(L.farm, L.pday)]
Lc = L.dropna()
print("causal links %d; both labelled %d; label corr along link %.3f, mean |diff| %.3f"
      % (len(L), len(Lc), np.corrcoef(Lc.y, Lc.yp)[0, 1], (Lc.y - Lc.yp).abs().mean()))
dg = R.set_index(["farm", "day"]).dong
same = np.mean([dg.get((f, d)) == dg.get((f, p)) for f, d, p in zip(L.farm, L.day, L.pday)])
print("link keeps the same 동 (st_dong_assign): %.2f" % same)
print("anchor availability: " + "  ".join("%s %d/%d (pass-2 %d/%d)" % (c, D["has_" + c].sum(), len(D), D[D.p2]["has_" + c].sum(), D.p2.sum()) for c in P))
print("CC first anchor steps back (pass 2):", D[D.p2].cc_steps.value_counts().sort_index().to_dict())

cols = ["pm"] + P + ["b_" + c for c in P]
print("\nday-level RMSE")
print("%-14s %4s " % ("segment", "n") + " ".join("%9s" % c for c in cols))
for nm, m in (("all", D.y.notna()), ("pass-2", D.p2), ("pass-2 normal", D.p2 & (D.y < 1)), ("pass-2 high", D.p2 & (D.y >= 1)),
              ("pass-1", ~D.p2), ("pass-1 normal", ~D.p2 & (D.y < 1))):
    g = D[m]
    print("%-14s %4d " % (nm, len(g)) + " ".join("%9.4f" % r_(g[c].fillna(g.pm) - g.y) for c in cols))

O2 = O.merge(D[["farm", "day", "pm", "p2"] + ["b_" + c for c in P]], on=["farm", "day"])
O2["cl"] = O2.farm + "_" + (O2.day // 5).astype(str)
print("\nrow-level RMSE (R3S hourly shape + blended level)")
rng = np.random.default_rng(20261004); res = {}
for nm, m in (("all", O2.sub_ec.notna()), ("pass-2", O2.p2), ("pass-2 normal", O2.p2 & (O2.groupby(["farm", "day"]).sub_ec.transform("mean") < 1))):
    g = O2[m]; line = "  %-14s R3S %.4f" % (nm, r_(g.p - g.sub_ec))
    for c in P:
        e1 = g.p - g.pm + g["b_" + c] - g.sub_ec; e0 = g.p - g.sub_ec
        line += "  %s %.4f" % (c, r_(e1))
        if nm == "pass-2":
            dd = (e1 ** 2 - e0 ** 2).groupby(g.cl).agg(["sum", "count"]); s_, n_ = dd["sum"].values, dd["count"].values
            idx = rng.integers(0, len(s_), (20000, len(s_)))
            res[c] = (float(((s_[idx].sum(1) / n_[idx].sum(1)) >= 0).mean()), r_(e1) < r_(e0))
    print(line)
print("\npass-2 P(worse) (cluster bootstrap): " + "  ".join("%s %.4f" % (c, v[0]) for c, v in res.items()))
nrm = O2[O2.p2 & (O2.groupby(["farm", "day"]).sub_ec.transform("mean") < 1)]
clue = any(res[c][0] < .0125 and res[c][1] and r_(nrm.p - nrm.pm + nrm["b_" + c] - nrm.sub_ec) < r_(nrm.p - nrm.sub_ec) for c in ("cc_prev", "cc_prev2"))
D.to_csv(os.path.join(env.LOCAL, "lb1_daytable_v1.csv"), index=False)
print("\nLB1 clue (causal chain, previous labels):", clue)
