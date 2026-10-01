# -*- coding: utf-8 -*-
"""Verification (analysis-verification gate) of the 2026-10-02 EC conclusions:
recompute key numbers by a different route than the original scripts.
Run:  cd 집/클로드/research && PYTHONPATH="" py -3.12 -u verify_ec_20261002.py"""
import env, os, json, math
import numpy as np, pandas as pd
R = env.ROOT
o = pd.read_csv(os.path.join(R, u"집", u"코덱스", "local", "ec_restart_phase3_20261001_v1", "oof_predictions.csv"), encoding="utf-8-sig")
d = o[o.validator == "DIAG10"]
# 1. v2 DIAG10 RMSE via plain-python sum (orig: numpy 0.2139 / Codex .213874)
ss = 0.0
for a, b in zip(d.v2.tolist(), d.sub_ec.tolist()):
    ss += (a - b) ** 2
print("1. v2 DIAG10 RMSE python-sum %.6f (n=%d, days=%d)" % (math.sqrt(ss / len(d)), len(d), d.groupby(["farm", "day"]).ngroups))
# 2. level-oracle 0.0778 via within-day residual deviation (orig: swap level)
e = d.v2 - d.sub_ec
dev = e - e.groupby([d.farm, d.day]).transform("mean")
print("2. shape-only RMSE (resid minus its day mean) %.4f ; level share %.2f%%" % (np.sqrt((dev ** 2).mean()), 100 * (1 - (dev ** 2).sum() / (e ** 2).sum())))
# 3. sealed&high squared-error share 45.9% via explicit day loop
tx = pd.read_csv(os.path.join(env.DATA, "train_X.csv"), usecols=["row_id", "act_circfan", "act_vent"])
d = d.merge(tx, on="row_id")
num = den = 0.0; nd = 0
for (f, day), g in d.groupby(["farm", "day"]):
    se = float(((g.v2 - g.sub_ec) ** 2).sum()); den += se
    if g.act_circfan.mean() < 10 and (g.act_vent == 0).mean() > 0.85 and g.sub_ec.mean() >= 1.2:
        num += se; nd += 1
print("3. sealed&high days %d, squared-error share %.1f%%" % (nd, 100 * num / den))
# 4. same calendar date pairs (alternative calendar: deep_cal_9 instead of deep_cal_11), lock excluded
lock = {(s["farm"], int(s["day"])) for s in json.load(open(os.path.join(R, u"집", u"코덱스", "analysis", "codex_independent", "ec_final_lock", "locked_days.json"), encoding="utf-8"))["selected"]}
y = pd.read_csv(os.path.join(env.DATA, "train_y.csv")); p = y.row_id.str.split("_", expand=True)
y["farm"], y["day"] = p[0], p[1].astype(int)
y = y[y.farm.isin(["F13", "F47"]) & y.sub_ec.notna()]
ec = y.groupby(["farm", "day"]).sub_ec.mean()
ec = ec[[k not in lock for k in ec.index]]
c9 = pd.read_csv(os.path.join(env.LOCAL, "deep_cal_9_days.csv")).set_index(["farm", "day"]).cal
both = one = n = 0
for (f, c), g in pd.DataFrame({"ec": ec, "cal": c9.reindex(ec.index)}).reset_index().groupby(["farm", "cal"]):
    if len(g) >= 2:
        n += 1; h = (g.ec.values[:2] >= 1.2).sum(); both += h == 2; one += h == 1
print("4. same-date pairs (deep_cal_9): n=%d both-high %d one-high %d" % (n, both, one))
# 5. TW1 H4 on seed-mean v2 (orig: per seed -0.03..-0.14%)
o = o.merge(pd.read_csv(os.path.join(env.DATA, "train_X.csv"), usecols=["row_id", "in_temp"]), on="row_id")
o = o.sort_values(["validator", "validation_fold", "farm", "day", "hour"])
S = (o.v2 - 0.8 * o.r3) / 0.2
def inv(g):
    out = g.values; pp = np.empty(len(out)); s = 0.0
    for h, v in enumerate(out):
        pp[h] = (v - 0.5 * s / (h + 1)) / (0.5 + 0.5 / (h + 1)); s += pp[h]
    return pd.Series(pp, index=g.index)
t = S.groupby([o.validator, o.validation_fold, o.farm, o.day]).transform(inv)
gate = ((o.r3 - t).abs() >= 0.15) & (o.in_temp >= 10)
h4 = np.where(gate, 0.75 * o.v2 + 0.25 * t, o.v2)
for v in ("DIAG10", "A", "EXT12"):
    m = (o.validator == v).values
    a = np.sqrt(((o.v2[m] - o.sub_ec[m]) ** 2).mean()); b = np.sqrt(((h4[m] - o.sub_ec[m]) ** 2).mean())
    print("5. H4 on seed-mean %-6s %.4f -> %.4f (%+.2f%%)" % (v, a, b, 100 * (b / a - 1)))
# 6. CH1 b_near row RMSE 0.1912 via row-level merge from saved day table
D = pd.read_csv(os.path.join(env.LOCAL, "ch1_daytable.csv"))[["farm", "day", "pm", "b_near"]]
d6 = o[o.validator == "DIAG10"].merge(D, on=["farm", "day"])
print("6. CH1 b_near row RMSE %.4f (v2 %.4f)" % (np.sqrt(((d6.v2 - d6.pm + d6.b_near - d6.sub_ec) ** 2).mean()), np.sqrt(((d6.v2 - d6.sub_ec) ** 2).mean())))
