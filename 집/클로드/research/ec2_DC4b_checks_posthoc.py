# -*- coding: utf-8 -*-
"""DC4b (descriptive checks, no new fitting on labels): (a) label-free sanity of the
season map on the 60 TEST days (map built from all training days' inputs); (b) DC4
OOF breakdown by farm / high-EC / sealed / late x calendar.
Run:  cd 집/클로드/research && PYTHONPATH="" py -3.12 -u ec2_DC4b_checks_posthoc.py"""
import env, os, sys, importlib.util, numpy as np, pandas as pd
sys.argv = ["x"]
spec = importlib.util.spec_from_file_location("dc4", os.path.join(os.path.dirname(os.path.abspath(__file__)), "ec2_DC4_exact_twin_anchor_v1.py"))
dc4 = importlib.util.module_from_spec(spec); spec.loader.exec_module(dc4)
tX = pd.read_csv(os.path.join(env.DATA, "train_X.csv")); sX = pd.read_csv(os.path.join(env.DATA, "test_X.csv"))
full = pd.concat([tX, sX], ignore_index=True)
full = full[full.row_id.str[:3].isin(["F13", "F47"])]
wv = dc4.weather_vectors(full[full.row_id.isin(tX.row_id)])        # TRAINING inputs only
p = tX.row_id.str.split("_", expand=True); tr = pd.DataFrame({"farm": p[0], "day": p[1].astype(int)})
tr = tr[tr.farm.isin(["F13", "F47"])].drop_duplicates()
q = sX.row_id.str.split("_", expand=True); te = pd.DataFrame({"farm": q[0], "day": q[1].astype(int)}).drop_duplicates().reset_index(drop=True)
season, te["season"] = dc4.season_index(tr, te, wv)
cal = pd.read_csv(os.path.join(env.LOCAL, "deep_cal_9_days.csv"))[["farm", "day", "cal"]]
te = te.merge(cal, on=["farm", "day"])
print("(a) TEST days: corr(season map, weather calendar) %.3f; spearman %.3f" % (np.corrcoef(te.season, te.cal)[0, 1], te[["season", "cal"]].corr("spearman").iloc[0, 1]))
tr2 = tr.merge(cal, on=["farm", "day"]); tr2["season"] = [season.get((f, d), np.nan) for f, d in zip(tr2.farm, tr2.day)]
c1 = tr2[tr2.day < 179].dropna(); b = np.polyfit(c1.cal, c1.season, 1)
te["cal_hat"] = (te.season - b[1]) / b[0]
print("    season->calendar scale from pass-1 training days: season = %.2f*cal + %.1f; test |cal_hat - cal| median %.1f, max %.1f" % (b[0], b[1], (te.cal_hat - te.cal).abs().median(), (te.cal_hat - te.cal).abs().max()))
print(te.assign(cal_hat=te.cal_hat.round(1)).groupby("farm").apply(lambda g: list(zip(g.day, g.cal, g.cal_hat.round(0))), include_groups=False).to_string()[:1500])
o = pd.read_csv(os.path.join(env.LOCAL, "ec2_DC4_oof.csv")); D = o[o.validator == "DIAG10"].copy()
D["dm"] = D.groupby(["farm", "day"]).sub_ec.transform("mean")
r = lambda e: float(np.sqrt(np.mean(np.square(e))))
S = (7, 101, 2024)
print("\n(b) DIAG10 breakdown (seed mean): base -> season")
for nm, m in (("F13", D.farm == "F13"), ("F47", D.farm == "F47"), ("high EC>=1.2", D.dm >= 1.2), ("rest", D.dm < 1.2),
              ("late & cal<70", (D.day >= 179) & (D.cal < 70)), ("late & cal>=70", (D.day >= 179) & (D.cal >= 70)), ("early", D.day < 179)):
    g = D[m]
    print("  %-15s rows %5d  %.4f -> %.4f" % (nm, len(g), np.mean([r(g["base_%d" % s] - g.sub_ec) for s in S]), np.mean([r(g["seas_%d" % s] - g.sub_ec) for s in S])))
