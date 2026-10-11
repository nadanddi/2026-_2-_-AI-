# -*- coding: utf-8 -*-
"""EC audit 2: (a) OOF calibration of the base R3-like members (ET+LGB, no MLP/TabPFN) by level;
(b) season sensitivity: eval predictions with interpolated season vs exact-weather-twin season;
(c) anchor bias: training day running mean vs previous same-dong date's 23 h EC."""
import sys, os
sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env  # noqa
PK = r"C:\Users\aozks\AppData\Local\Temp\claude\C--work-farmai\0038d811-b5c9-4a0d-913f-084fb31e14ab\scratchpad\build22\팜모니_정형데이터_재현패키지_22\EC"
sys.path.insert(0, PK)
import numpy as np, pandas as pd
from pathlib import Path
import model as M
OUT = Path(r"C:\work\farmai\집\클로드\research\audit_ec")
tr = pd.read_pickle(OUT / "tr.pkl"); q = pd.read_pickle(OUT / "q.pkl")
tr["day"] = tr.row_id.str[4:7].astype(int); q["day"] = q.row_id.str[4:7].astype(int)
y = tr.sub_ec.to_numpy(float)
def fitpred(trn, tst, seed=7, n_et=250):
    from sklearn.pipeline import make_pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.ensemble import ExtraTreesRegressor
    e = make_pipeline(SimpleImputer(strategy="median"), ExtraTreesRegressor(n_estimators=n_et, max_features=1., n_jobs=6, random_state=seed))
    e.fit(trn[M.FULL_R3], trn.sub_ec); l = M.lg(seed); l.set_params(n_jobs=6); l.fit(trn[M.BASE_R3], trn.sub_ec)
    return e.predict(tst[M.FULL_R3]), l.predict(tst[M.BASE_R3])
# (a) OOF by record, 5 folds, contiguous record blocks of 10 to reduce twin leakage
rec = tr.farm + "_" + tr.day.astype(str)
blk = (tr.day // 10)
fold = (blk * 7 + (tr.farm == "F47") * 3) % 5
oe = np.zeros(len(tr)); ol = np.zeros(len(tr))
for k in range(5):
    m = fold.values == k
    a, b = fitpred(tr[~m], tr[m]); oe[m] = a; ol[m] = b
    print("fold", k, "done", flush=True)
tr["oof"] = (.6 * oe + .3 * ol) / .9; tr["oe"] = oe; tr["ol"] = ol
tr["oof_s"] = M.shrink(tr.oof.values, tr)
d = tr.groupby(["farm", "day"]).agg(y=("sub_ec", "mean"), p=("oof_s", "mean"), pe=("oe", "mean"), pl=("ol", "mean")).reset_index()
d["p2"] = d.day >= 179
for name, g in [("all", d), ("pass2", d[d.p2])]:
    sl = np.polyfit(g.p, g.y, 1)
    print("%s days %d: mean y %.3f mean p %.3f (ET %.3f LGB %.3f) | slope y~p %.3f intercept %.3f | RMSE day-level %.3f" %
          (name, len(g), g.y.mean(), g.p.mean(), g.pe.mean(), g.pl.mean(), sl[0], sl[1], np.sqrt(((g.y - g.p) ** 2).mean())))
d["bin"] = pd.qcut(d.p, 6)
print(d.groupby("bin", observed=True).agg(n=("y", "size"), p=("p", "mean"), y=("y", "mean"), y_med=("y", "median")).round(3).to_string())
g2 = d[d.p2].copy(); g2["bin"] = pd.qcut(g2.p, 4)
print("pass2:\n", g2.groupby("bin", observed=True).agg(n=("y", "size"), p=("p", "mean"), y=("y", "mean")).round(3).to_string())
# simple linear recalibration value (OOF on pass-1 -> pass-2): fit y~p on pass-1 days, score on pass-2 rows
c = np.polyfit(d[~d.p2].p, d[~d.p2].y, 1)
rr = tr[tr.day >= 179]
base = np.sqrt(((rr.sub_ec - rr.oof_s) ** 2).mean()); cal = np.sqrt(((rr.sub_ec - (c[0] * rr.oof_s + c[1])) ** 2).mean())
print("recalibration fit on pass-1 days: slope %.3f int %.3f ; pass-2 row RMSE %.4f -> %.4f" % (c[0], c[1], base, cal))
tr[["row_id", "oe", "ol", "oof", "oof_s", "sub_ec"]].to_csv(OUT / "a2_oof.csv", index=False)

# (b) season sensitivity
E = pd.read_csv(OUT / "a1_eval_season.csv")
pe, pl = fitpred(tr, q)
q2 = q.copy(); tw = dict(((f, d), s) for f, d, s in E[["farm", "day", "twin_season"]].itertuples(index=False) if np.isfinite(s))
q2["season"] = [tw.get((f, d), s) for f, d, s in zip(q2.farm, q2.day, q2.season)]
pe2, pl2 = fitpred(tr, q2)
r1 = (.6 * pe + .3 * pl) / .9; r2 = (.6 * pe2 + .3 * pl2) / .9
Q = pd.DataFrame({"rec": q.row_id.str[:7], "r1": r1, "r2": r2}).groupby("rec").mean()
Q["d"] = Q.r2 - Q.r1
print("season interp -> twin: record-mean change: RMS %.4f, max |.| %.4f, mean %.4f; records changed >.02: %d" %
      (np.sqrt((Q.d ** 2).mean()), Q.d.abs().max(), Q.d.mean(), int((Q.d.abs() > .02).sum())))
print(Q[Q.d.abs() > .01].round(3).to_string())
# seasonal +-5 sensitivity overall
q3 = q.copy(); q3["season"] = q3.season + 5
pe3, pl3 = fitpred(tr, q3); r3 = (.6 * pe3 + .3 * pl3) / .9
print("season +5 for all eval: row RMS change %.4f, mean %.4f" % (np.sqrt(((r3 - r1) ** 2).mean()), (r3 - r1).mean()))
