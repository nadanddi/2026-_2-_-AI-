# -*- coding: utf-8 -*-
"""EC audit 5:
(A) Season train/eval mismatch test on training pass-2 days: OOF prediction with the season the model saw in training
    (PAV of exact-twin dates) vs the eval-style season (interpolation by record number from anchors >5 records away).
(B) EA2 anchor formula on OOF predictions of training same-dong pairs: full running-mean shift vs hour-0-only shift vs none."""
import sys, os
sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env  # noqa
PK = r"C:\Users\aozks\AppData\Local\Temp\claude\C--work-farmai\0038d811-b5c9-4a0d-913f-084fb31e14ab\scratchpad\build22\팜모니_정형데이터_재현패키지_22\EC"
sys.path.insert(0, PK)
import json
import numpy as np, pandas as pd
from pathlib import Path
import model as M
OUT = Path(r"C:\work\farmai\집\클로드\research\audit_ec")
tr = pd.read_pickle(OUT / "tr.pkl"); tr["day"] = tr.row_id.str[4:7].astype(int)
notes = json.load(open(os.path.join(PK, "season_transform.json"), encoding="utf-8"))
fold = ((tr.day // 10) * 7 + (tr.farm == "F47") * 3) % 5
# eval-style season for pass-2 training days
def evstyle(f, d):
    ax = np.array(notes[f]["anchor_day"]); ay = np.array(notes[f]["anchor_season_pav"])
    k = np.abs(ax - d) > 5
    return float(np.interp(d, ax[k], ay[k]))
tr["season_ev"] = [evstyle(f, d) if d >= 179 else s for f, d, s in zip(tr.farm, tr.day, tr.season)]
p2 = tr.day >= 179
dd = tr[p2].groupby(["farm", "day"]).agg(s=("season", "first"), se=("season_ev", "first"))
print("pass-2 training days: |season_used - eval-style| mean %.2f median %.2f max %.2f" % ((dd.s - dd.se).abs().mean(), (dd.s - dd.se).abs().median(), (dd.s - dd.se).abs().max()))
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import ExtraTreesRegressor
pa = np.full(len(tr), np.nan); pb = np.full(len(tr), np.nan)
for k in range(5):
    m = (fold == k).values; t = m & p2.values
    if not t.any(): continue
    e = make_pipeline(SimpleImputer(strategy="median"), ExtraTreesRegressor(n_estimators=250, max_features=1., n_jobs=6, random_state=7)).fit(tr[~m][M.FULL_R3], tr[~m].sub_ec)
    l = M.lg(7); l.set_params(n_jobs=6); l.fit(tr[~m][M.BASE_R3], tr[~m].sub_ec)
    X = tr[t]; Xe = X.copy(); Xe["season"] = X.season_ev
    pa[t] = (.6 * e.predict(X[M.FULL_R3]) + .3 * l.predict(X[M.BASE_R3])) / .9
    pb[t] = (.6 * e.predict(Xe[M.FULL_R3]) + .3 * l.predict(Xe[M.BASE_R3])) / .9
    print("fold", k, flush=True)
r = tr[p2].copy(); r["pa"] = pa[p2.values]; r["pb"] = pb[p2.values]
ra, rb = np.sqrt(((r.sub_ec - r.pa) ** 2).mean()), np.sqrt(((r.sub_ec - r.pb) ** 2).mean())
print("pass-2 rows RMSE: train-style season %.4f | eval-style season %.4f  (%+.1f%%)" % (ra, rb, 100 * (rb / ra - 1)))
g = r.groupby(["farm", "day"]).agg(y=("sub_ec", "mean"), a=("pa", "mean"), b=("pb", "mean"), s=("season", "first"), se=("season_ev", "first"))
g["ea"] = (g.a - g.y).abs(); g["eb"] = (g.b - g.y).abs()
print("days where eval-style is worse by >.05:", int((g.eb - g.ea > .05).sum()), " better by >.05:", int((g.ea - g.eb > .05).sum()), "of", len(g))
print(g[(g.eb - g.ea).abs() > .05].round(3).to_string())
def interior(f, d):
    ax = np.array(notes[f]["anchor_day"]); k = np.abs(ax - d) > 5
    return (ax[k] < d).any() and (ax[k] > d).any()
r["inter"] = [interior(f, d) for f, d in zip(r.farm, r.day)]
for exc in (True,):
    rr = r[r.inter]
    print("interior pass-2 days only (%d days, |season diff| mean %.2f): RMSE train-style %.4f eval-style %.4f" % (rr.groupby(["farm","day"]).ngroups, (rr.season-rr.season_ev).abs().groupby([rr.farm,rr.day]).first().mean(), np.sqrt(((rr.sub_ec-rr.pa)**2).mean()), np.sqrt(((rr.sub_ec-rr.pb)**2).mean())))
    small = rr[(rr.season-rr.season_ev).abs() <= 13]
    print("  ... with |season diff|<=13 (eval-like magnitude; %d days, mean diff %.2f): RMSE %.4f -> %.4f" % (small.groupby(["farm","day"]).ngroups, (small.season-small.season_ev).abs().groupby([small.farm,small.day]).first().mean(), np.sqrt(((small.sub_ec-small.pa)**2).mean()), np.sqrt(((small.sub_ec-small.pb)**2).mean())))

