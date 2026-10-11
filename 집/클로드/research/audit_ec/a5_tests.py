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

# (B) anchor formula on OOF (a2_oof.csv)
O = pd.read_csv(OUT / "a2_oof.csv"); O["farm"], O["day"], O["hour"] = O.row_id.str[:3], O.row_id.str[4:7].astype(int), O.row_id.str[8:10].astype(int)
R = pd.read_csv(os.path.join(PK, "st_dong_assign_v1.csv")).sort_values(["farm", "day"]).reset_index(drop=True)
R["date"] = R.groupby("farm").role.transform(lambda s: np.cumsum(s.values != "second") - 1)
Y = O.pivot_table(index=["farm", "day"], columns="hour", values="sub_ec"); P = O.pivot_table(index=["farm", "day"], columns="hour", values="oof_s")
key = {}
for f, dg, dt, d in R[["farm", "dong", "date", "day"]].itertuples(index=False): key.setdefault((f, dg, dt), d)
res = []
for f, dg, dt, d in R[["farm", "dong", "date", "day"]].itertuples(index=False):
    pdv = key.get((f, dg, dt - 1))
    if pdv is None or (f, d) not in Y.index or (f, pdv) not in Y.index: continue
    y = Y.loc[(f, d)].values; p = P.loc[(f, d)].values; A = Y.loc[(f, pdv), 23]
    cm = np.cumsum(p) / np.arange(1, 25)
    full = p + (A - cm)
    h0 = p + (A - p[0])
    for w in (1.0,):
        pass
    res.append(dict(farm=f, day=d, p2=d >= 179, e_none=np.mean((y - p) ** 2), e_full=np.mean((y - full) ** 2), e_h0=np.mean((y - h0) ** 2),
                    **{"r%02d" % h: y[h] - p[h] for h in range(24)}, **{"d%02d" % h: A - cm[h] for h in range(24)},
                    **{"sn%02d" % h: (y[h] - p[h]) ** 2 for h in range(24)}, **{"sf%02d" % h: (y[h] - full[h]) ** 2 for h in range(24)}))
T = pd.DataFrame(res)
for nm, gg in [("all", T), ("pass2", T[T.p2])]:
    print("%s pairs %d: RMSE none %.4f | EA2 full %.4f | h0-anchor %.4f" % (nm, len(gg), np.sqrt(gg.e_none.mean()), np.sqrt(gg.e_full.mean()), np.sqrt(gg.e_h0.mean())))
print("by hour RMSE none vs EA2-full (all pairs):")
print(" ".join("h%02d %.3f/%.3f" % (h, np.sqrt(T["sn%02d" % h].mean()), np.sqrt(T["sf%02d" % h].mean())) for h in range(0, 24, 2)))
# least-squares optimal per-hour weight w_h for p + w_h (A - cm_h)
for nm, gg in [("all", T), ("pass2", T[T.p2])]:
    print(nm, "LS-optimal weight w_h for p + w_h (A - cm_h):", " ".join("h%02d %.2f" % (h, (gg["r%02d" % h] * gg["d%02d" % h]).sum() / (gg["d%02d" % h] ** 2).sum()) for h in range(0, 24, 3)))
# guarded version: only when |A - cm| <= .30 (like SG2 guard)
for nm, gg in [("all", T), ("pass2", T[T.p2])]:
    tot_n = tot_f = 0
    for h in range(24):
        d = gg["d%02d" % h]; r = gg["r%02d" % h]
        tot_n += (r ** 2).sum(); tot_f += ((r - d) ** 2).sum()
    print(nm, "check RMSE none %.4f full %.4f" % (np.sqrt(tot_n / (24 * len(gg))), np.sqrt(tot_f / (24 * len(gg)))))
