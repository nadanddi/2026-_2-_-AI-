# -*- coding: utf-8 -*-
"""EC audit 1: base-model features train vs eval, season indicator for eval records."""
import sys, os
sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env  # noqa
PK = r"C:\Users\aozks\AppData\Local\Temp\claude\C--work-farmai\0038d811-b5c9-4a0d-913f-084fb31e14ab\scratchpad\build22\팜모니_정형데이터_재현패키지_22\EC"
sys.path.insert(0, PK)
import numpy as np, pandas as pd
from pathlib import Path
import model as M
from season import identify, vectors, W

D = Path(PK) / "data"
x, tx, tr, q, ids, ts, table, notes = M.load(D)
print("train rows", len(tr), "eval rows", len(q))
tr["p2"] = tr.day >= 179
cols = M.FULL_R3 + ["season"]
cols = list(dict.fromkeys(cols))
rows = []
T2 = tr[tr.p2]
for c in cols:
    a, b, e = tr[c].astype(float), T2[c].astype(float), q[c].astype(float)
    lo, hi = a.min(), a.max()
    rows.append(dict(f=c, tr_nan=a.isna().mean(), ev_nan=e.isna().mean(), tr_mean=a.mean(), p2_mean=b.mean(), ev_mean=e.mean(),
                     tr_sd=a.std(), ev_sd=e.std(), ev_out=((e < lo) | (e > hi)).mean(),
                     smd_p2=(e.mean() - b.mean()) / (b.std() + 1e-9), ev_nuniq=e.nunique()))
R = pd.DataFrame(rows)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
print(R.round(3).to_string())

# season indicator
ev = pd.DataFrame([(f, d, v) for (f, d), v in table.items()], columns=["farm", "day", "season"]).sort_values(["farm", "day"])
vec = vectors(pd.concat([x, tx[["row_id"] + W + M.RAW]]))
p1 = [(f, d) for (f, d) in vec if d < 179 and (f, d) in ts]
ref = np.stack([vec[k] for k in p1])
mu = np.array(notes["standardization"]["mu"]); sd = np.array(notes["standardization"]["sd"])
refz = (ref - mu[None, :, None]) / sd[None, :, None]
out = []
for f, d, s in ev.itertuples(index=False):
    qz = (vec[(f, d)] - mu[:, None]) / sd[:, None]
    dist = np.sqrt(np.nanmean((refz - qz) ** 2, axis=(1, 2)))
    k = np.argsort(dist)[:3]
    sel = np.flatnonzero(dist <= .05)
    twin = np.mean([p1[i][1] for i in sel]) if len(sel) else np.nan
    same = [p1[i] for i in sel]
    out.append(dict(farm=f, day=d, season_interp=s, twin_season=twin, n_twin=len(sel), twin_farms=",".join(sorted(set(a for a, _ in same))),
                    nn_dist=dist[k[0]], nn_day=p1[k[0]][1], nn_farm=p1[k[0]][0]))
E = pd.DataFrame(out)
print(E.round(3).to_string())
E["diff"] = E.season_interp - E.twin_season
print("eval records with twin:", E.n_twin.gt(0).sum(), "/", len(E), " |interp - twin| mean %.1f median %.1f max %.1f" %
      (E["diff"].abs().mean(), E["diff"].abs().median(), E["diff"].abs().max()))
# same for training pass-2: what model used vs raw twin
o2 = []
for (f, d), s in ts.items():
    if d < 179: continue
    qz = (vec[(f, d)] - mu[:, None]) / sd[:, None]
    dist = np.sqrt(np.nanmean((refz - qz) ** 2, axis=(1, 2))); sel = np.flatnonzero(dist <= .05)
    o2.append(dict(farm=f, day=d, season_used=s, twin=np.mean([p1[i][1] for i in sel]) if len(sel) else np.nan, nn_dist=dist.min()))
T = pd.DataFrame(o2)
T["diff"] = T.season_used - T.twin
print("train pass-2 days", len(T), "with twin", T.twin.notna().sum(), "|used - twin| mean %.1f max %.1f" % (T["diff"].abs().mean(), T["diff"].abs().max()))
print(T[T["diff"].abs() > 10].round(2).to_string())
print("nn_dist eval median %.3f, train p2 median %.3f" % (E.nn_dist.median(), T.nn_dist.median()))
OUT = Path(r"C:\work\farmai\집\클로드\research\audit_ec")
E.to_csv(OUT / "a1_eval_season.csv", index=False); R.to_csv(OUT / "a1_feature_table.csv", index=False); T.to_csv(OUT / "a1_train_p2_season.csv", index=False)
q.to_pickle(OUT / "q.pkl"); tr.to_pickle(OUT / "tr.pkl")
