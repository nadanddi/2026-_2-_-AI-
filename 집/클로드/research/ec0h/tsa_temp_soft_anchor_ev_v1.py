# -*- coding: utf-8 -*-
"""TSA-EV (2026-10-12 집 클로드): temperature soft anchor on the EV final chain (v22), allowed by 6.631.  Fixed before running.
  base     : EV v22 final-chain predictions, 4 cells (columns v22_a_{7,101}{A,B}), from ev_validator/temp/local_ckpt/ev_all.pkl.
  targets  : EV val records whose v22 == v18 at all hours (no anchor applied) and whose v7m prev p is same ID, p >= .8, smaller
             number, and p is a val record of the same fold (inputs allowed; its v22 prediction exists).
  x        = v22_p(23) - v22_t(0) ;  pred_h = v22_t,h + w_h * x
  w_h      = per-hour LS slope of (y_t,h - v22_t,h) on x over targets of the OTHER folds with number < 179 (cell-pooled), clipped [0,1].
PASS iff pass-1 targets: all 4 cells better AND chunk-cluster P(worse) < .025 AND top-record share <= 50%; pass-2 reported.
"""
import os, sys
sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env
import numpy as np, pandas as pd
rm = lambda e: float(np.sqrt(np.mean(np.square(e))))
H = os.path.join(env.ROOT, u"집", u"클로드", "research")
E = pd.read_pickle(os.path.join(H, "ev_validator", "temp", "local_ckpt", "ev_all.pkl")); E["rec"] = E.row_id.str[:7]
Fv = pd.read_csv(os.path.join(H, "ev_validator", "ev_folds_v1.csv")); V = Fv[Fv.role == "val"].set_index("rec")
L = pd.read_csv(os.path.join(env.LOCAL, "linker_v7", "cv_all400_choice_v7m.csv")).set_index("rec")
cells = [c for c in E.columns if c.startswith("v22_a_")]; print("cells", cells)
Y = E.pivot_table(index="rec", columns="hour", values="sub_temp")
P = {c: E.pivot_table(index="rec", columns="hour", values=c) for c in cells}
P18 = {c: E.pivot_table(index="rec", columns="hour", values=c.replace("v22_", "v18_")) for c in cells}
noanch = {r for r in Y.index if all(np.allclose(P[c].loc[r].values, P18[c].loc[r].values) for c in cells)}
T = []
for t in V.index:
    if t not in L.index or t not in noanch or t not in Y.index: continue
    pr = L.loc[t]; p = pr.prev
    if pr.prev_is_labelled or pr.xf or pr.p < .8 or p not in V.index or V.at[p, "fold"] != V.at[t, "fold"] or p[:3] != t[:3] or int(p[4:7]) >= int(t[4:7]) or p not in Y.index: continue
    T.append(dict(t=t, p=p, fold=V.at[t, "fold"], chunk=V.at[t, "chunk"]))
T = pd.DataFrame(T); print("targets", len(T), "| pass-2", T[T.t.str[4:7].astype(int) >= 179].t.tolist())
out = {}
for c in cells:
    new = P[c].copy(); Q = P[c]
    for k in T.fold.unique():
        oth = T[(T.fold != k) & (T.t.str[4:7].astype(int) < 179)]
        x = Q.loc[oth.p, 23].values - Q.loc[oth.t, 0].values
        res = (Y.loc[oth.t] - Q.loc[oth.t]).values
        w = np.clip((x[:, None] * res).sum(0) / (x ** 2).sum(), 0, 1)
        for _, r in T[T.fold == k].iterrows(): new.loc[r.t] = Q.loc[r.t].values + w * (Q.at[r.p, 23] - Q.at[r.t, 0])
    out[c] = new
print("w_h example:", " ".join("%d:%.2f" % (h, v) for h, v in enumerate(w) if h % 3 == 0))
rng = np.random.default_rng(20261223)
for part, cond in (("pass-1", lambda s: int(s[4:7]) < 179), ("pass-2", lambda s: int(s[4:7]) >= 179)):
    recs = [r for r in T.t if cond(r)]
    if not recs: print(part, "none"); continue
    h0 = np.mean([rm(out[c].loc[recs, 0] - Y.loc[recs, 0]) / rm(P[c].loc[recs, 0] - Y.loc[recs, 0]) - 1 for c in cells])
    cl_ = [rm((out[c].loc[recs] - Y.loc[recs]).values) / rm((P[c].loc[recs] - Y.loc[recs]).values) - 1 for c in cells]
    per = pd.Series(np.mean([(((out[c].loc[recs] - Y.loc[recs]) ** 2).sum(1) - ((P[c].loc[recs] - Y.loc[recs]) ** 2).sum(1)).values for c in cells], 0), index=recs)
    cl = per.groupby(T.set_index("t").chunk.reindex(recs).values).sum(); idx = rng.integers(0, len(cl), (20000, len(cl))); pw = float((cl.values[idx].sum(1) >= 0).mean())
    share = float(per.min() / per.sum()) if per.sum() < 0 else float("nan")
    print(f"{part}: records {len(recs)} | 0h {100*h0:+.1f}% | cells {' '.join('%+.2f%%' % (100*x) for x in cl_)} | improved {int((per<0).sum())}/{len(per)} | cluster P {pw:.3f} | top share {share:.2f}")
    if part == "pass-1": print("VERDICT TSA-EV:", "PASS" if all(x < 0 for x in cl_) and pw < .025 and share <= .5 else "FAIL")
