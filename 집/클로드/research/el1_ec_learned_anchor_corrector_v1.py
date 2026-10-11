# -*- coding: utf-8 -*-
"""EL1x (2026-10-12 집 클로드; new direction: let a model LEARN when and how much to trust the same-house anchor instead of a
hand rule).  Earlier anchor-feature attempt AF0 (6.317) failed because neighbours were often the wrong house; linker v7 is now
.95 accurate at p >= .8.  Fixed before running.
Residual corrector (LightGBM, small, monotone-free): target r = y - sg (9th-round-like EC OOF, RV1 CUR + SG2, seed mean).
Features per row: gap = A - cm_h (A = linker v7 mirror-CV previous record 23 h EC if labelled, else NaN), p, xf, hour, cm_h, sg,
  gap * p, has_anchor.  Trained on DIAG10 rows of the OTHER folds (same fold structure as the OOF), with purge (same farm +-1,
  other farm +-5 around the predicted fold's days); predictions only where has_anchor (else 0).  Grid (researcher sweep, small):
  num_leaves {4, 8}, n_estimators {100, 300}, shrink {.5, 1.0}  -> 8 settings; SELECTION by pass-2 DIAG10 of F13, JUDGEMENT on
  F47 and vice versa (pass-1/pass-2 shift, 6.584); compared with the EA2 rule (alpha 1).
Reported: per-farm held-out pass-2 RMSE vs sg and vs EA2, record-level counts.
Run:  cd 집/클로드/research && PYTHONPATH="" py -3.12 -u el1_ec_learned_anchor_corrector_v1.py
"""
import os, sys, itertools
import env  # noqa: F401
import numpy as np, pandas as pd
import lightgbm as lgb
LK = os.path.join(env.LOCAL, "linker_v7", "cv_all400_choice_v7m.csv")
rm = lambda e: float(np.sqrt(np.mean(np.square(e))))
FE = ["gap", "p", "xf", "hour", "cm", "sg", "gp", "has"]


def main():
    O = pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "ec_rv1_preds_v1.csv")); O = O[(O.tag == "CUR") & (O.validator == "DIAG10")].copy()
    O = O.groupby(["row_id", "fold", "farm", "day", "hour"], as_index=False).agg(sg=("sg", "mean"), y=("sub_ec", "first"))
    O["rec"] = O.row_id.str[:7]; O = O.sort_values(["rec", "hour"]).reset_index(drop=True)
    O["cm"] = O.groupby("rec").sg.transform(lambda s: s.expanding().mean())
    L = pd.read_csv(LK); L = L[L.prev_is_labelled.astype(bool) & L.anchor_ec23.notna()]
    O = O.merge(L[["rec", "anchor_ec23", "p", "xf"]], on="rec", how="left")
    O["has"] = O.anchor_ec23.notna().astype(float); O["gap"] = O.anchor_ec23 - O.cm; O["gp"] = O.gap * O.p; O["xf"] = O.xf.astype(float)
    O["r"] = O.y - O.sg
    ea2 = np.where(O.has.values > 0 & (O.p.values >= .8), O.sg + O.gap.fillna(0), O.sg)
    ea2 = np.where((O.has > 0) & (O.p >= .8), O.sg + O.gap, O.sg)
    grid = list(itertools.product((4, 8), (100, 300), (.5, 1.0))); preds = {}
    for nl, ne, sh in grid:
        out = np.zeros(len(O))
        for k in O.fold.unique():
            te = (O.fold == k).values
            vd = set(zip(O.farm[te], O.day[te]))
            forb = {(f, d + j) for f, d in vd for j in (-1, 0, 1)} | {("F47" if f == "F13" else "F13", d + j) for f, d in vd for j in range(-5, 6)}
            tr = (~te) & np.array([(f, d) not in forb for f, d in zip(O.farm, O.day)]) & (O.has.values > 0)
            mdl = lgb.LGBMRegressor(n_estimators=ne, learning_rate=.03, num_leaves=nl, min_child_samples=40, subsample=.8, subsample_freq=1,
                                    colsample_bytree=.9, random_state=1, verbose=-1, deterministic=True, force_col_wise=True, n_jobs=2)
            mdl.fit(O.loc[tr, FE], O.r.values[tr]); m = te & (O.has.values > 0)
            out[m] = sh * mdl.predict(O.loc[m, FE])
        preds[(nl, ne, sh)] = O.sg.values + out
    p2 = (O.day >= 179).values
    rows = []
    for g, pr in preds.items():
        rows.append(dict(nl=g[0], ne=g[1], sh=g[2], **{f: 100 * (rm((pr - O.y)[p2 & (O.farm == f).values]) / rm((O.sg - O.y)[p2 & (O.farm == f).values]) - 1) for f in ("F13", "F47")}))
    G = pd.DataFrame(rows); print(G.round(2).to_string())
    e2 = {f: 100 * (rm((ea2 - O.y)[p2 & (O.farm == f).values]) / rm((O.sg - O.y)[p2 & (O.farm == f).values]) - 1) for f in ("F13", "F47")}
    print("EA2 rule vs sg (pass-2): F13 %+.2f%% F47 %+.2f%%" % (e2["F13"], e2["F47"]))
    for sel, held in (("F13", "F47"), ("F47", "F13")):
        b = G.sort_values(sel).iloc[0]; pr = preds[(int(b.nl), int(b.ne), b.sh)]
        m = p2 & (O.farm == held).values
        per = pd.Series((pr - O.y.values) ** 2 - (ea2 - O.y.values) ** 2)[m].groupby(O.rec.values[m]).sum(); per = per[per.abs() > 1e-12]
        print("select on %s -> %s: learned %+.2f%% vs sg, EA2 %+.2f%% vs sg | learned vs EA2 %+.2f%%, records changed %d improved %d" % (
            sel, held, b[held], e2[held], 100 * (rm((pr - O.y)[m]) / rm((ea2 - O.y)[m]) - 1), len(per), int((per < 0).sum())))


if __name__ == "__main__":
    main()
