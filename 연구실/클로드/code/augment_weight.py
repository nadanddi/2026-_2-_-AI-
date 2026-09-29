# -*- coding: utf-8 -*-
"""Fair retest: sweep donor weight, and try borrowing the slope instead of rows.

The first pass added donor cold rows at full weight, which swamped each
greenhouse's own data.  The catalogue reports a small IMPROVEMENT for a similar
idea, so the amount matters.  Two things are measured here:

  (a) donor rows at 2/5/10/20% of the own-data weight;
  (b) a surgical alternative -- donors inform ONLY the linear baseline (which
      is what has to extrapolate into the cold), while the tree residual stays
      on the greenhouse's own rows.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from common import load_raw
from augment_cold import feats, CORE, WARM, COLD, DONOR_COLD, MIN_COLD, MIN_WARM, LGB


def mix(lin_pred_tr, lin_pred_te, X, y, Xt, w=None, Xtree=None, ytree=None, wtree=None):
    """0.4*tree + 0.6*(linear + tree-on-residual); tree sets may differ."""
    Xtree = X if Xtree is None else Xtree
    ytree = y if ytree is None else ytree
    res_target = ytree - (lin_pred_tr if Xtree is X else None)
    return None  # unused placeholder


def build_mix(Xlin, ylin, wlin, Xtree, ytree, wtree, Xt):
    lin = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    lin.fit(Xlin, ylin, ridge__sample_weight=wlin)
    res = lgb.LGBMRegressor(**LGB).fit(Xtree, ytree - lin.predict(Xtree),
                                       sample_weight=wtree)
    tree = lgb.LGBMRegressor(**LGB).fit(Xtree, ytree, sample_weight=wtree)
    return 0.4 * tree.predict(Xt) + 0.6 * (lin.predict(Xt) + res.predict(Xt))


def main():
    tX, ty, sX = load_raw()
    d = tX.merge(ty[["row_id", "sub_temp"]], on="row_id", how="left")
    farms = sorted(f for f in d.farm.unique() if f not in ("F13", "F47", "F32"))
    print("피처 생성 중 ...", flush=True)
    F = {f: feats(d[d.farm == f].copy()) for f in farms}
    cols = [c for c in next(iter(F.values())).columns if c not in ("sub_temp", "farm")]
    level = {f: float((g.loc[g.in_temp > WARM, "sub_temp"]
                       - g.loc[g.in_temp > WARM, "in_temp_ewm3"]).median())
             for f, g in F.items()}
    usable = [f for f in farms
              if (F[f].in_temp <= COLD).sum() >= MIN_COLD
              and (F[f].in_temp > WARM).sum() >= MIN_WARM]

    WEIGHTS = [0.02, 0.05, 0.10, 0.20]
    rows = []
    for i, X in enumerate(usable, 1):
        g = F[X]
        own = g[g.in_temp > WARM]
        te = g[g.in_temp <= COLD]
        med = own[cols].median()
        Xo, yo = own[cols].fillna(med), own.sub_temp
        Xt, yt = te[cols].fillna(med), te.sub_temp.values

        donors = [f for f in farms if f != X]
        sim = sorted(donors, key=lambda f: abs(level[f] - level[X]))[:10]
        pool = pd.concat([F[f][F[f].in_temp <= DONOR_COLD] for f in sim]).copy()
        pool["sub_temp"] = pool.sub_temp + pool.farm.map(lambda f: level[X] - level[f])
        Xp, yp = pool[cols].fillna(med), pool.sub_temp

        out = {"farm": X, "n_cold": len(te)}
        p = build_mix(Xo, yo, None, Xo, yo, None, Xt)
        out["own"] = float(np.sqrt(np.mean((p - yt) ** 2)))

        Xa, ya = pd.concat([Xo, Xp]), pd.concat([yo, yp])
        for wf in WEIGHTS:
            w = np.r_[np.ones(len(Xo)), np.full(len(Xp), wf * len(Xo) / len(Xp))]
            p = build_mix(Xa, ya, w, Xa, ya, w, Xt)
            out["w%02d" % int(wf * 100)] = float(np.sqrt(np.mean((p - yt) ** 2)))

        # donors inform only the linear baseline; tree stays on own rows
        for wf in (0.20, 1.00):
            w = np.r_[np.ones(len(Xo)), np.full(len(Xp), wf * len(Xo) / len(Xp))]
            p = build_mix(Xa, ya, w, Xo, yo, None, Xt)
            out["lin%03d" % int(wf * 100)] = float(np.sqrt(np.mean((p - yt) ** 2)))
        rows.append(out)
        print("  [%2d/%2d] %s own %.3f | w02 %.3f w10 %.3f | lin100 %.3f"
              % (i, len(usable), X, out["own"], out["w02"], out["w10"],
                 out["lin100"]), flush=True)

    t = pd.DataFrame(rows).set_index("farm")
    t.to_csv("augment_weight.csv")
    rng = np.random.default_rng(0)
    names = [("w02", "기증행 가중 2%"), ("w05", "기증행 가중 5%"),
             ("w10", "기증행 가중 10%"), ("w20", "기증행 가중 20%"),
             ("lin020", "선형만 기증(가중 20%), 트리는 자기것"),
             ("lin100", "선형만 기증(가중 100%), 트리는 자기것")]
    print("\n=== 대상 %d곳, 기준(자기 온실만) RMSE 중앙 %.3f ===" % (len(t), t.own.median()))
    print("  %-34s %9s %8s %22s" % ("구성", "RMSE중앙", "나은곳", "짝지은 차이 [95%]"))
    for k, nm in names:
        dd = (t[k] - t.own).dropna().values
        bs = [np.median(rng.choice(dd, len(dd))) for _ in range(2000)]
        print("  %-34s %9.3f %6d/%d  %+.4f [%+.4f, %+.4f]"
              % (nm, t[k].median(), int((dd < 0).sum()), len(dd),
                 np.median(dd), np.percentile(bs, 2.5), np.percentile(bs, 97.5)))


if __name__ == "__main__":
    main()
