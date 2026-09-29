# -*- coding: utf-8 -*-
"""Which CV predicts the leaderboard?  Calibrate against two real submissions.

Real scores (platform):
  submission 1  sub_temp 0.7450  sub_ec 0.2442
  submission 2  sub_temp 0.6666  sub_ec 0.2287      ratio 0.895 / 0.937

The geometry CV (label-gap matched, days 63-180) predicted ratios 0.955 /
0.745 on placement A: it understated the temperature gain 2.3x and overstated
the EC gain 4x.  The test lies in days 183-239, which the geometry folds never
touch.  For sub_temp the day-level error has ~0 autocorrelation beyond 3 days
(struct_resid.py), so label-gap fidelity matters little there while period
fidelity may matter a lot.

Candidate CV here: hold out LATE labelled days (>= LATE_FROM) in contiguous
chunks, 1-day buffer, train on everything else.  If its ratios land nearer
0.895 / 0.937 it is the better tool for the next decisions.

Run:  cd research && PYTHONPATH="" <python> -u calib_cv.py
"""
import env  # noqa: F401
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import split_mask, rmse, USABLE, OUT_COLS, TARGET_FARMS
from harness import load, views, folds, DOMAIN_MARKS
import feat_temp74 as T74
import feat_new

LATE_FROM = 170
N_CHUNK = 5
SEED = (7,)
DET = dict(deterministic=True, force_col_wise=True, n_jobs=4, verbose=-1)
T_HUB = dict(objective="huber", n_estimators=1200, learning_rate=0.03,
             num_leaves=63, min_child_samples=40, subsample=0.8,
             subsample_freq=1, colsample_bytree=0.6, reg_lambda=1.0)
E_HUB = dict(objective="huber", n_estimators=400, learning_rate=0.02,
             num_leaves=7, min_child_samples=240, subsample=0.7,
             subsample_freq=1, colsample_bytree=0.4, reg_lambda=5.0)
ET8 = dict(n_estimators=100, max_features=1.0, min_samples_leaf=8, n_jobs=4)
ET1 = dict(n_estimators=600, max_features=1.0, min_samples_leaf=1, n_jobs=4)
LGBP = dict(n_estimators=800, learning_rate=0.03, num_leaves=31,
            min_child_samples=40, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.8, reg_lambda=1.0)


def late_folds(lab):
    out = []
    per = {}
    for f in TARGET_FARMS:
        days = np.array(sorted(lab[(lab.farm == f) & (lab.day >= LATE_FROM)].day.unique()))
        per[f] = np.array_split(days, N_CHUNK)
    for k in range(N_CHUNK):
        out.append({f: set(int(x) for x in per[f][k]) for f in TARGET_FARMS})
    return out


def fit_pred(parts, cols, tr, va, target):
    p = np.zeros(len(va))
    for w, fac in parts:
        ps = []
        for s in SEED:
            m = fac(s)
            m.fit(tr[cols], tr[target].values)
            ps.append(np.asarray(m.predict(va[cols]), float))
        p += w * np.mean(ps, axis=0)
    return p


def shrink(p, fr, L):
    d = pd.DataFrame({"f": fr.farm.values, "d": fr.day.values, "h": fr.hour.values,
                      "p": p}).reset_index(drop=True).sort_values(["f", "d", "h"])
    em = d.groupby(["f", "d"]).p.transform(lambda s: s.expanding().mean())
    o = np.empty(len(p))
    o[d.index.values] = (em + L * (d.p - em)).values
    return o


def run(lab, target, cols, parts, fds, post=None):
    y = lab[target].values
    oof = np.full(len(lab), np.nan)
    for fd in fds:
        trm, vam = split_mask(lab, fd)
        va = lab[vam]
        p = fit_pred(parts, cols, lab[trm], va, target)
        if post:
            p = post(p, va)
        oof[np.where(vam)[0]] = p
    g = ~np.isnan(oof)
    return rmse(oof[g], y[g])


def main():
    panel, lab_t0, lab_e = load()
    v = views(panel)
    ex, blocks = feat_new.build_extra()
    lab_t = lab_t0.merge(ex, on="row_id", how="left")

    f98 = v["temp"]
    f93 = T74.base74(f98) + list(blocks["dew"]) + list(blocks["event"])
    f68 = [c for c in v["ec_full"] if not any(m in c for m in DOMAIN_MARKS)]
    f14 = [c for c in (list(USABLE) + ["day", "hr_sin", "hr_cos", "midnight"])
           if c not in OUT_COLS]

    lgbh = lambda s: lgb.LGBMRegressor(random_state=s, **DET, **T_HUB)
    ridge = lambda s: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=100.0))
    nys = lambda s: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                  Nystroem(gamma=0.005, n_components=500, random_state=s), Ridge(alpha=1.0))
    et8 = lambda s: make_pipeline(SimpleImputer(strategy="median"), ExtraTreesRegressor(random_state=s, **ET8))
    ehub = lambda s: lgb.LGBMRegressor(random_state=s, **DET, **E_HUB)
    et1 = lambda s: make_pipeline(SimpleImputer(strategy="median"), ExtraTreesRegressor(random_state=s, **ET1))
    ltw = lambda s: lgb.LGBMRegressor(random_state=s, objective="tweedie",
                                      tweedie_variance_power=1.5, **DET, **LGBP)
    mlp = lambda s: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                  MLPRegressor(hidden_layer_sizes=(128, 64), alpha=1e-2,
                                               learning_rate_init=1e-3, max_iter=800,
                                               early_stopping=True, n_iter_no_change=25,
                                               validation_fraction=0.12, random_state=s))

    cfg = {
        "sub_temp": (lab_t, [("sub1", f98, [(1.0, lgbh)], None),
                             ("sub2", f93, [(0.65, lgbh), (0.25, ridge), (0.10, nys)], None)]),
        "sub_ec": (lab_e, [("sub1", f68, [(0.5, et8), (0.5, ehub)], None),
                           ("sub2", f14, [(0.60, et1), (0.30, ltw), (0.10, mlp)],
                            lambda p, fr: np.clip(shrink(p, fr, 0.5), 0.062, 3.46))]),
    }
    real = {"sub_temp": (0.7450, 0.6666), "sub_ec": (0.2442, 0.2287)}

    lf = late_folds(lab_e)
    gaps = []
    for fd in lf:
        tr, _ = split_mask(lab_e, fd)
        for f, vd in fd.items():
            td = np.array(sorted(lab_e[tr & (lab_e.farm == f).values].day.unique()))
            gaps += [np.abs(td - d).min() for d in vd]
    print("late folds: days >= %d, %d chunks, held-out day -> nearest label mean %.1f"
          % (LATE_FROM, N_CHUNK, float(np.mean(gaps))))

    for target, (lab, pair) in cfg.items():
        r1, r2 = real[target]
        print("\n######## %s  실제 %.4f -> %.4f  (비율 %.3f) ########"
              % (target, r1, r2, r2 / r1))
        for fname, fds in (("geometry A", folds("A")), ("geometry B", folds("B")),
                           ("late", lf)):
            s = [run(lab, target, cols, parts, fds, post) for _, cols, parts, post in pair]
            print("  %-11s  sub1 %.4f  sub2 %.4f  비율 %.3f  (실제와 차이 %+.3f)"
                  % (fname, s[0], s[1], s[1] / s[0], s[1] / s[0] - r2 / r1))


if __name__ == "__main__":
    main()
