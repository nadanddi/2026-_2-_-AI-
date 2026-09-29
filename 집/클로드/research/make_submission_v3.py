# -*- coding: utf-8 -*-
"""Raw CSVs -> submissions/submission_03.csv  (deterministic, with manifest)

Configuration, and the geometry-CV evidence behind each choice:

  sub_temp  93 columns = the sub_temp view of features_v2 (98) minus the
            prev_day (18) and mem_long (6) groups, plus the dew (5) and
            event (14) blocks of feat_new.  Blend LGB-huber .65 /
            Ridge(alpha=100) .25 / Nystroem+Ridge .10, three seeds each.
            A 0.8603 -> 0.8212, B 0.7862 -> 0.7671 against the submitted
            model (paired: A CI [-0.0602,-0.0167] P=0.000, 5/5 folds;
            B -0.0191 P=0.053, 5/5 folds).

  sub_ec    14 columns = the v5 recipe minus the four outside-weather
            columns.  Blend ExtraTrees600/leaf1 .60 / LGB-tweedie(1.5) .30
            / MLP128-64 .10, three seeds each, then a causal within-day
            shrink (L=0.5) toward the expanding same-day mean, then clipped
            to the observed training range.
            A 0.2898 -> 0.2718, B 0.2615 -> 0.2376 against the prepared v5
            submission (blend A P=0.004 / B P=0.047; shrink A P=0.013 /
            B P=0.000).

Causality: every input column comes from features_v2 (causality_test.py:
ALL PASS) or feat_new (causality_new.py: ALL PASS).  The shrink at hour h
uses only predictions of hours 0..h of the SAME greenhouse-day, each of
which is a function of inputs at or before its own hour, so it introduces
no dependence on later inputs.

Run:  cd research && PYTHONPATH="" <python> -u make_submission_v3.py
"""
import hashlib
import json
import os
import platform

import env  # noqa: F401  MUST be the first project import
import numpy as np
import pandas as pd
import sklearn
import lightgbm as lgb
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import common
from common import USABLE, OUT_COLS
from harness import load, views
import feat_temp74 as T74
import feat_new

SEEDS = (7, 101, 2024)
DET = dict(deterministic=True, force_col_wise=True, n_jobs=4, verbose=-1)
T_HUB = dict(objective="huber", n_estimators=1200, learning_rate=0.03,
             num_leaves=63, min_child_samples=40, subsample=0.8,
             subsample_freq=1, colsample_bytree=0.6, reg_lambda=1.0)
ET1 = dict(n_estimators=600, max_features=1.0, min_samples_leaf=1, n_jobs=4)
LGBP = dict(n_estimators=800, learning_rate=0.03, num_leaves=31,
            min_child_samples=40, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.8, reg_lambda=1.0)
W_TEMP = dict(lgb=0.65, ridge=0.25, nys=0.10)
W_EC = dict(et=0.60, lgbtw=0.30, mlp=0.10)
SHRINK_L = 0.5
HERE = os.path.dirname(os.path.abspath(__file__))
# In the research tree this writes to research/submissions/.  The reproduction
# package ships an env.py that sets OUTDIR, so the same file works in both
# layouts without being edited.
OUT = getattr(env, "OUTDIR", os.path.join(HERE, "submissions"))


def m_lgbh(s):
    return lgb.LGBMRegressor(random_state=s, **DET, **T_HUB)


def m_ridge(s):
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         Ridge(alpha=100.0))


def m_nys(s):
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         Nystroem(gamma=0.005, n_components=500, random_state=s),
                         Ridge(alpha=1.0))


def m_et(s):
    return make_pipeline(SimpleImputer(strategy="median"),
                         ExtraTreesRegressor(random_state=s, **ET1))


def m_lgbtw(s):
    return lgb.LGBMRegressor(random_state=s, objective="tweedie",
                             tweedie_variance_power=1.5, **DET, **LGBP)


def m_mlp(s):
    return make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler(),
        MLPRegressor(hidden_layer_sizes=(128, 64), alpha=1e-2,
                     learning_rate_init=1e-3, max_iter=800,
                     early_stopping=True, n_iter_no_change=25,
                     validation_fraction=0.12, random_state=s))


def fit_predict(fac, cols, train, target, test):
    """Mean over seeds.  ExtraTrees predicts single-threaded: parallel sums
    reorder floating-point addition and can flip the 6th decimal."""
    ps = []
    for s in SEEDS:
        est = fac(s)
        est.fit(train[cols], train[target].values)
        last = est.steps[-1][1] if hasattr(est, "steps") else est
        if isinstance(last, ExtraTreesRegressor):
            last.n_jobs = 1
        ps.append(np.asarray(est.predict(test[cols]), float))
    return np.mean(ps, axis=0)


def causal_shrink(p, frame, L):
    d = pd.DataFrame({"f": frame.farm.values, "d": frame.day.values,
                      "h": frame.hour.values, "p": p})
    d = d.sort_values(["f", "d", "h"])
    em = d.groupby(["f", "d"]).p.transform(lambda s: s.expanding().mean())
    out = np.empty(len(p))
    out[d.index.values] = (em + L * (d.p - em)).values
    return out


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main():
    os.makedirs(OUT, exist_ok=True)
    panel, lab_t0, lab_e0 = load()
    tX, ty, sX = common.load_raw()
    v = views(panel)

    ex, blocks = feat_new.build_extra()
    cols93 = T74.base74(v["temp"]) + list(blocks["dew"]) + list(blocks["event"])
    cols14 = [c for c in (list(USABLE) + ["day", "hr_sin", "hr_cos", "midnight"])
              if c not in OUT_COLS]
    print("sub_temp %d cols | sub_ec %d cols" % (len(cols93), len(cols14)))

    lab_t = lab_t0.merge(ex, on="row_id", how="left")
    test = panel[panel.is_test].merge(ex, on="row_id", how="left")
    test = test.set_index("row_id").loc[sX.row_id].reset_index()
    print("train temp %d | train ec %d | test %d"
          % (len(lab_t), len(lab_e0), len(test)))

    pt = (W_TEMP["lgb"] * fit_predict(m_lgbh, cols93, lab_t, "sub_temp", test)
          + W_TEMP["ridge"] * fit_predict(m_ridge, cols93, lab_t, "sub_temp", test)
          + W_TEMP["nys"] * fit_predict(m_nys, cols93, lab_t, "sub_temp", test))
    print("sub_temp done")

    pe = (W_EC["et"] * fit_predict(m_et, cols14, lab_e0, "sub_ec", test)
          + W_EC["lgbtw"] * fit_predict(m_lgbtw, cols14, lab_e0, "sub_ec", test)
          + W_EC["mlp"] * fit_predict(m_mlp, cols14, lab_e0, "sub_ec", test))
    pe_raw = pe.copy()
    pe = causal_shrink(pe, test, SHRINK_L)
    lo, hi = float(ty.sub_ec.min()), float(ty.sub_ec.max())
    pe = np.clip(pe, lo, hi)
    print("sub_ec done (shrink L=%.1f, clip [%.3f, %.3f], shrink moved RMS %.4f)"
          % (SHRINK_L, lo, hi, float(np.sqrt(((pe - pe_raw) ** 2).mean()))))

    sub = pd.DataFrame({"row_id": sX.row_id.values, "sub_temp": pt, "sub_ec": pe})

    assert list(sub.columns) == ["row_id", "sub_temp", "sub_ec"], "column order"
    assert len(sub) == 1440, "row count"
    assert sub.row_id.is_unique, "duplicate row_id"
    assert sub.row_id.tolist() == sX.row_id.tolist(), "row_id order != test_X"
    vals = sub[["sub_temp", "sub_ec"]].to_numpy(float)
    assert np.isfinite(vals).all(), "NaN or inf present"
    assert (sub.sub_ec > 0).all(), "non-positive EC"
    ss = os.path.join(common.DATA, "sample_submission.csv")
    if os.path.exists(ss):
        assert pd.read_csv(ss).row_id.tolist() == sub.row_id.tolist(), \
            "row_id order != sample_submission"
        print("sample_submission order: match")

    path = os.path.join(OUT, "submission_03.csv")
    sub.to_csv(path, index=False, encoding="utf-8", float_format="%.6f")
    back = pd.read_csv(path)
    assert back.row_id.tolist() == sX.row_id.tolist() and len(back) == 1440
    print("wrote %s" % path)
    print("  sub_temp [%.3f, %.3f] mean %.3f"
          % (back.sub_temp.min(), back.sub_temp.max(), back.sub_temp.mean()))
    print("  sub_ec   [%.3f, %.3f] mean %.3f"
          % (back.sub_ec.min(), back.sub_ec.max(), back.sub_ec.mean()))

    man = dict(
        name="submission_03",
        python=platform.python_version(),
        versions=dict(numpy=np.__version__, pandas=pd.__version__,
                      sklearn=sklearn.__version__, lightgbm=lgb.__version__),
        inputs={n: sha256(os.path.join(common.DATA, n))
                for n in ["train_X.csv", "train_y.csv", "test_X.csv"]},
        submission_sha256=sha256(path),
        seeds=list(SEEDS),
        sub_temp=dict(n_features=len(cols93), features=cols93, weights=W_TEMP,
                      lgb_params=T_HUB, ridge_alpha=100.0,
                      nystroem=dict(gamma=0.005, n_components=500),
                      cv_geometry_A=0.8212, cv_geometry_B=0.7671),
        sub_ec=dict(n_features=len(cols14), features=cols14, weights=W_EC,
                    et_params=ET1,
                    lgb_params=dict(objective="tweedie",
                                    tweedie_variance_power=1.5, **LGBP),
                    mlp=dict(hidden_layer_sizes=[128, 64], alpha=1e-2),
                    shrink_lambda=SHRINK_L, clip=[lo, hi],
                    cv_geometry_A=0.2718, cv_geometry_B=0.2376),
        n_train=dict(sub_temp=len(lab_t), sub_ec=len(lab_e0)),
        external_data="none", pretrained_models="none",
        note=("Environment differs from submission_01 (py 3.7.9 / sklearn "
              "1.0.2 / lightgbm 4.6.0). This file is reproducible under the "
              "versions recorded above, and is not byte-identical to the "
              "earlier submissions."),
    )
    mp = os.path.join(OUT, "manifest_03.json")
    with open(mp, "w", encoding="utf-8") as f:
        json.dump(man, f, ensure_ascii=False, indent=2)
    print("wrote %s" % mp)


if __name__ == "__main__":
    main()
