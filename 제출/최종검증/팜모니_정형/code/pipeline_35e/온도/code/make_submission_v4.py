# -*- coding: utf-8 -*-
"""Raw CSVs -> submission_04.csv  (candidate for platform submission 3)

sub_temp  columns = submission-2's 93 + day-restarted filters (18) +
          source fingerprint (24).  Blend:
            .65  physics-linear baseline + LGB-huber on the residual
            .25  Ridge(alpha=100)
            .10  Nystroem + Ridge
          geometry CV vs submission-2 config: A 0.8211 -> 0.7831 (-4.6%,
          5/5 folds, CI [-0.079,-0.0008]); B 0.7718 -> 0.7419 (-3.9%, 5/5,
          CI [-0.054,-0.0065])   (combine_v4.py, seed 7)

sub_ec    submission-2 recipe with the fingerprint added to the ExtraTrees
          member only: ET(14 + fp) .60 / LGB-tweedie(14) .30 / MLP(14) .10,
          causal within-day shrink L=0.5, clip.
          A 0.2725 -> 0.2555 (-6.2%, 5/5); B 0.2372 -> 0.2385 (+0.5%)

The two targets are scored separately, so one upload measures both changes.

Run:  cd research && PYTHONPATH="" <python> -u make_submission_v4.py
"""
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
from sklearn.linear_model import LinearRegression

import common
from common import USABLE, OUT_COLS
from harness import load, views
import feat_temp74 as T74
import feat_new
import features_v4 as F4
from make_submission_v3 import (SEEDS, T_HUB, ET1, LGBP, SHRINK_L, m_lgbh, m_ridge,
                                m_nys, m_et, m_lgbtw, m_mlp, causal_shrink, sha256)

W_TEMP = dict(resid_lgb=0.65, ridge=0.25, nys=0.10)
W_EC = dict(et=0.60, lgbtw=0.30, mlp=0.10)
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = getattr(env, "OUTDIR", os.path.join(HERE, "submissions"))


def fit_mean(fac, X, y, Xt):
    ps = []
    for s in SEEDS:
        est = fac(s)
        est.fit(X, y)
        last = est.steps[-1][1] if hasattr(est, "steps") else est
        if isinstance(last, ExtraTreesRegressor):
            last.n_jobs = 1
        ps.append(np.asarray(est.predict(Xt), float))
    return np.mean(ps, axis=0)


def main():
    os.makedirs(OUT, exist_ok=True)
    panel, lab_t0, lab_e0 = load()
    tX, ty, sX = common.load_raw()
    v = views(panel)

    ex, blocks = feat_new.build_extra()
    sg, ph, fp = F4.seg_features(), F4.phys_features(), F4.fp_features()
    segc, phc, fpc = F4.names(sg), F4.names(ph), F4.names(fp)
    f93 = T74.base74(v["temp"]) + list(blocks["dew"]) + list(blocks["event"])
    ct = f93 + segc + fpc
    f14 = [c for c in (list(USABLE) + ["day", "hr_sin", "hr_cos", "midnight"]) if c not in OUT_COLS]
    ce_et = f14 + fpc

    def attach(df):
        return (df.merge(ex, on="row_id", how="left").merge(sg, on="row_id", how="left")
                  .merge(ph, on="row_id", how="left").merge(fp, on="row_id", how="left"))

    lab_t = attach(lab_t0)
    lab_e = lab_e0.merge(fp, on="row_id", how="left")
    test = attach(panel[panel.is_test].reset_index(drop=True))
    test = test.set_index("row_id").loc[sX.row_id].reset_index()
    print("sub_temp %d cols (+%d phys) | sub_ec ET %d / others %d | test %d"
          % (len(ct), len(phc), len(ce_et), len(f14), len(test)))

    # ---- sub_temp -----------------------------------------------------------
    imp = SimpleImputer(strategy="median").fit(lab_t[phc])
    base = LinearRegression().fit(imp.transform(lab_t[phc]), lab_t.sub_temp.values)
    b_tr = base.predict(imp.transform(lab_t[phc]))
    b_te = base.predict(imp.transform(test[phc]))
    y = lab_t.sub_temp.values
    pt = (W_TEMP["resid_lgb"] * (b_te + fit_mean(m_lgbh, lab_t[ct], y - b_tr, test[ct]))
          + W_TEMP["ridge"] * fit_mean(m_ridge, lab_t[ct], y, test[ct])
          + W_TEMP["nys"] * fit_mean(m_nys, lab_t[ct], y, test[ct]))
    print("sub_temp done")

    # ---- sub_ec -------------------------------------------------------------
    ye = lab_e.sub_ec.values
    pe = (W_EC["et"] * fit_mean(m_et, lab_e[ce_et], ye, test[ce_et])
          + W_EC["lgbtw"] * fit_mean(m_lgbtw, lab_e[f14], ye, test[f14])
          + W_EC["mlp"] * fit_mean(m_mlp, lab_e[f14], ye, test[f14]))
    pe = causal_shrink(pe, test, SHRINK_L)
    lo, hi = float(ty.sub_ec.min()), float(ty.sub_ec.max())
    pe = np.clip(pe, lo, hi)
    print("sub_ec done")

    sub = pd.DataFrame({"row_id": sX.row_id.values, "sub_temp": pt, "sub_ec": pe})
    assert list(sub.columns) == ["row_id", "sub_temp", "sub_ec"]
    assert len(sub) == 1440 and sub.row_id.is_unique
    assert sub.row_id.tolist() == sX.row_id.tolist()
    assert np.isfinite(sub[["sub_temp", "sub_ec"]].to_numpy(float)).all()
    assert (sub.sub_ec > 0).all()
    ss = os.path.join(common.DATA, "sample_submission.csv")
    if os.path.exists(ss):
        assert pd.read_csv(ss).row_id.tolist() == sub.row_id.tolist()
        print("sample_submission order: match")

    path = os.path.join(OUT, "submission_04.csv")
    sub.to_csv(path, index=False, encoding="utf-8", float_format="%.6f")
    back = pd.read_csv(path)
    assert back.row_id.tolist() == sX.row_id.tolist() and len(back) == 1440
    print("wrote %s" % path)
    print("  sub_temp [%.3f, %.3f] mean %.3f" % (back.sub_temp.min(), back.sub_temp.max(), back.sub_temp.mean()))
    print("  sub_ec   [%.3f, %.3f] mean %.3f" % (back.sub_ec.min(), back.sub_ec.max(), back.sub_ec.mean()))

    man = dict(
        name="submission_04", python=platform.python_version(),
        versions=dict(numpy=np.__version__, pandas=pd.__version__,
                      sklearn=sklearn.__version__, lightgbm=lgb.__version__),
        inputs={n: sha256(os.path.join(common.DATA, n))
                for n in ["train_X.csv", "train_y.csv", "test_X.csv"]},
        submission_sha256=sha256(path), seeds=list(SEEDS),
        sub_temp=dict(n_features=len(ct), features=ct, phys_features=phc, weights=W_TEMP,
                      lgb_params=T_HUB, ridge_alpha=100.0,
                      nystroem=dict(gamma=0.005, n_components=500),
                      target="residual over in-sample linear physics baseline (lgb member)",
                      cv_geometry_A=0.7831, cv_geometry_B=0.7419),
        sub_ec=dict(features_et=ce_et, features_other=f14, weights=W_EC, et_params=ET1,
                    lgb_params=dict(objective="tweedie", tweedie_variance_power=1.5, **LGBP),
                    mlp=dict(hidden_layer_sizes=[128, 64], alpha=1e-2),
                    shrink_lambda=SHRINK_L, clip=[lo, hi],
                    cv_geometry_A=0.2555, cv_geometry_B=0.2385),
        n_train=dict(sub_temp=len(lab_t), sub_ec=len(lab_e)),
        external_data="none", pretrained_models="none",
    )
    with open(os.path.join(OUT, "manifest_04.json"), "w", encoding="utf-8") as f:
        json.dump(man, f, ensure_ascii=False, indent=2)
    print("wrote manifest_04.json")


if __name__ == "__main__":
    main()
