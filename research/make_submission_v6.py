# -*- coding: utf-8 -*-
"""Raw CSVs -> submission_06.csv  (platform submission round 5, an experiment)

Two independent experiments in one upload (the targets are scored
separately):

sub_temp  round-3 configuration (135 columns; 0.65 physics-linear baseline +
          LGB-huber residual, 0.25 Ridge, 0.10 Nystroem) trained with row
          weights from train_flags_v6.py:
            * 60 restored training rows (night in_temp < out_temp - 3, in_temp
              stuck >= 6 h counted backwards, non-midnight night CO2 jump
              without supply) -> 0.2
            * 96 noisy training days (collapsed autocorrelation of hourly CO2
              differences + rough CO2, ranked among training days only; days
              with cold rows exempt) -> 0.2
          Weights use TRAINING inputs only, looking backwards; no test input
          and no label is involved (audit: research/감사결과_2026-09-26.md).
          eval_v6.py (non-overlapping folds, all rows), vs round 3:
            diagnostic -1.4% [CI excl. 0], EXTRAP8 -0.2%, EXTRAP10 -3.1%,
            EXTRAP12 -4.4%; never worse.

sub_ec    round-3 recipe with `day` removed from the ExtraTrees member only.
          The second-pass validator said -20%, geometry A said +7%; only the
          leaderboard can settle it.

Output goes to submissions/ but the script refuses to overwrite an existing
file (an earlier generator overwrote an uploaded answer in place).

Run:  cd research && PYTHONPATH="" <python> -u make_submission_v6.py
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
import train_flags_v6 as TF
from make_submission_v3 import (SEEDS, T_HUB, ET1, LGBP, SHRINK_L, m_lgbh, m_ridge,
                                m_nys, m_et, m_lgbtw, m_mlp, causal_shrink, sha256)
from make_submission_v4 import W_EC

W_TEMP = dict(resid_lgb=0.65, ridge=0.25, nys=0.10)
W_FLAG, W_NOISY = 0.2, 0.2
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = getattr(env, "OUTDIR", os.path.join(HERE, "submissions"))
NAME = "submission_06"


def fit_mean_w(fac, X, y, Xt, w=None, step=None):
    ps = []
    for s in SEEDS:
        est = fac(s)
        kw = {} if w is None else ({"sample_weight": w} if step is None else {step + "__sample_weight": w})
        est.fit(X, y, **kw)
        last = est.steps[-1][1] if hasattr(est, "steps") else est
        if isinstance(last, ExtraTreesRegressor):
            last.n_jobs = 1
        ps.append(np.asarray(est.predict(Xt), float))
    return np.mean(ps, axis=0)


def main():
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, NAME + ".csv")
    mpath = os.path.join(OUT, NAME.replace("submission", "manifest") + ".json")
    for p in (path, mpath):
        assert not os.path.exists(p), "refusing to overwrite %s" % p

    panel, lab_t0, lab_e0 = load()
    tX, ty, sX = common.load_raw()
    v = views(panel)
    ex, blocks = feat_new.build_extra()
    sg, ph, fp = F4.seg_features(), F4.phys_features(), F4.fp_features()
    segc, phc, fpc = F4.names(sg), F4.names(ph), F4.names(fp)
    f93 = T74.base74(v["temp"]) + list(blocks["dew"]) + list(blocks["event"])
    ct = f93 + segc + fpc
    f14 = [c for c in (list(USABLE) + ["day", "hr_sin", "hr_cos", "midnight"]) if c not in OUT_COLS]
    f13 = [c for c in f14 if c != "day"]
    ce_et = f13 + fpc

    def attach(df):
        return (df.merge(ex, on="row_id", how="left").merge(sg, on="row_id", how="left")
                  .merge(ph, on="row_id", how="left").merge(fp, on="row_id", how="left"))

    lab_t = attach(lab_t0)
    lab_e = lab_e0.merge(fp, on="row_id", how="left")
    test = attach(panel[panel.is_test].reset_index(drop=True))
    test = test.set_index("row_id").loc[sX.row_id].reset_index()

    w = TF.row_weights(lab_t, W_FLAG, w_noisy=W_NOISY)
    print("sub_temp %d cols | down-weighted training rows %d of %d | test %d"
          % (len(ct), int((w < 1).sum()), len(w), len(test)))

    # ---- sub_temp -----------------------------------------------------------
    y = lab_t.sub_temp.values
    imp = SimpleImputer(strategy="median").fit(lab_t[phc])
    base = LinearRegression().fit(imp.transform(lab_t[phc]), y, sample_weight=w)
    b_tr = base.predict(imp.transform(lab_t[phc]))
    b_te = base.predict(imp.transform(test[phc]))
    pt = (W_TEMP["resid_lgb"] * (b_te + fit_mean_w(m_lgbh, lab_t[ct], y - b_tr, test[ct], w))
          + W_TEMP["ridge"] * fit_mean_w(m_ridge, lab_t[ct], y, test[ct], w, "ridge")
          + W_TEMP["nys"] * fit_mean_w(m_nys, lab_t[ct], y, test[ct], w, "ridge"))
    print("sub_temp done")

    # ---- sub_ec -------------------------------------------------------------
    ye = lab_e.sub_ec.values
    pe = (W_EC["et"] * fit_mean_w(m_et, lab_e[ce_et], ye, test[ce_et])
          + W_EC["lgbtw"] * fit_mean_w(m_lgbtw, lab_e[f14], ye, test[f14])
          + W_EC["mlp"] * fit_mean_w(m_mlp, lab_e[f14], ye, test[f14]))
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

    sub.to_csv(path, index=False, encoding="utf-8", float_format="%.6f")
    back = pd.read_csv(path)
    assert back.row_id.tolist() == sX.row_id.tolist() and len(back) == 1440
    print("wrote %s" % path)
    ref = os.path.join(HERE, "submissions", "submission_04.csv")
    if not os.path.exists(ref):
        ref = os.path.join(os.path.dirname(HERE), "reference", "submission_04.csv")
    if os.path.exists(ref):
        r3 = pd.read_csv(ref)
        dt, de = back.sub_temp - r3.sub_temp, back.sub_ec - r3.sub_ec
        print("  change vs round 3: temp RMS %.3f (min %+.3f, max %+.3f) | ec RMS %.4f"
              % (float(np.sqrt((dt ** 2).mean())), dt.min(), dt.max(), float(np.sqrt((de ** 2).mean()))))

    man = dict(
        name=NAME, platform_round=5, python=platform.python_version(),
        versions=dict(numpy=np.__version__, pandas=pd.__version__,
                      sklearn=sklearn.__version__, lightgbm=lgb.__version__),
        inputs={n: sha256(os.path.join(common.DATA, n)) for n in ["train_X.csv", "train_y.csv", "test_X.csv"]},
        submission_sha256=sha256(path), seeds=list(SEEDS),
        line_endings="CRLF on Windows (pandas default); normalise line endings before comparing hashes elsewhere",
        sub_temp=dict(features=ct, phys_features=phc, member_weights=W_TEMP, lgb_params=T_HUB,
                      training_row_weights=dict(restored_rows=W_FLAG, noisy_days=W_NOISY,
                                                n_downweighted=int((w < 1).sum()),
                                                source="train_flags_v6.py (training inputs only, backward-looking)")),
        sub_ec=dict(features_et=ce_et, features_other=f14, weights=W_EC, et_params=ET1,
                    lgb_params=dict(objective="tweedie", tweedie_variance_power=1.5, **LGBP),
                    shrink_lambda=SHRINK_L, clip=[lo, hi], note="`day` removed from the ExtraTrees member only"),
        external_data="none", pretrained_models="none",
    )
    with open(mpath, "w", encoding="utf-8") as f:
        json.dump(man, f, ensure_ascii=False, indent=2)
    print("wrote %s" % os.path.basename(mpath))


if __name__ == "__main__":
    main()
