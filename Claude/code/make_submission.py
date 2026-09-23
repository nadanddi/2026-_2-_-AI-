# -*- coding: utf-8 -*-
"""Raw CSVs -> submission.csv, deterministically, with saved models.

    python make_submission.py                # fit everything, write outputs
    python make_submission.py --predict-only # reload saved models, re-predict

Outputs (next to this file's parent folder):
    submission.csv           1,440 rows: row_id, sub_temp, sub_ec
    models/*.joblib          fitted estimators + the exact feature columns
    models/manifest.json     library versions, input-file hashes, parameters

Model (see README.md for the validation story):
    sub_temp  LightGBM, huber loss, 98 causal features incl. the corrected-sign
              screen/radiation physics block, 3 seeds averaged.
    sub_ec    --ec v5      (default, submission 2): ExtraTrees(100 trees,
              min_samples_leaf=2) on 17 raw/clock features, 3 seeds.
              --ec current (submission 1, scored 0.2442): mean of
              ExtraTrees(100, leaf 8) and LightGBM-huber on 68 causal features.
"""
import argparse
import hashlib
import json
import os
import platform
import sys

import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
import sklearn
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

from common import load_raw, DATA, TARGET_FARMS, USABLE
import features_v2 as F2

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT_CSV = os.path.join(ROOT, "submission.csv")
MODEL_DIR = os.path.join(ROOT, "models")

SEEDS = (7, 101, 2024)
DETERMINISTIC = dict(deterministic=True, force_col_wise=True, n_jobs=4, verbose=-1)

T_HUB = dict(objective="huber", n_estimators=1200, learning_rate=0.03,
             num_leaves=63, min_child_samples=40, subsample=0.8,
             subsample_freq=1, colsample_bytree=0.6, reg_lambda=1.0)
E_HUB = dict(objective="huber", n_estimators=400, learning_rate=0.02,
             num_leaves=7, min_child_samples=240, subsample=0.7,
             subsample_freq=1, colsample_bytree=0.4, reg_lambda=5.0)
# 300 trees / leaf 2 scored the same on CV (late 0.3046 vs 0.3004) but weighed
# 472 MB; 100 trees / leaf 8 is 9x smaller with no measured loss (et_size.log).
ET = dict(n_estimators=100, max_features=1.0, min_samples_leaf=8, n_jobs=4)

# --ec v5 : the earlier repo's v5 EC recipe.  Under a CV whose label-gap
# geometry matches the real test (geometry_cv.py: 3.6 days to the nearest
# label, like the test) it scored 0.290 vs 0.365 for the recipe above; leaf
# size matters here (leaf 2 > 4 > 8) and `day` is essential.  17 raw/clock
# features, no history, ExtraTrees only.
V5_ET = dict(n_estimators=100, max_features=1.0, min_samples_leaf=2, n_jobs=4)

# literature-derived block: kept for sub_temp, dropped for sub_ec (CV-decided)
DOMAIN_MARKS = ("rad_eff", "transp_pm", "root_dh", "heat_input", "rtr_",
                "transp_per_rad", "_cum", "screen_ins", "heat_screen")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build(ec_recipe="current"):
    tX, ty, sX = load_raw()
    panel = F2.build(tX, sX)
    panel = panel.merge(ty[["row_id", "sub_temp", "sub_ec"]], on="row_id", how="left")
    panel["is_test"] = panel.row_id.isin(set(sX.row_id))
    panel["midnight"] = (panel.hour == 0).astype(float)
    v_temp = F2.view(panel, "sub_temp")
    if ec_recipe == "v5":
        v_ec = USABLE + ["day", "hr_sin", "hr_cos", "midnight"]
    else:
        v_ec = [c for c in F2.view(panel, "sub_ec")
                if not any(m in c for m in DOMAIN_MARKS)]
    return panel, tX, ty, sX, v_temp, v_ec


def _single_thread_predict(pipe):
    """Parallel tree prediction sums in a thread-dependent order, which can
    flip the 6th decimal between runs; predict single-threaded instead."""
    pipe.steps[-1][1].n_jobs = 1
    return pipe


def fit_all(panel, v_temp, v_ec, ec_recipe="current"):
    lab_t = panel[(~panel.is_test) & panel.sub_temp.notna()]
    lab_e = panel[(~panel.is_test) & panel.sub_ec.notna()]
    packs = {}
    for sd in SEEDS:
        m = lgb.LGBMRegressor(random_state=sd, **DETERMINISTIC, **T_HUB)
        m.fit(lab_t[v_temp], lab_t.sub_temp)
        packs["sub_temp_lgb_%d" % sd] = dict(model=m, columns=v_temp,
                                             target="sub_temp", weight=1.0 / len(SEEDS))
    if ec_recipe == "v5":
        for sd in SEEDS:
            m = make_pipeline(SimpleImputer(strategy="median"),
                              ExtraTreesRegressor(random_state=sd, **V5_ET))
            m.fit(lab_e[v_ec], lab_e.sub_ec)
            _single_thread_predict(m)
            packs["sub_ec_v5et_%d" % sd] = dict(model=m, columns=v_ec,
                                                target="sub_ec", weight=1.0 / len(SEEDS))
        return packs, len(lab_t), len(lab_e)
    for sd in SEEDS:
        m = make_pipeline(SimpleImputer(strategy="median"),
                          ExtraTreesRegressor(random_state=sd, **ET))
        m.fit(lab_e[v_ec], lab_e.sub_ec)
        _single_thread_predict(m)
        packs["sub_ec_et_%d" % sd] = dict(model=m, columns=v_ec,
                                          target="sub_ec", weight=0.5 / len(SEEDS))
        g = lgb.LGBMRegressor(random_state=sd, **DETERMINISTIC, **E_HUB)
        g.fit(lab_e[v_ec], lab_e.sub_ec)
        packs["sub_ec_lgb_%d" % sd] = dict(model=g, columns=v_ec,
                                           target="sub_ec", weight=0.5 / len(SEEDS))
    return packs, len(lab_t), len(lab_e)


def predict(packs, panel, sX, ty):
    test = panel[panel.is_test].set_index("row_id").loc[sX.row_id]
    out = pd.DataFrame({"row_id": sX.row_id.values,
                        "sub_temp": 0.0, "sub_ec": 0.0})
    for p in packs.values():
        out[p["target"]] += p["weight"] * p["model"].predict(test[p["columns"]])
    out["sub_ec"] = out.sub_ec.clip(ty.sub_ec.min(), ty.sub_ec.max())
    return out


def validate(sub, sX):
    assert list(sub.columns) == ["row_id", "sub_temp", "sub_ec"], "column order"
    assert len(sub) == 1440, "row count"
    assert sub.row_id.is_unique, "duplicate row_id"
    assert set(sub.row_id) == set(sX.row_id), "row_id set differs from test_X"
    assert sub.row_id.tolist() == sX.row_id.tolist(), "row_id order differs from test_X"
    vals = sub[["sub_temp", "sub_ec"]].to_numpy(dtype=float)
    assert np.isfinite(vals).all(), "NaN / inf present"
    assert (sub.sub_ec > 0).all(), "non-positive EC"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predict-only", action="store_true")
    ap.add_argument("--ec", choices=["current", "v5"], default="v5",
                    help="EC recipe: 'current' (submission 1) or 'v5' (submission 2)")
    args = ap.parse_args()

    out_csv = OUT_CSV if args.ec == "v5" else OUT_CSV.replace(".csv", "_ec_current.csv")
    model_dir = MODEL_DIR if args.ec == "v5" else MODEL_DIR + "_ec_current"

    panel, tX, ty, sX, v_temp, v_ec = build(args.ec)
    print("ec recipe: %s | features: sub_temp %d | sub_ec %d" % (args.ec, len(v_temp), len(v_ec)))

    if args.predict_only:
        packs = joblib.load(os.path.join(model_dir, "models.joblib"))
        n_t = n_e = None
    else:
        packs, n_t, n_e = fit_all(panel, v_temp, v_ec, args.ec)
        os.makedirs(model_dir, exist_ok=True)
        # ExtraTrees with small leaves grows large trees; compress on disk
        joblib.dump(packs, os.path.join(model_dir, "models.joblib"), compress=("zlib", 6))
        print("fitted %d estimators on %d / %d labelled rows" % (len(packs), n_t, n_e))

    sub = predict(packs, panel, sX, ty)
    validate(sub, sX)
    sub.to_csv(out_csv, index=False, encoding="utf-8", float_format="%.6f")
    print("wrote", out_csv)

    # round-trip: re-read and re-validate exactly what will be uploaded
    back = pd.read_csv(out_csv)
    validate(back, sX)
    print("re-read OK: %d rows, sub_temp [%.3f, %.3f], sub_ec [%.3f, %.3f]"
          % (len(back), back.sub_temp.min(), back.sub_temp.max(),
             back.sub_ec.min(), back.sub_ec.max()))

    if not args.predict_only:
        manifest = dict(
            ec_recipe=args.ec,
            python=platform.python_version(),
            versions=dict(numpy=np.__version__, pandas=pd.__version__,
                          sklearn=sklearn.__version__, lightgbm=lgb.__version__),
            inputs={n: sha256(os.path.join(DATA, n))
                    for n in ["train_X.csv", "train_y.csv", "test_X.csv"]},
            submission_sha256=sha256(out_csv),
            seeds=list(SEEDS),
            params=dict(sub_temp_lgb=T_HUB,
                        sub_ec=(dict(v5_et=V5_ET) if args.ec == "v5"
                                else dict(lgb=E_HUB, et=ET)),
                        lightgbm_common=DETERMINISTIC),
            features=dict(sub_temp=v_temp, sub_ec=v_ec),
            n_train=dict(sub_temp=n_t, sub_ec=n_e),
            ec_clip=[float(ty.sub_ec.min()), float(ty.sub_ec.max())],
            external_data="none", pretrained_models="none",
        )
        with open(os.path.join(model_dir, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        print("wrote", os.path.join(model_dir, "manifest.json"))


if __name__ == "__main__":
    main()
