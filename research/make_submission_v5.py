# -*- coding: utf-8 -*-
"""Raw CSVs -> submission_05.csv  (candidate for platform submission round 4)

Only sub_temp changes; sub_ec is the round-3 recipe unchanged, so the round-4
score isolates the effect of the cold-end change.

sub_temp  = 0.5 * round-3 blend  +  0.5 * hinge blend
  round-3 blend: 0.65 (linear physics baseline + LGB-huber on the residual)
                 + 0.25 Ridge + 0.10 Nystroem, 135 columns       (= v4)
  hinge blend:   identical, but the linear baseline also gets cold hinges
                 max(0, k - x), k = 8/10/12 C, x = ewm3 and raw in_temp
Validation (cold_v5b.py, seed 7), vs round 3:
  extrapolation hold-out < 8 C (264 rows): 1.0028 -> 0.9510, cold rows
      (< 8 C) 0.899 -> 0.785, bias +0.54 -> +0.22
  extrapolation hold-out < 7 C (72 rows):  0.7259 -> 0.7551 (small sample)
  geometry A / B: 0.7850 -> 0.7825 / 0.7434 -> 0.7410
The extrapolation validator reproduced the real round-2 -> round-3 ratio
(0.858 vs 0.844); the geometry CV did not (0.956).

Self-checks: the round-3 half of the temperature blend and the whole sub_ec
column must equal submission_04.csv bit for bit.

Run:  cd research && PYTHONPATH="" <python> -u make_submission_v5.py
"""
import json
import os
import platform

import env  # noqa: F401  MUST be the first project import
import numpy as np
import pandas as pd
import sklearn
import lightgbm as lgb
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression

import common
from common import USABLE, OUT_COLS
from harness import load, views
import feat_temp74 as T74
import feat_new
import features_v5 as F5
from make_submission_v3 import (SEEDS, T_HUB, ET1, LGBP, SHRINK_L, m_lgbh, m_ridge,
                                m_nys, m_et, m_lgbtw, m_mlp, causal_shrink, sha256)
from make_submission_v4 import fit_mean, W_EC

W_TEMP_MEMBERS = dict(resid_lgb=0.65, ridge=0.25, nys=0.10)
W_HALF = 0.5
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = getattr(env, "OUTDIR", os.path.join(HERE, "submissions"))


def main():
    os.makedirs(OUT, exist_ok=True)
    panel, lab_t0, lab_e0 = load()
    tX, ty, sX = common.load_raw()
    v = views(panel)

    ex, blocks = feat_new.build_extra()
    sg, ph, fp = F5.seg_features(), F5.phys_features(), F5.fp_features()
    hg = F5.hinge_features(ph)
    segc, phc, fpc, hgc = F5.names(sg), F5.names(ph), F5.names(fp), F5.names(hg)
    f93 = T74.base74(v["temp"]) + list(blocks["dew"]) + list(blocks["event"])
    ct = f93 + segc + fpc
    f14 = [c for c in (list(USABLE) + ["day", "hr_sin", "hr_cos", "midnight"]) if c not in OUT_COLS]
    ce_et = f14 + fpc

    def attach(df):
        return (df.merge(ex, on="row_id", how="left").merge(sg, on="row_id", how="left")
                  .merge(ph, on="row_id", how="left").merge(fp, on="row_id", how="left")
                  .merge(hg, on="row_id", how="left"))

    lab_t = attach(lab_t0)
    lab_e = lab_e0.merge(fp, on="row_id", how="left")
    test = attach(panel[panel.is_test].reset_index(drop=True))
    test = test.set_index("row_id").loc[sX.row_id].reset_index()
    print("sub_temp %d cols | baseline %d phys + %d hinge | test %d"
          % (len(ct), len(phc), len(hgc), len(test)))

    # ---- sub_temp -----------------------------------------------------------
    y = lab_t.sub_temp.values
    p_ridge = fit_mean(m_ridge, lab_t[ct], y, test[ct])
    p_nys = fit_mean(m_nys, lab_t[ct], y, test[ct])

    def resid_member(bcols):
        imp = SimpleImputer(strategy="median").fit(lab_t[bcols])
        base = LinearRegression().fit(imp.transform(lab_t[bcols]), y)
        b_tr = base.predict(imp.transform(lab_t[bcols]))
        b_te = base.predict(imp.transform(test[bcols]))
        return b_te + fit_mean(m_lgbh, lab_t[ct], y - b_tr, test[ct])

    # same addition order as make_submission_v4 so the round-3 half is bit-equal
    W = W_TEMP_MEMBERS
    pt_r3 = W["resid_lgb"] * resid_member(phc) + W["ridge"] * p_ridge + W["nys"] * p_nys
    pt_hg = W["resid_lgb"] * resid_member(phc + hgc) + W["ridge"] * p_ridge + W["nys"] * p_nys
    pt = (1 - W_HALF) * pt_r3 + W_HALF * pt_hg
    print("sub_temp done")

    # ---- sub_ec (round 3 unchanged) ------------------------------------------
    ye = lab_e.sub_ec.values
    pe = (W_EC["et"] * fit_mean(m_et, lab_e[ce_et], ye, test[ce_et])
          + W_EC["lgbtw"] * fit_mean(m_lgbtw, lab_e[f14], ye, test[f14])
          + W_EC["mlp"] * fit_mean(m_mlp, lab_e[f14], ye, test[f14]))
    pe = causal_shrink(pe, test, SHRINK_L)
    lo, hi = float(ty.sub_ec.min()), float(ty.sub_ec.max())
    pe = np.clip(pe, lo, hi)
    print("sub_ec done")

    # ---- self-checks against round 3 ------------------------------------------
    # research tree: submissions/submission_04.csv; package: reference/submission_04.csv
    ref_paths = [os.path.join(HERE, "submissions", "submission_04.csv"),
                 os.path.join(os.path.dirname(HERE), "reference", "submission_04.csv")]
    ref = next((p for p in ref_paths if os.path.exists(p)), None)
    r3 = pd.read_csv(ref) if ref else None
    if r3 is not None:
        fmt = lambda a: np.array(["%.6f" % x for x in a])
        same_ec = bool((fmt(pe) == r3.sub_ec.map(lambda x: "%.6f" % x).values).all())
        same_t3 = bool((fmt(pt_r3) == r3.sub_temp.map(lambda x: "%.6f" % x).values).all())
        print("self-check vs round 3 (%s): sub_ec equal %s | round-3 half of sub_temp equal %s"
              % (os.path.basename(os.path.dirname(ref)), same_ec, same_t3))
        assert same_ec and same_t3, "round-3 components did not reproduce"
    else:
        print("self-check skipped: submission_04.csv reference not found")

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

    path = os.path.join(OUT, "submission_05.csv")
    sub.to_csv(path, index=False, encoding="utf-8", float_format="%.6f")
    back = pd.read_csv(path)
    assert back.row_id.tolist() == sX.row_id.tolist() and len(back) == 1440
    print("wrote %s" % path)
    print("  sub_temp [%.3f, %.3f] mean %.3f"
          % (back.sub_temp.min(), back.sub_temp.max(), back.sub_temp.mean()))
    if r3 is not None:
        d = back.sub_temp.values - r3.sub_temp.values
        print("  change vs round 3: RMS %.3f, min %+.3f, max %+.3f"
              % (float(np.sqrt((d ** 2).mean())), d.min(), d.max()))

    man = dict(
        name="submission_05", platform_round=4, python=platform.python_version(),
        versions=dict(numpy=np.__version__, pandas=pd.__version__,
                      sklearn=sklearn.__version__, lightgbm=lgb.__version__),
        inputs={n: sha256(os.path.join(common.DATA, n))
                for n in ["train_X.csv", "train_y.csv", "test_X.csv"]},
        submission_sha256=sha256(path), seeds=list(SEEDS),
        sub_temp=dict(n_features=len(ct), features=ct, phys_features=phc, hinge_features=hgc,
                      member_weights=W_TEMP_MEMBERS, half_weight_on_hinge_blend=W_HALF,
                      lgb_params=T_HUB, ridge_alpha=100.0,
                      nystroem=dict(gamma=0.005, n_components=500)),
        sub_ec=dict(note="identical to submission_04 (platform round 3)",
                    features_et=ce_et, features_other=f14, weights=W_EC, et_params=ET1,
                    lgb_params=dict(objective="tweedie", tweedie_variance_power=1.5, **LGBP),
                    shrink_lambda=SHRINK_L, clip=[lo, hi]),
        n_train=dict(sub_temp=len(lab_t), sub_ec=len(lab_e)),
        external_data="none", pretrained_models="none",
    )
    with open(os.path.join(OUT, "manifest_05.json"), "w", encoding="utf-8") as f:
        json.dump(man, f, ensure_ascii=False, indent=2)
    print("wrote manifest_05.json")


if __name__ == "__main__":
    main()
