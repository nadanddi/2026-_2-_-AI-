# -*- coding: utf-8 -*-
"""Raw CSVs -> temperature candidate G_C2 (HANDOFF §3, catalog 6.62/6.67/6.70/6.74/6.75).

    g    = clip((in_temp - 8) / 2, 0, 1)            (current hour, test input; NaN -> 1)
    pred = (0.6 - 0.2(1-g)) * BASE + (0.2 + 0.4(1-g)) * CODEX + 0.2 g * TABPFN

BASE    round-5 temperature recipe (0.65 physics-linear + LGB-huber residual,
        0.25 Ridge, 0.10 Nystroem; seeds 7/101/2024 averaged), round-5 row
        weights (train_flags_v6: restored rows 0.2, noisy days 0.2).
CODEX   Ridge physics baseline + LGB residual on resid_reset features
        (temp_mask_v1.codex_fit_predict), seeds 726/727 averaged.
TABPFN  TabPFN v2 regressor, CPU, n_estimators 4, context 2,000 training rows
        drawn with the round-5 weights, samples 1..8 (context seed = model
        seed = sample id), predictions averaged.

Rules (catalog 6.30, 10, CLAUDE.md):
  * training-row features are built in the MASK world (every test_X input NaN),
    so the fitted models never see test inputs;
  * test-row features use only the same greenhouse's current and earlier
    inputs (full history);
  * no label other than train_y, no external data at prediction time.

Output (NOT a submission): <OUTDIR>/temp_candidate_v8.csv with
  row_id, sub_temp, sub_ec  where sub_ec is copied from submission_04 (EC best,
  0.2055) only so the file is upload-shaped; members saved to
  temp_candidate_v8_members.npz; manifest temp_candidate_v8.json.
Refuses to overwrite existing outputs.

Run:  cd research && PYTHONPATH="" <python> -u make_submission_v8_temp.py [OUTDIR]
"""
import json
import os
import platform
import sys
import time

import env  # noqa: F401  MUST be the first project import
import env_extra  # noqa: F401
import numpy as np
import pandas as pd
import sklearn
import lightgbm as lgb
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer

import common
import harness
from harness import load
import feat_new
import features_v4 as F4
import train_flags_v6 as TF
from make_submission_v3 import SEEDS, T_HUB, m_lgbh, m_ridge, m_nys, sha256
from make_submission_v7 import fit_mean_w
from temp_mask_v1 import masked_loader, build_world, ORIG, codex_fit_predict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "analysis", "codex_independent", "2차"))
from resid_reset_features import build_features, FEATURE_COLUMNS, PHYSICS_COLUMNS  # noqa: E402

W_TEMP = dict(resid_lgb=0.65, ridge=0.25, nys=0.10)
W_FLAG, W_NOISY = 0.2, 0.2
CODEX_SEEDS = (726, 727)
PFN_SAMPLES = tuple(range(1, 9))
N_CTX = 2000
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "local", "v8")
NAME = "temp_candidate_v8"


def gate(in_temp):
    t = np.asarray(in_temp, float)
    return np.where(np.isnan(t), 1.0, np.clip((t - 8.0) / 2.0, 0.0, 1.0))


def combine(base, codex, pfn, g):
    return (0.6 - 0.2 * (1 - g)) * base + (0.2 + 0.4 * (1 - g)) * codex + 0.2 * g * pfn


def build_frames(loader=None):
    """Training frame (MASK world) and test frame (given world) with every column
    the three members need.  `loader` replaces common.load_raw for the test world
    (used by the causality check); default = the real data."""
    loader = loader or ORIG
    # MASK world -> training rows
    common.load_raw = lambda: masked_loader_from(loader)
    try:
        lab, ct, phc = build_world()
    finally:
        common.load_raw = ORIG
        harness._CACHE.clear()
    # test world -> test rows
    common.load_raw = loader
    harness._CACHE.clear()
    try:
        panel, _, _ = load()
        ex, _ = feat_new.build_extra()
        sg, ph, fp = F4.seg_features(), F4.phys_features(), F4.fp_features()
        tX, _, sX = loader()
        test = (panel[panel.is_test].reset_index(drop=True)
                .merge(ex, on="row_id", how="left").merge(sg, on="row_id", how="left")
                .merge(ph, on="row_id", how="left").merge(fp, on="row_id", how="left"))
        test = test.set_index("row_id").loc[sX.row_id].reset_index()
        CF_full = build_features(tX, sX).set_index("row_id")
    finally:
        common.load_raw = ORIG
        harness._CACHE.clear()
    tXm, _, sXm = masked_loader_from(loader)
    CF_mask = build_features(tXm, sXm).set_index("row_id")
    cx_cols = list(dict.fromkeys(list(FEATURE_COLUMNS) + list(PHYSICS_COLUMNS)))
    for c in cx_cols:
        lab["cx__" + c] = CF_mask.loc[lab.row_id, c].values
        test["cx__" + c] = CF_full.loc[test.row_id, c].values
    return lab, test, ct, phc, sX


def masked_loader_from(loader):
    tX, ty, sX = loader()
    sX = sX.copy()
    cols = [c for c in sX.columns if c not in ("row_id", "farm", "day", "hour", "t")]
    sX[cols] = np.nan
    return tX, ty, sX


def codex_frame(df):
    out = pd.DataFrame({c[4:]: df[c].values for c in df.columns if c.startswith("cx__")})
    out["sub_temp"] = df["sub_temp"].values if "sub_temp" in df else np.nan
    return out


def predict_members(lab, test, ct, phc, w, log=print):
    y = lab.sub_temp.values
    t0 = time.time()
    imp = SimpleImputer(strategy="median").fit(lab[phc])
    lin = LinearRegression().fit(imp.transform(lab[phc]), y, sample_weight=w)
    b_tr, b_te = lin.predict(imp.transform(lab[phc])), lin.predict(imp.transform(test[phc]))
    base = (W_TEMP["resid_lgb"] * (b_te + fit_mean_w(m_lgbh, lab[ct], y - b_tr, test[ct], w))
            + W_TEMP["ridge"] * fit_mean_w(m_ridge, lab[ct], y, test[ct], w, "ridge")
            + W_TEMP["nys"] * fit_mean_w(m_nys, lab[ct], y, test[ct], w, "ridge"))
    log("  base done (%.0fs)" % (time.time() - t0))

    ctr, cte = codex_frame(lab), codex_frame(test)
    codex = np.mean([codex_fit_predict(ctr, cte, w, s) for s in CODEX_SEEDS], axis=0)
    log("  codex done (%.0fs)" % (time.time() - t0))

    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion
    Xtr = ctr[list(FEATURE_COLUMNS)].values.astype(np.float32)
    Xte = cte[list(FEATURE_COLUMNS)].values.astype(np.float32)
    pfn_all = []
    for s in PFN_SAMPLES:
        rng = np.random.default_rng(s)
        idx = rng.choice(len(Xtr), size=min(N_CTX, len(Xtr)), replace=False, p=w / w.sum())
        m = TabPFNRegressor.create_default_for_version(ModelVersion.V2, device="cpu", n_estimators=4,
                                                       random_state=s, ignore_pretraining_limits=True)
        m.fit(Xtr[idx], y[idx])
        pfn_all.append(np.asarray(m.predict(Xte), float))
        log("  tabpfn sample %d done (%.0fs)" % (s, time.time() - t0))
    pfn_all = np.vstack(pfn_all)
    return base, codex, pfn_all


def main():
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, NAME + ".csv")
    npz = os.path.join(OUT, NAME + "_members.npz")
    mpath = os.path.join(OUT, NAME + ".json")
    for p in (path, npz, mpath):
        assert not os.path.exists(p), "refusing to overwrite %s" % p

    lab, test, ct, phc, sX = build_frames()
    w = TF.row_weights(lab, W_FLAG, w_noisy=W_NOISY)
    print("train rows %d (down-weighted %d) | test rows %d | base cols %d | codex cols %d"
          % (len(lab), int((w < 1).sum()), len(test), len(ct), len(FEATURE_COLUMNS)), flush=True)
    base, codex, pfn_all = predict_members(lab, test, ct, phc, w, log=lambda s: print(s, flush=True))
    pfn = pfn_all.mean(0)
    g = gate(test.in_temp.values)
    pt = combine(base, codex, pfn, g)

    sub04 = pd.read_csv(os.path.join(HERE, "submissions", "submission_04.csv"))
    sub06 = pd.read_csv(os.path.join(HERE, "submissions", "submission_06.csv"))
    assert sub04.row_id.tolist() == sX.row_id.tolist() == sub06.row_id.tolist()
    sub = pd.DataFrame({"row_id": sX.row_id.values, "sub_temp": pt, "sub_ec": sub04.sub_ec.values})
    assert list(sub.columns) == ["row_id", "sub_temp", "sub_ec"]
    assert len(sub) == 1440 and sub.row_id.is_unique
    assert np.isfinite(sub[["sub_temp", "sub_ec"]].to_numpy(float)).all()
    sub.to_csv(path, index=False, encoding="utf-8", float_format="%.6f")
    np.savez(npz, row_id=sX.row_id.values, base=base, codex=codex, pfn_samples=pfn_all, gate=g, pred=pt)

    d = pt - sub06.sub_temp.values
    print("\nwrote %s" % path)
    print("vs submission_06 temp: RMS change %.4f, mean %+.4f, min %+.3f, max %+.3f"
          % (np.sqrt((d ** 2).mean()), d.mean(), d.min(), d.max()))
    print("member means: base %.3f codex %.3f tabpfn %.3f | gate<1 rows %d (<=8C: %d)"
          % (base.mean(), codex.mean(), pfn.mean(), int((g < 1).sum()), int((g == 0).sum())))
    print("tabpfn sample spread (row-wise std, mean): %.4f" % pfn_all.std(0).mean())
    print("base vs round-5 temp RMS diff: %.4f (MASK world + same recipe)"
          % np.sqrt(((base - sub06.sub_temp.values) ** 2).mean()))

    man = dict(
        name=NAME, python=platform.python_version(),
        versions=dict(numpy=np.__version__, pandas=pd.__version__, sklearn=sklearn.__version__,
                      lightgbm=lgb.__version__),
        inputs={n: sha256(os.path.join(common.DATA, n)) for n in ["train_X.csv", "train_y.csv", "test_X.csv"]},
        output_sha256=sha256(path),
        formula="(0.6-0.2(1-g))*BASE + (0.2+0.4(1-g))*CODEX + 0.2g*TABPFN, g=clip((in_temp-8)/2,0,1)",
        base=dict(features=ct, phys_features=phc, member_weights=W_TEMP, lgb_params=T_HUB, seeds=list(SEEDS)),
        codex=dict(features=list(FEATURE_COLUMNS), physics=list(PHYSICS_COLUMNS), seeds=list(CODEX_SEEDS)),
        tabpfn=dict(version="v2", device="cpu", n_estimators=4, context_rows=N_CTX, samples=list(PFN_SAMPLES),
                    context_sampling="round-5 row weights"),
        training_row_weights=dict(restored_rows=W_FLAG, noisy_days=W_NOISY, n_downweighted=int((w < 1).sum())),
        training_features="MASK world (test_X inputs NaN)", test_features="full history, same greenhouse, <= current hour",
        sub_ec="copied from submission_04 (not produced here)",
        external_data="none", pretrained_models="TabPFN v2 regressor weights (Prior-Labs, Apache-2.0 with attribution)",
    )
    with open(mpath, "w", encoding="utf-8") as f:
        json.dump(man, f, ensure_ascii=False, indent=2)
    print("wrote %s" % os.path.basename(mpath))


if __name__ == "__main__":
    main()
